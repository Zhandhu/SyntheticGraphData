"""Graph Builder — constructs a *nuclear* (atomic) knowledge graph.

Every node is a single indivisible fact from exactly one tool lookup.
Edges connect facts that relate to one another (supports, elaborates,
quantifies, causes, etc.).

Paper 2, Section 3.3.2: 'Grounded in continuously updated open-world
knowledge, we construct an entity-anchored open-world memory.'
"""

import asyncio
import re
from typing import Optional

from pydantic import BaseModel

from src.graph.knowledge_graph import KnowledgeGraph
from src.tools.tool_registry import ToolRegistry
from src.utils.data_models import Entity, Relation
from src.utils.llm_client import LLMClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ── LLM response schemas ────────────────────────────────────────────

class AtomicFact(BaseModel):
    """One indivisible fact extracted from a single tool result."""
    name: str           # short label (< 10 words)
    fact_type: str      # fact | quantity | date | definition | person | place | event
    value: str          # the atomic fact sentence
    key_terms: list[str] = []   # terms that might link to other facts


class FactExtractionResponse(BaseModel):
    facts: list[AtomicFact] = []


class EdgeSuggestion(BaseModel):
    source_name: str
    target_name: str
    relation: str       # supports | elaborates | quantifies | causes | contradicts | temporal | same_topic


class EdgeResponse(BaseModel):
    edges: list[EdgeSuggestion] = []


# ── Prompts ──────────────────────────────────────────────────────────

FACT_EXTRACTION_PROMPT = """You are a fact atomiser. Given the raw result of a single tool lookup,
break it into the smallest possible independent facts. Each fact must:
- Be a SINGLE statement that cannot be split further
- Be self-contained (understandable without the other facts)
- Carry exactly one piece of information (one number, one date, one relationship, etc.)

Tool used: {tool_name}
Query: {query}
Result:
{result}

Return JSON:
{{
  "facts": [
    {{"name": "short label", "fact_type": "fact|quantity|date|definition|person|place|event", "value": "the atomic fact sentence", "key_terms": ["term1", "term2"]}}
  ]
}}

Extract up to {max_facts} facts. Prefer the most informative and specific ones."""

EDGE_DISCOVERY_PROMPT = """You are given two groups of atomic facts from different tool lookups.
Identify meaningful relationships between facts across the two groups.

Group A (from {tool_a}: "{query_a}"):
{facts_a}

Group B (from {tool_b}: "{query_b}"):
{facts_b}

For each pair of related facts, specify the relationship type:
- supports: fact B provides evidence for fact A
- elaborates: fact B adds detail to fact A
- quantifies: fact B adds a number/measurement to fact A
- causes: fact A causes or leads to fact B
- contradicts: the facts conflict
- temporal: the facts are linked by time ordering
- same_topic: the facts are about the same subject

Return JSON:
{{
  "edges": [
    {{"source_name": "label from A", "target_name": "label from B", "relation": "relation_type"}}
  ]
}}

Only include edges where there is a clear, meaningful connection. Up to {max_edges} edges."""


class GraphBuilder:
    """Build nuclear (atomic) knowledge graphs where every node = one tool lookup fact."""

    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry):
        self.llm = llm_client
        self.tool_registry = tool_registry

    # Default facts-per-lookup cap per tool.
    # search gets a higher cap because its path visits full pages (much more
    # content than a snippet), so there is genuinely more to atomize.
    _DEFAULT_TOOL_FACT_CAPS: dict[str, int] = {
        "search":    8,   # search → visit full pages → richer content
        "wikipedia": 4,   # encyclopedic article sections
        "scholar":   3,   # abstract-length academic snippets
    }
    # How many search result URLs to visit per query
    _SEARCH_VISIT_URLS: int = 3

    async def build_from_seeds(
        self,
        seed_entities: list[str],
        expansion_rounds: int = 2,
        max_entities_per_expansion: int = 10,
        tools_per_seed: list[str] | None = None,
        max_facts_per_lookup: int = 6,
        tool_fact_caps: dict[str, int] | None = None,
        cross_topic_edges: bool = True,
        cross_topic_sample: int = 3,
    ) -> KnowledgeGraph:
        """Build an atomic KG.

        For each seed topic:
          1. Run multiple tool lookups (search, wikipedia, scholar)
          2. Atomise each tool result into individual fact nodes
          3. Discover edges between facts within and across lookups
          4. On subsequent rounds, pick key_terms from existing facts
             as new seeds to expand the frontier

        After all rounds, if cross_topic_edges=True, run a second edge-discovery
        pass that cross-links representative groups from *different* seed topics.
        This prevents the KG from collapsing into isolated per-topic cliques.

        tool_fact_caps overrides the per-tool fact limits. Falls back to
        _DEFAULT_TOOL_FACT_CAPS, then max_facts_per_lookup.
        """
        kg = KnowledgeGraph()
        tools = tools_per_seed or self._pick_tools()
        caps = {**self._DEFAULT_TOOL_FACT_CAPS, **(tool_fact_caps or {})}

        frontier: list[str] = list(seed_entities)
        expanded: set[str] = set()

        # Track one representative group per seed topic for cross-topic linking
        # key: seed topic (lowercased), value: one group dict with most nodes
        seed_representative_groups: dict[str, dict] = {}
        # Map each topic's expansion terms back to its original seed
        term_to_seed: dict[str, str] = {s.lower(): s for s in seed_entities}

        for round_num in range(expansion_rounds):
            logger.info(
                f"Expansion round {round_num + 1}/{expansion_rounds}, "
                f"frontier size: {len(frontier)}"
            )

            # Collect all lookup groups for this round so we can cross-link
            round_groups: list[dict] = []  # {tool, query, entity_ids, seed_topic}

            for topic in frontier:
                if topic.lower() in expanded:
                    continue
                expanded.add(topic.lower())

                # Resolve which original seed this topic belongs to
                seed_topic = term_to_seed.get(topic.lower(), topic)

                for tool_name in tools:
                    tool_cap = caps.get(tool_name, max_facts_per_lookup)
                    # Small pause before each search call to avoid rate-limits
                    if tool_name == "search":
                        await asyncio.sleep(2)
                    try:
                        if tool_name == "search" and "visit" in self.tool_registry.get_tool_names():
                            entity_ids = await self._lookup_and_atomise_search(
                                kg, topic, tool_cap
                            )
                        else:
                            entity_ids = await self._lookup_and_atomise(
                                kg, tool_name, topic, tool_cap
                            )
                        if entity_ids:
                            group = {
                                "tool": tool_name,
                                "query": topic,
                                "entity_ids": entity_ids,
                                "seed_topic": seed_topic,
                            }
                            round_groups.append(group)
                            # Keep the largest group per seed as representative
                            key = seed_topic.lower()
                            if (key not in seed_representative_groups or
                                    len(entity_ids) > len(seed_representative_groups[key]["entity_ids"])):
                                seed_representative_groups[key] = group
                    except Exception as e:
                        logger.warning(f"Lookup {tool_name}('{topic}') failed: {e}")

            # Discover edges between same-round groups (intra-topic AND cross-topic)
            for i, ga in enumerate(round_groups):
                for gb in round_groups[i + 1:]:
                    try:
                        await self._discover_edges(kg, ga, gb)
                    except Exception as e:
                        logger.warning(f"Edge discovery failed: {e}")

            # Build next frontier from key_terms on newest nodes
            new_terms: set[str] = set()
            for group in round_groups:
                for eid in group["entity_ids"]:
                    entity = kg.get_entity(eid)
                    for term in entity.attributes.get("key_terms", []):
                        t_lower = term.lower()
                        if t_lower not in expanded:
                            new_terms.add(term)
                            # Track which seed this expansion term came from
                            if t_lower not in term_to_seed:
                                term_to_seed[t_lower] = group["seed_topic"]

            frontier = list(new_terms)[:max_entities_per_expansion]
            if not frontier:
                logger.info("No new terms to expand, stopping early")
                break

        # ── Cross-topic edge discovery ──────────────────────────────────────
        # After all expansion rounds, link representative groups from DIFFERENT
        # seed topics. This is the key step that breaks up per-topic cliques and
        # produces a genuinely interconnected graph.
        if cross_topic_edges and len(seed_representative_groups) > 1:
            logger.info(
                f"Cross-topic edge discovery across "
                f"{len(seed_representative_groups)} seed topics..."
            )
            rep_list = list(seed_representative_groups.values())
            import random
            cross_added = 0
            for i, ga in enumerate(rep_list):
                # Don't link every pair — sample `cross_topic_sample` partners
                # per group to keep API calls bounded while still achieving
                # broad connectivity
                partners = rep_list[i + 1:]
                if len(partners) > cross_topic_sample:
                    partners = random.sample(partners, cross_topic_sample)
                for gb in partners:
                    if ga["seed_topic"] == gb["seed_topic"]:
                        continue
                    try:
                        before = kg.num_relations()
                        await self._discover_edges(kg, ga, gb, max_edges=3)
                        cross_added += kg.num_relations() - before
                    except Exception as e:
                        logger.warning(f"Cross-topic edge discovery failed: {e}")
            logger.info(f"  Cross-topic pass added {cross_added} edges")

        logger.info(
            f"Built atomic KG: {kg.num_entities()} fact-nodes, "
            f"{kg.num_relations()} edges"
        )
        return kg

    async def _lookup_and_atomise(
        self,
        kg: KnowledgeGraph,
        tool_name: str,
        query: str,
        max_facts: int,
    ) -> list[str]:
        """Run one tool call, atomise the result into fact-nodes, return their IDs."""
        result = await self.tool_registry.execute(tool_name, query=query)
        result_text = self._result_to_text(result)

        if not result_text or len(result_text.strip()) < 20:
            return []

        # Ask LLM to break into atomic facts
        prompt = FACT_EXTRACTION_PROMPT.format(
            tool_name=tool_name,
            query=query,
            result=result_text[:4000],
            max_facts=max_facts,
        )
        response = await self.llm.complete_json(
            [{"role": "user", "content": prompt}], FactExtractionResponse
        )

        entity_ids = []
        for fact in response.facts[:max_facts]:
            entity = Entity(
                name=fact.name,
                entity_type=fact.fact_type,
                attributes={
                    "value": fact.value,
                    "key_terms": fact.key_terms,
                },
                tool_source=tool_name,
                tool_query=query,
                snippet=fact.value,
            )
            kg.add_entity(entity)
            entity_ids.append(entity.id)

        # Add edges between facts from the SAME lookup (intra-group).
        # All nodes from one tool call share the same query context, so connect
        # them all with same_topic edges — this guarantees every lookup group
        # forms a connected component regardless of key_term overlap.
        for i, eid_a in enumerate(entity_ids):
            for eid_b in entity_ids[i + 1:]:
                ea = kg.get_entity(eid_a)
                eb = kg.get_entity(eid_b)
                terms_a = set(ea.attributes.get("key_terms", []))
                terms_b = set(eb.attributes.get("key_terms", []))
                shared = terms_a & terms_b
                # Always connect same-lookup nodes; note shared terms if any
                kg.add_relation(Relation(
                    source_id=eid_a,
                    target_id=eid_b,
                    relation_type="same_topic",
                    evidence=(
                        f"Shared terms: {', '.join(shared)}"
                        if shared
                        else f"Same lookup: {tool_name}('{query}')"
                    ),
                ))

        logger.info(f"  {tool_name}('{query}') -> {len(entity_ids)} atomic facts")
        return entity_ids

    async def _lookup_and_atomise_search(
        self,
        kg: KnowledgeGraph,
        query: str,
        max_facts: int,
    ) -> list[str]:
        """Search → visit top URLs → atomise full page content into fact-nodes.

        This two-step retrieval produces far richer content than snippet-only
        search, making 'search' competitive with scholar/wikipedia in fact yield
        while remaining fully externally grounded.
        """
        # Step 1: get search results (title + snippet + URL).
        # DuckDuckGo rate-limits on rapid successive calls — retry with backoff.
        raw = []
        for attempt in range(3):
            if attempt:
                await asyncio.sleep(3 * attempt)
            search_result = await self.tool_registry.execute("search", query=query)
            raw = search_result.get("result", [])
            if not isinstance(raw, list):
                raw = []
            if raw:
                break

        # Filter to results that have a URL and a non-trivial snippet
        candidates = [r for r in raw if r.get("url") and r.get("snippet")]
        urls_to_visit = [r["url"] for r in candidates[:self._SEARCH_VISIT_URLS]]

        if not candidates:
            # DDG rate-limited or returned nothing — skip silently, no fallback
            # (snippet-only would produce near-zero facts anyway)
            logger.warning(f"  search('{query}') returned no usable results, skipping")
            return []

        # Step 2: visit pages + extract tables concurrently
        async def visit_url(url: str) -> str:
            try:
                result = await self.tool_registry.execute(
                    "visit", url=url, goal=f"Extract facts about: {query}"
                )
                return str(result.get("result", ""))[:3000]
            except Exception as e:
                logger.warning(f"  visit({url}) failed: {e}")
                return ""

        async def extract_tables(url: str) -> str:
            if "table" not in self.tool_registry.get_tool_names():
                return ""
            try:
                result = await self.tool_registry.execute(
                    "table", url=url, topic=query
                )
                return result.get("text", "")[:2000]
            except Exception:
                return ""

        page_contents = await asyncio.gather(*[visit_url(u) for u in urls_to_visit])
        table_contents = await asyncio.gather(*[extract_tables(u) for u in urls_to_visit])

        # Step 3: build combined content — snippets first, then pages, then tables
        snippet_text = "\n".join(
            f"[{r['title']}] {r['snippet']}" for r in candidates[:self._SEARCH_VISIT_URLS]
        )
        page_text = "\n\n---\n\n".join(c for c in page_contents if c.strip())
        table_text = "\n\n".join(t for t in table_contents if t.strip())
        combined = "\n\n".join(filter(None, [snippet_text, page_text, table_text])).strip()

        if not combined or len(combined) < 20:
            return []

        # Step 4: atomize (label source as "search+visit" for transparency)
        prompt = FACT_EXTRACTION_PROMPT.format(
            tool_name="search+visit",
            query=query,
            result=combined[:5000],
            max_facts=max_facts,
        )
        response = await self.llm.complete_json(
            [{"role": "user", "content": prompt}], FactExtractionResponse
        )

        entity_ids = []
        for fact in response.facts[:max_facts]:
            entity = Entity(
                name=fact.name,
                entity_type=fact.fact_type,
                attributes={"value": fact.value, "key_terms": fact.key_terms},
                tool_source="search+visit",
                tool_query=query,
                snippet=fact.value,
            )
            kg.add_entity(entity)
            entity_ids.append(entity.id)

        # Intra-group edges
        for i, eid_a in enumerate(entity_ids):
            for eid_b in entity_ids[i + 1:]:
                ea = kg.get_entity(eid_a)
                eb = kg.get_entity(eid_b)
                terms_a = set(ea.attributes.get("key_terms", []))
                terms_b = set(eb.attributes.get("key_terms", []))
                shared = terms_a & terms_b
                kg.add_relation(Relation(
                    source_id=eid_a,
                    target_id=eid_b,
                    relation_type="same_topic",
                    evidence=(
                        f"Shared terms: {', '.join(shared)}"
                        if shared
                        else f"Same search+visit lookup: '{query}'"
                    ),
                ))

        logger.info(
            f"  search+visit('{query}') -> {len(entity_ids)} atomic facts "
            f"from {len([c for c in page_contents if c.strip()])} pages"
        )
        return entity_ids

    async def _discover_edges(
        self,
        kg: KnowledgeGraph,
        group_a: dict,
        group_b: dict,
        max_edges: int = 5,
    ) -> None:
        """Ask LLM to find edges between two groups of facts."""
        facts_a_text = self._format_fact_group(kg, group_a["entity_ids"])
        facts_b_text = self._format_fact_group(kg, group_b["entity_ids"])

        if not facts_a_text or not facts_b_text:
            return

        prompt = EDGE_DISCOVERY_PROMPT.format(
            tool_a=group_a["tool"],
            query_a=group_a["query"],
            facts_a=facts_a_text,
            tool_b=group_b["tool"],
            query_b=group_b["query"],
            facts_b=facts_b_text,
            max_edges=max_edges,
        )
        response = await self.llm.complete_json(
            [{"role": "user", "content": prompt}], EdgeResponse
        )

        # Resolve names → IDs
        name_to_id_a = {
            kg.get_entity(eid).name: eid for eid in group_a["entity_ids"]
        }
        name_to_id_b = {
            kg.get_entity(eid).name: eid for eid in group_b["entity_ids"]
        }

        for edge in response.edges[:max_edges]:
            src_id = name_to_id_a.get(edge.source_name)
            tgt_id = name_to_id_b.get(edge.target_name)
            if src_id and tgt_id:
                kg.add_relation(Relation(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=edge.relation,
                ))

    def _pick_tools(self) -> list[str]:
        """Pick which tools to use for each seed topic.

        KG construction tools:
          search     — triggers search→visit chain: DuckDuckGo URLs + full page
                       content fetch, producing the richest and most diverse facts
          wikipedia  — encyclopedic article sections
          scholar    — academic abstracts and citations

        Excluded from KG building:
          comparison — LLM self-knowledge, not externally retrieved
          visit      — used internally by the search→visit chain, not called directly
          timeline, calculator, unit_converter, python, knowledge_base,
          document_reader — computational/local-only, need specific inputs not topics
        """
        available = self.tool_registry.get_tool_names()
        preferred = ["search", "wikipedia", "scholar"]
        return [t for t in preferred if t in available] or available[:2]

    @staticmethod
    def _result_to_text(result: dict) -> str:
        """Convert a tool result dict to a flat string."""
        raw = result.get("result", "")
        if isinstance(raw, list):
            lines = []
            for i, item in enumerate(raw[:10], 1):
                if isinstance(item, dict):
                    parts = [f"{k}: {v}" for k, v in item.items() if v]
                    lines.append(f"{i}. {' | '.join(parts)}")
                else:
                    lines.append(f"{i}. {item}")
            return "\n".join(lines)
        return str(raw)

    @staticmethod
    def _format_fact_group(kg: KnowledgeGraph, entity_ids: list[str]) -> str:
        """Format a group of fact-nodes for the LLM prompt."""
        lines = []
        for eid in entity_ids:
            e = kg.get_entity(eid)
            lines.append(f"- [{e.name}]: {e.attributes.get('value', e.snippet)}")
        return "\n".join(lines)
