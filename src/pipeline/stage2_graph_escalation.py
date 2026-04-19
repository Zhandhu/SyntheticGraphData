"""Stage 2 (Graph Track): Graph-aware Complexity Escalation.

Tongyi Paper 2 (arXiv:2510.24701v2), Section 3.4.1 — "High-quality Data Synthesis":
After generating initial QAs from subgraphs and applying uncertainty injection,
this stage iteratively escalates question complexity by expanding each question
to span a larger portion of the knowledge graph.

Key distinction from uncertainty injection:
  - Uncertainty injection: changes the *framing* of a question within its
    EXISTING subgraph (fuzzier entity references, temporal constraints, etc.)
  - Graph escalation: expands the *scope* by pulling in ADDITIONAL KG nodes,
    requiring multi-hop reasoning across larger graph regions

Escalation strategies (all graph-topology-driven):
  hop_extension         — one more reasoning hop in the graph
  subgraph_bridging     — force traversal between disconnected clusters
  multi_entity_aggregation — aggregate across all matching graph entities
  constraint_chain      — multi-entity filter chain narrows the answer space
  inverse_reasoning     — flip lookup direction, require graph-wide search
"""

import asyncio
import random
import re
from typing import Optional

from pydantic import BaseModel

from src.graph.knowledge_graph import KnowledgeGraph
from src.tools.tool_registry import ToolRegistry
from src.utils.data_models import QAPair, QADifficulty, ToolCallRecord, ToolType
from src.utils.llm_client import LLMClient
from src.utils.logging_utils import get_logger, log_stage_transition

logger = get_logger(__name__)

PROMPT_PATH = "prompts/graph_escalation.txt"

STRATEGIES = [
    "hop_extension",
    "subgraph_bridging",
    "multi_entity_aggregation",
    "constraint_chain",
    "inverse_reasoning",
]

# How many extension-candidate entities to surface per round
_MAX_CANDIDATES = 12
# Only use direct (1-hop) neighbours so every candidate has a guaranteed edge
# to the existing subgraph — avoids disconnected induced subgraphs.
_CANDIDATE_HOPS = 1


class GraphEscalationResponse(BaseModel):
    strategy: str
    reasoning: str
    question: str
    answer: str
    entities_incorporated: list[str] = []


class GraphEscalationResult(BaseModel):
    original: QAPair
    escalated: QAPair
    strategy: str
    reasoning: str
    tool_calls: list[ToolCallRecord] = []
    entities_incorporated: list[str] = []


class GraphComplexityEscalator:
    """Iteratively escalate graph-based QA pairs by expanding their required subgraph.

    For each round:
      1. Identify which KG entities the current question covers
         (stored in qa.source_chunk_ids after subgraph generation)
      2. Find candidate extension entities from the 1-hop neighbourhood
      3. Run a ReAct-style tool loop that verifies new facts and crafts a
         harder question spanning both the original and extension entities
      4. The escalated QA becomes the seed for the next round
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        config: dict,
        prompt_path: str = PROMPT_PATH,
    ):
        self.llm = llm_client
        self.tool_registry = tool_registry
        self.config = config
        with open(prompt_path) as f:
            self.prompt_template = f.read()

    # ── Public API ──────────────────────────────────────────────────────────

    async def escalate_single(
        self,
        qa: QAPair,
        kg: KnowledgeGraph,
        rounds: int = 2,
    ) -> QAPair:
        """Iteratively escalate one QA pair across `rounds` escalation steps.

        Each round's output is fed as the input for the next round, creating a
        virtuous cycle where each iteration produces a strictly harder question.
        """
        current = qa
        for i in range(rounds):
            try:
                result = await self._escalate_one_round(current, kg)
                escalated = result.escalated
                # Tag difficulty and record provenance
                level = i + 1
                escalated.difficulty = QADifficulty(f"escalated_{min(level, 3)}")
                escalated.escalation_history.append(current.question)
                # Merge entity IDs, including any KG connector nodes needed to
                # keep the induced subgraph fully connected (no islands).
                new_ids = self._resolve_entity_ids(kg, result.entities_incorporated)
                connector_ids = self._find_connector_nodes(
                    kg, current.source_chunk_ids, new_ids
                )
                escalated.source_chunk_ids = list(
                    dict.fromkeys(
                        current.source_chunk_ids + new_ids + connector_ids
                    )
                )
                current = escalated
            except Exception as e:
                logger.warning(f"Graph escalation round {i + 1} failed: {e}")
                break

        return current

    async def escalate_batch(
        self,
        qas: list[QAPair],
        kg: KnowledgeGraph,
        rounds: int = 2,
        max_concurrent: int = 5,
    ) -> list[QAPair]:
        """Escalate a batch with concurrency control."""
        log_stage_transition(
            logger, "stage2_graph_escalation", "started", count=len(qas)
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run(qa: QAPair) -> QAPair:
            async with semaphore:
                return await self.escalate_single(qa, kg, rounds=rounds)

        results = await asyncio.gather(
            *[_run(qa) for qa in qas], return_exceptions=True
        )

        escalated = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Graph escalation failed for QA {qas[i].id}: {result}"
                )
                escalated.append(qas[i])
            else:
                escalated.append(result)

        log_stage_transition(
            logger, "stage2_graph_escalation", "completed", count=len(escalated)
        )
        return escalated

    # ── Internal helpers ────────────────────────────────────────────────────

    async def _escalate_one_round(
        self,
        qa: QAPair,
        kg: KnowledgeGraph,
    ) -> GraphEscalationResult:
        """One escalation round: find candidates, run ReAct loop, return result."""
        max_tool_calls = self.config.get("max_tool_calls_per_round", 5)

        # 1. Describe entities the question currently requires
        current_entity_ids = [
            eid for eid in qa.source_chunk_ids if eid in {e.id for e in kg.get_entities()}
        ]
        current_entities_desc = self._describe_entities(kg, current_entity_ids)

        # 2. Find candidate extension entities (neighbours not already covered)
        candidates = self._find_extension_candidates(kg, current_entity_ids)
        candidates_desc = self._describe_entities(kg, [e.id for e in candidates])

        if not candidates:
            raise ValueError("No extension candidates available in the graph")

        # 3. Build the prompt and run the ReAct loop
        tool_descriptions = self.tool_registry.get_descriptions_for_prompt()
        system_msg = self.prompt_template.format(
            question=qa.question,
            answer=qa.answer,
            current_entities=current_entities_desc or "(not available)",
            extension_candidates=candidates_desc,
            tool_descriptions=tool_descriptions,
            max_tool_calls=max_tool_calls,
        )

        conversation = [{"role": "system", "content": system_msg}]
        tool_calls: list[ToolCallRecord] = []

        for _ in range(max_tool_calls + 1):
            response_text = await self.llm.complete(
                conversation, params_key="graph_escalation"
            )
            conversation.append({"role": "assistant", "content": response_text})

            tool_match = re.search(
                r"TOOL_CALL:\s*(\w+)\s*\|\s*(.+)", response_text
            )

            if tool_match and len(tool_calls) < max_tool_calls:
                tool_name = tool_match.group(1).strip().lower()
                tool_query = tool_match.group(2).strip()

                try:
                    result = await self.tool_registry.execute(
                        tool_name, query=tool_query
                    )
                    result_text = str(result.get("result", ""))[:2000]
                except Exception as e:
                    result_text = f"Tool error: {e}"

                tool_calls.append(ToolCallRecord(
                    tool=ToolType(tool_name),
                    query=tool_query,
                    result_summary=result_text[:500],
                ))
                conversation.append({
                    "role": "user",
                    "content": f"OBSERVATION: {result_text}",
                })
                continue

            # Try to parse the final JSON answer
            parsed = self._extract_response(response_text)
            if parsed:
                escalated_qa = QAPair(
                    question=parsed.question,
                    answer=parsed.answer,
                    domain=qa.domain,
                    source_chunk_ids=qa.source_chunk_ids,
                    source_subgraph_id=qa.source_subgraph_id,
                    required_tools=[tc.tool for tc in tool_calls],
                    metadata={
                        "strategy": parsed.strategy,
                        "entities_incorporated": parsed.entities_incorporated,
                        "graph_escalation": True,
                    },
                )
                return GraphEscalationResult(
                    original=qa,
                    escalated=escalated_qa,
                    strategy=parsed.strategy,
                    reasoning=parsed.reasoning,
                    tool_calls=tool_calls,
                    entities_incorporated=parsed.entities_incorporated,
                )

        logger.warning("Graph escalation loop exhausted without valid response")
        raise RuntimeError("Escalation loop exhausted without valid JSON response")

    def _find_extension_candidates(
        self,
        kg: KnowledgeGraph,
        current_ids: list[str],
    ):
        """Find KG entities adjacent to the current subgraph but not already in it.

        Returns up to _MAX_CANDIDATES entities ordered by graph connectivity
        (more connected entities make for richer extension opportunities).
        """
        covered = set(current_ids)
        candidate_ids: set[str] = set()

        for eid in current_ids:
            try:
                neighbors = kg.get_neighbors(eid, hops=_CANDIDATE_HOPS)
                candidate_ids.update(n for n in neighbors if n not in covered)
            except Exception:
                pass

        # If the graph is small, fall back to all uncovered entities
        if not candidate_ids:
            all_ids = {e.id for e in kg.get_entities()}
            candidate_ids = all_ids - covered

        # Sort by out-degree descending (prefer highly connected nodes)
        def degree(eid: str) -> int:
            try:
                return kg.graph.degree(eid)
            except Exception:
                return 0

        sorted_ids = sorted(candidate_ids, key=degree, reverse=True)

        # Return up to _MAX_CANDIDATES, with a random element from the tail
        # to avoid always picking the same high-degree nodes
        top = sorted_ids[:_MAX_CANDIDATES - 1]
        tail = sorted_ids[_MAX_CANDIDATES - 1:]
        if tail:
            top.append(random.choice(tail))

        return [kg.get_entity(eid) for eid in top if eid in {e.id for e in kg.get_entities()}]

    @staticmethod
    def _describe_entities(kg: KnowledgeGraph, entity_ids: list[str]) -> str:
        """Format a list of entities for the prompt."""
        lines = []
        for eid in entity_ids:
            try:
                e = kg.get_entity(eid)
                value = e.attributes.get("value", e.snippet) or ""
                src = f" [via {e.tool_source}: \"{e.tool_query}\"]" if e.tool_source else ""
                lines.append(f"  • [{e.name}] ({e.entity_type}){src}: {value}")
                # Surface key terms to help the LLM see connection points
                terms = e.attributes.get("key_terms", [])
                if terms:
                    lines.append(f"    key terms: {', '.join(terms)}")
            except KeyError:
                pass
        return "\n".join(lines)

    @staticmethod
    def _resolve_entity_ids(kg: KnowledgeGraph, names: list[str]) -> list[str]:
        """Map entity names back to IDs (best-effort, case-insensitive)."""
        ids = []
        for name in names:
            entity = kg.get_entity_by_name(name)
            if entity:
                ids.append(entity.id)
        return ids

    @staticmethod
    def _find_connector_nodes(
        kg: KnowledgeGraph,
        existing_ids: list[str],
        new_ids: list[str],
    ) -> list[str]:
        """Return intermediate KG nodes needed to keep the subgraph connected.

        When a new entity is added that is reachable from the existing subgraph
        only through intermediate nodes (e.g. it was a 2-hop candidate despite
        _CANDIDATE_HOPS=1 via the fallback path, or came from a name-resolved
        entity with indirect connections), we walk the shortest undirected path
        in the full KG and add any missing intermediate nodes.
        """
        import networkx as nx

        G_un = kg.graph.to_undirected()
        existing_set = set(existing_ids)
        connector_ids: list[str] = []

        for nid in new_ids:
            if nid not in G_un:
                continue
            # Check if nid already has a direct edge to any existing node
            direct = any(G_un.has_edge(nid, eid) for eid in existing_set)
            if direct:
                continue
            # Find shortest path from nid to any existing node
            try:
                path = nx.shortest_path(G_un, nid,
                                        next(e for e in existing_set if e in G_un))
                # Add intermediates (exclude endpoints — they're already tracked)
                for intermediate in path[1:-1]:
                    if intermediate not in existing_set and intermediate not in new_ids:
                        connector_ids.append(intermediate)
                        existing_set.add(intermediate)  # prevent duplicates
            except (nx.NetworkXNoPath, nx.NodeNotFound, StopIteration):
                pass

        return connector_ids

    @staticmethod
    def _extract_response(text: str) -> Optional[GraphEscalationResponse]:
        """Parse GraphEscalationResponse JSON from LLM output."""
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return GraphEscalationResponse.model_validate_json(match.group(1))
            except Exception:
                pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return GraphEscalationResponse.model_validate_json(match.group(0))
            except Exception:
                pass

        return None
