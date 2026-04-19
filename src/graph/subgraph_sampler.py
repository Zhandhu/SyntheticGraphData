"""Subgraph Sampler — samples subgraphs for QA generation.

Paper 2, Section 3.4.1: 'We then sample subgraphs and subtables to
generate initial questions and answers.'
"""

import random

import networkx as nx
from pydantic import BaseModel

from src.graph.knowledge_graph import KnowledgeGraph
from src.utils.data_models import QAPair, SubgraphSample
from src.utils.llm_client import LLMClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

GRAPH_QA_PROMPT_PATH = "prompts/graph_qa_generator.txt"


class GraphQAResponse(BaseModel):
    question: str
    answer: str
    reasoning: str
    entities_required: list[str] = []


class SubgraphSampler:
    """Sample subgraphs and generate QA pairs from them."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def sample_by_random_walk(
        self,
        min_entities: int = 3,
        max_entities: int = 8,
        walk_length: int = 10,
    ) -> SubgraphSample:
        """Start from a random entity, random walk to collect entities."""
        entities = self.kg.get_entities()
        if not entities:
            raise ValueError("Knowledge graph has no entities")

        start = random.choice(entities)
        walked_ids = self.kg.random_walk(start.id, walk_length)

        # Deduplicate while preserving order
        seen = set()
        unique_ids = []
        for eid in walked_ids:
            if eid not in seen:
                seen.add(eid)
                unique_ids.append(eid)

        # Trim to max_entities
        if len(unique_ids) > max_entities:
            unique_ids = unique_ids[:max_entities]

        # Ensure minimum
        if len(unique_ids) < min_entities:
            # Add random neighbors
            all_ids = [e.id for e in entities]
            for eid in random.sample(all_ids, min(len(all_ids), min_entities * 2)):
                if eid not in seen:
                    unique_ids.append(eid)
                    seen.add(eid)
                if len(unique_ids) >= min_entities:
                    break

        # Stitch any disconnected components (can occur when random neighbors
        # are added to meet min_entities and they don't lie on the walk path)
        unique_ids = self._ensure_connected(unique_ids)

        return SubgraphSample(
            entity_ids=unique_ids,
            seed_entity=start.name,
            hops=walk_length,
        )

    def sample_bridging(
        self,
        n_components: int = 2,
        entities_per_component: int = 3,
    ) -> SubgraphSample:
        """Sample entities from n different neighborhoods.

        Forces QA pairs that require connecting information across
        distant parts of the graph.
        """
        entities = self.kg.get_entities()
        if len(entities) < n_components:
            raise ValueError(
                f"Not enough entities ({len(entities)}) for "
                f"{n_components} components"
            )

        # Pick n_components random seed entities
        seeds = random.sample(entities, n_components)
        all_ids = []

        for seed in seeds:
            # Get local neighborhood
            neighbors = self.kg.get_neighbors(seed.id, hops=2)
            local = [seed.id] + neighbors[:entities_per_component - 1]
            all_ids.extend(local)

        # Deduplicate
        seen = set()
        unique_ids = []
        for eid in all_ids:
            if eid not in seen:
                seen.add(eid)
                unique_ids.append(eid)

        # Stitch disconnected components: add shortest-path connector nodes
        unique_ids = self._ensure_connected(unique_ids)

        return SubgraphSample(
            entity_ids=unique_ids,
            seed_entity=seeds[0].name,
            hops=2,
        )

    def _ensure_connected(self, entity_ids: list[str]) -> list[str]:
        """Add connector nodes so the induced subgraph is fully connected.

        For each pair of disconnected components, finds the shortest undirected
        path in the full KG and adds intermediate nodes to the sample. If no
        path exists between two components (disconnected KG), leaves them as-is
        — the QA generator will still work, just with a richer multi-island context.
        """
        if len(entity_ids) < 2:
            return entity_ids

        G_un = self.kg.graph.to_undirected()
        node_set = set(entity_ids)
        Gsub = G_un.subgraph(node_set)
        components = list(nx.connected_components(Gsub))

        if len(components) == 1:
            return entity_ids  # already connected

        # Greedily stitch: pick a representative from each component pair and
        # find the shortest path in the full KG, adding intermediates.
        result_ids = list(entity_ids)
        result_set = set(result_ids)
        # Start with the largest component as the anchor
        components.sort(key=len, reverse=True)
        anchor_nodes = list(components[0])

        for comp in components[1:]:
            comp_nodes = list(comp)
            best_path: list[str] = []
            best_len = float("inf")

            # Try a sample of pairs to keep this fast
            sample_a = anchor_nodes[:3]
            sample_b = comp_nodes[:3]
            for u in sample_a:
                for v in sample_b:
                    try:
                        path = nx.shortest_path(G_un, u, v)
                        if len(path) < best_len:
                            best_len = len(path)
                            best_path = path
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

            # Add intermediates (not the endpoints — they're already included)
            for node in best_path[1:-1]:
                if node not in result_set:
                    result_ids.append(node)
                    result_set.add(node)
            # Expand anchor for the next iteration
            anchor_nodes.extend(comp_nodes + best_path)

        return result_ids

    async def generate_qa_from_subgraph(
        self,
        sample: SubgraphSample,
        llm_client: LLMClient,
    ) -> QAPair:
        """Given a subgraph, generate a QA pair requiring multiple entities."""
        # Build description of the subgraph
        subgraph_desc = self.kg.describe_subgraph(sample.entity_ids)

        # Load prompt template
        with open(GRAPH_QA_PROMPT_PATH) as f:
            prompt_template = f.read()

        prompt = prompt_template.format(subgraph_description=subgraph_desc)
        messages = [{"role": "user", "content": prompt}]

        response = await llm_client.complete_json(messages, GraphQAResponse)

        return QAPair(
            question=response.question,
            answer=response.answer,
            source_subgraph_id=sample.id,
            source_chunk_ids=sample.entity_ids,
            metadata={
                "reasoning": response.reasoning,
                "entities_required": response.entities_required,
            },
        )
