"""Knowledge Graph — NetworkX-based entity-relation graph.

Paper 2, Section 3.4.1: 'constructing a highly interconnected knowledge
graph via random walks, leveraging web search to acquire relevant knowledge.'
"""

import random
from typing import Optional

import networkx as nx

from src.utils.data_models import Entity, Relation
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class KnowledgeGraph:
    """NetworkX-based knowledge graph for entity-relation storage."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []

    def add_entity(self, entity: Entity) -> None:
        """Add entity as node with attributes."""
        self._entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            attributes=entity.attributes,
        )

    def add_relation(self, relation: Relation) -> None:
        """Add relation as directed edge."""
        self._relations.append(relation)
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation_type=relation.relation_type,
            evidence=relation.evidence,
        )

    def get_entity(self, entity_id: str) -> Entity:
        """Get entity by ID."""
        if entity_id not in self._entities:
            raise KeyError(f"Entity {entity_id} not found")
        return self._entities[entity_id]

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get entity by name (case-insensitive)."""
        name_lower = name.lower()
        for entity in self._entities.values():
            if entity.name.lower() == name_lower:
                return entity
        return None

    def get_neighbors(self, entity_id: str, hops: int = 1) -> list[str]:
        """Get entity IDs reachable within `hops` steps."""
        if entity_id not in self.graph:
            return []

        visited = set()
        frontier = {entity_id}

        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.graph.successors(node):
                    if neighbor not in visited and neighbor != entity_id:
                        next_frontier.add(neighbor)
                for neighbor in self.graph.predecessors(node):
                    if neighbor not in visited and neighbor != entity_id:
                        next_frontier.add(neighbor)
            visited.update(frontier)
            frontier = next_frontier

        visited.update(frontier)
        visited.discard(entity_id)
        return list(visited)

    def random_walk(self, start_id: str, steps: int) -> list[str]:
        """Random walk from start node, return visited entity IDs."""
        if start_id not in self.graph:
            return [start_id]

        visited = [start_id]
        current = start_id

        for _ in range(steps):
            neighbors = list(self.graph.successors(current)) + list(
                self.graph.predecessors(current)
            )
            if not neighbors:
                break
            current = random.choice(neighbors)
            visited.append(current)

        return visited

    def get_subgraph(self, entity_ids: list[str]) -> "KnowledgeGraph":
        """Extract induced subgraph on given entity set."""
        sub = KnowledgeGraph()
        id_set = set(entity_ids)

        for eid in entity_ids:
            if eid in self._entities:
                sub.add_entity(self._entities[eid])

        for rel in self._relations:
            if rel.source_id in id_set and rel.target_id in id_set:
                sub.add_relation(rel)

        return sub

    def get_entities(self) -> list[Entity]:
        return list(self._entities.values())

    def get_relations(self) -> list[Relation]:
        return list(self._relations)

    def num_entities(self) -> int:
        return len(self._entities)

    def num_relations(self) -> int:
        return len(self._relations)

    def to_dict(self) -> dict:
        """Serialize to dict for checkpointing."""
        return {
            "entities": [e.model_dump() for e in self._entities.values()],
            "relations": [r.model_dump() for r in self._relations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        """Deserialize from dict."""
        kg = cls()
        for e_data in data.get("entities", []):
            kg.add_entity(Entity(**e_data))
        for r_data in data.get("relations", []):
            kg.add_relation(Relation(**r_data))
        return kg

    def describe_subgraph(self, entity_ids: list[str] | None = None) -> str:
        """Produce a human-readable description of (a subgraph of) the KG."""
        ids = set(entity_ids) if entity_ids else set(self._entities.keys())
        lines = ["Atomic facts:"]
        for eid in ids:
            if eid in self._entities:
                e = self._entities[eid]
                value = e.attributes.get("value", e.snippet) or ""
                src = f" [via {e.tool_source}]" if e.tool_source else ""
                lines.append(f"  - {e.name} ({e.entity_type}){src}: {value}")

        lines.append("\nRelations:")
        for rel in self._relations:
            if rel.source_id in ids and rel.target_id in ids:
                src = self._entities.get(rel.source_id)
                tgt = self._entities.get(rel.target_id)
                if src and tgt:
                    lines.append(
                        f"  - {src.name} --[{rel.relation_type}]--> {tgt.name}"
                    )
                    if rel.evidence:
                        lines.append(f"    Evidence: {rel.evidence}")

        return "\n".join(lines)
