"""Expand an existing knowledge graph with new seed topics.

Loads the current KG from data/output/knowledge_graph.json,
runs the graph builder on new seeds, and merges the results
back into the same file.  Skips seeds already present in the KG.

Usage:
    python scripts/expand_kg.py
"""

import asyncio
import json
import os
import sys
from src.utils.key_setup import ensure_keys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.graph_builder import GraphBuilder
from src.graph.knowledge_graph import KnowledgeGraph
from src.tools.calculator_tool import CalculatorTool
from src.tools.comparison_tool import ComparisonTool
from src.tools.scholar_tool import ScholarTool
from src.tools.search_tool import SearchTool
from src.tools.table_tool import TableTool
from src.tools.timeline_tool import TimelineTool
from src.tools.tool_registry import ToolRegistry
from src.tools.unit_converter_tool import UnitConverterTool
from src.tools.visit_tool import VisitTool
from src.tools.wikipedia_tool import WikipediaTool
from src.utils.llm_client import LLMClient

# ── New seeds to add ──────────────────────────────────────────────────────────
# Chosen to be:
#  - Obscure enough that the base LLM can't trivially answer (less parametric coverage)
#  - Cross-domain so subgraph bridging produces interesting multi-hop questions
#  - Factually dense (each topic has rich quantitative/temporal facts)
NEW_SEEDS = [
    # Science / engineering
    "mRNA vaccine technology",
    "lithium-ion battery chemistry",
    "transformer neural network architecture",
    # History / geography
    "Silk Road trade routes",
    "Manhattan Project",
    # Economics / policy
    "carbon credit market",
    # Biology / medicine
    "antibiotic resistance mechanisms",
    # Space / physics
    "black hole thermodynamics",
]

KG_PATH = "data/output/knowledge_graph.json"


async def main():
    print("=== KG Expansion ===\n")

    # Load existing KG
    if os.path.exists(KG_PATH):
        with open(KG_PATH) as f:
            kg = KnowledgeGraph.from_dict(json.load(f))
        print(f"Loaded existing KG: {kg.num_entities()} entities, {kg.num_relations()} relations")
    else:
        kg = KnowledgeGraph()
        print("No existing KG found, starting fresh")

    # Find which seeds are already covered
    existing_queries = {e.tool_query.lower() for e in kg.get_entities()}
    seeds_to_add = [s for s in NEW_SEEDS if s.lower() not in existing_queries]
    print(f"New seeds to add: {len(seeds_to_add)} / {len(NEW_SEEDS)}")
    for s in seeds_to_add:
        print(f"  + {s}")
    print()

    if not seeds_to_add:
        print("All seeds already present in KG, nothing to do.")
        return

    # Init tools
    llm = LLMClient("configs/llm_config.yaml")
    registry = ToolRegistry("configs/tool_config.yaml")
    cfg = registry.config
    registry.register(SearchTool(cfg.get("search", {})))
    registry.register(ScholarTool(cfg.get("scholar", {})))
    registry.register(WikipediaTool(cfg.get("wikipedia", {})))
    registry.register(CalculatorTool())
    registry.register(UnitConverterTool())
    registry.register(TimelineTool())
    registry.register(ComparisonTool(llm, cfg.get("comparison", {})))
    registry.register(VisitTool(llm, cfg.get("visit", {})))
    registry.register(TableTool(cfg.get("table", {})))

    # Build a fresh KG for the new seeds (with cross-topic edges enabled)
    builder = GraphBuilder(llm, registry)
    print("Building KG for new seeds (expansion_rounds=2, cross_topic_edges=True)...")
    new_kg = await builder.build_from_seeds(
        seeds_to_add,
        expansion_rounds=2,
        max_entities_per_expansion=8,
        cross_topic_edges=True,
        cross_topic_sample=3,
    )
    print(f"\nNew KG fragment: {new_kg.num_entities()} entities, {new_kg.num_relations()} relations")

    # Merge new KG into existing KG
    existing_entity_ids = {e.id for e in kg.get_entities()}
    added_entities = 0
    added_relations = 0

    for entity in new_kg.get_entities():
        if entity.id not in existing_entity_ids:
            kg.add_entity(entity)
            existing_entity_ids.add(entity.id)
            added_entities += 1

    existing_edges = {
        (r.source_id, r.target_id, r.relation_type)
        for r in kg.get_relations()
    }
    for relation in new_kg.get_relations():
        key = (relation.source_id, relation.target_id, relation.relation_type)
        if key not in existing_edges:
            # Only add if both endpoints exist in merged KG
            if relation.source_id in existing_entity_ids and relation.target_id in existing_entity_ids:
                kg.add_relation(relation)
                existing_edges.add(key)
                added_relations += 1

    print(f"\nMerged: +{added_entities} entities, +{added_relations} relations")
    print(f"Final KG: {kg.num_entities()} entities, {kg.num_relations()} relations")

    # Check connectivity
    import networkx as nx
    G_un = kg.graph.to_undirected()
    comps = list(nx.connected_components(G_un))
    print(f"Connected components: {len(comps)}")
    for i, c in enumerate(comps):
        print(f"  Component {i+1}: {len(c)} nodes")

    # Save
    os.makedirs("data/output", exist_ok=True)
    with open(KG_PATH, "w") as f:
        json.dump(kg.to_dict(), f, indent=2)
    print(f"\nSaved to {KG_PATH}")

    # Update ui_data.json KG section (preserves existing QA pairs)
    ui_path = "data/output/ui_data.json"
    if os.path.exists(ui_path):
        with open(ui_path) as f:
            ui_data = json.load(f)
        ui_data["knowledge_graph"] = kg.to_dict()
        with open(ui_path, "w") as f:
            json.dump(ui_data, f, indent=2)
        print(f"Updated {ui_path} (KG section only, QA pairs preserved)")


if __name__ == "__main__":
    ensure_keys()
    asyncio.run(main())
