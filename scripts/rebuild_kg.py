"""Rebuild the full knowledge graph from all seeds with cross-topic edges.

Runs all seeds in a SINGLE build_from_seeds call so the cross-topic edge
discovery pass sees every topic group and can link any pair.  This produces
a genuinely interconnected graph instead of isolated per-topic cliques.

Seeds = original 6 + 8 new domains added in expand_kg.py.
"""

import asyncio
import json
import os
import sys
from src.utils.key_setup import ensure_keys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

from src.graph.graph_builder import GraphBuilder
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

ALL_SEEDS = [
    # Original seeds
    "Great Wall of China",
    "photosynthesis",
    "Claude Shannon information theory",
    "James Webb Space Telescope",
    "CRISPR gene editing",
    "2024 Nobel Prize in Physics",
    # New diverse seeds
    "mRNA vaccine technology",
    "lithium-ion battery chemistry",
    "transformer neural network architecture",
    "Silk Road trade routes",
    "Manhattan Project",
    "carbon credit market",
    "antibiotic resistance mechanisms",
    "black hole thermodynamics",
]

KG_PATH = "data/output/knowledge_graph.json"


async def main():
    print("=== Full KG Rebuild (cross-topic edges enabled) ===")
    print(f"Seeds: {len(ALL_SEEDS)}\n")

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

    builder = GraphBuilder(llm, registry)
    kg = await builder.build_from_seeds(
        ALL_SEEDS,
        expansion_rounds=2,
        max_entities_per_expansion=8,
        cross_topic_edges=True,
        cross_topic_sample=4,   # link each seed to 4 random other seeds
    )

    print(f"\nFinal KG: {kg.num_entities()} entities, {kg.num_relations()} relations")

    # Community analysis
    G_un = kg.graph.to_undirected()
    comps = list(nx.connected_components(G_un))
    print(f"Connected components: {len(comps)}")
    communities = nx.community.louvain_communities(G_un, seed=42)
    mod = nx.community.modularity(G_un, communities)
    within = sum(
        sum(1 for u, v in G_un.edges() if u in c and v in c)
        for c in communities
    )
    print(f"Louvain communities: {len(communities)}")
    print(f"Modularity: {mod:.3f}  (was 0.780 before)")
    print(f"Within-community edges: {within}/{G_un.number_of_edges()} "
          f"({100*within/G_un.number_of_edges():.0f}%)")

    os.makedirs("data/output", exist_ok=True)
    with open(KG_PATH, "w") as f:
        json.dump(kg.to_dict(), f, indent=2)
    print(f"\nSaved to {KG_PATH}")

    # Update ui_data.json
    ui_path = "data/output/ui_data.json"
    if os.path.exists(ui_path):
        with open(ui_path) as f:
            ui_data = json.load(f)
        ui_data["knowledge_graph"] = kg.to_dict()
        with open(ui_path, "w") as f:
            json.dump(ui_data, f, indent=2)
        print(f"Updated {ui_path}")


if __name__ == "__main__":
    ensure_keys()
    asyncio.run(main())
