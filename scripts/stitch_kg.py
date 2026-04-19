"""Stitch disconnected KG components with cross-domain LLM-discovered edges.

For each pair of disconnected components, picks representative fact nodes and
asks the LLM to find meaningful cross-domain relationships between them.
Writes the enriched KG back to data/output/knowledge_graph.json.
"""

import asyncio
import json
import os
import random
import sys
from src.utils.key_setup import ensure_keys

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.graph_builder import EDGE_DISCOVERY_PROMPT, EdgeResponse
from src.graph.knowledge_graph import KnowledgeGraph
from src.utils.data_models import Relation
from src.utils.llm_client import LLMClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

KG_PATH = "data/output/knowledge_graph.json"
# Edges to request per component pair
EDGES_PER_PAIR = 4
# How many representative nodes to show the LLM per component
NODES_PER_COMPONENT = 6


async def stitch_components(kg: KnowledgeGraph, llm: LLMClient) -> int:
    """Discover and add cross-domain edges between all disconnected components.
    Returns total edges added."""

    G_un = kg.graph.to_undirected()
    comps = list(nx.connected_components(G_un))
    print(f"Components before stitching: {len(comps)}")
    if len(comps) == 1:
        print("Already connected, nothing to do.")
        return 0

    # Sort components by size descending so we anchor on the biggest ones
    comps = sorted(comps, key=len, reverse=True)

    total_added = 0

    # Run edge discovery between every pair of components
    for i, comp_a in enumerate(comps):
        for comp_b in comps[i + 1:]:
            added = await _discover_cross_edges(kg, llm, list(comp_a), list(comp_b))
            total_added += added
            print(f"  C{i+1}({len(comp_a)}) ↔ C{i+2+comps[i+1:].index(comp_b)}({len(comp_b)}): +{added} edges")

    return total_added


async def _discover_cross_edges(
    kg: KnowledgeGraph,
    llm: LLMClient,
    ids_a: list[str],
    ids_b: list[str],
) -> int:
    """Find cross-domain edges between two node sets."""

    # Pick representative nodes: prefer high-degree, add random diversity
    def pick_reps(ids: list[str]) -> list[str]:
        by_degree = sorted(ids, key=lambda x: kg.graph.degree(x), reverse=True)
        top = by_degree[:NODES_PER_COMPONENT - 1]
        tail = by_degree[NODES_PER_COMPONENT - 1:]
        if tail:
            top.append(random.choice(tail))
        return top

    reps_a = pick_reps(ids_a)
    reps_b = pick_reps(ids_b)

    def fmt(ids: list[str]) -> str:
        lines = []
        for eid in ids:
            try:
                e = kg.get_entity(eid)
                val = e.attributes.get("value", e.snippet) or ""
                lines.append(f"- [{e.name}]: {val[:200]}")
            except KeyError:
                pass
        return "\n".join(lines)

    # Infer domain labels from tool_query
    domain_a = kg.get_entity(reps_a[0]).tool_query if reps_a else "unknown"
    domain_b = kg.get_entity(reps_b[0]).tool_query if reps_b else "unknown"

    prompt = EDGE_DISCOVERY_PROMPT.format(
        tool_a="cross-domain",
        query_a=domain_a,
        facts_a=fmt(reps_a),
        tool_b="cross-domain",
        query_b=domain_b,
        facts_b=fmt(reps_b),
        max_edges=EDGES_PER_PAIR,
    )

    try:
        response = await llm.complete_json(
            [{"role": "user", "content": prompt}], EdgeResponse
        )
    except Exception as e:
        logger.warning(f"Edge discovery failed: {e}")
        return 0

    name_to_id_a = {kg.get_entity(eid).name: eid for eid in reps_a}
    name_to_id_b = {kg.get_entity(eid).name: eid for eid in reps_b}

    added = 0
    for edge in response.edges[:EDGES_PER_PAIR]:
        src_id = name_to_id_a.get(edge.source_name)
        tgt_id = name_to_id_b.get(edge.target_name)
        if src_id and tgt_id:
            kg.add_relation(Relation(
                source_id=src_id,
                target_id=tgt_id,
                relation_type=edge.relation,
                evidence=f"cross-domain: {domain_a} ↔ {domain_b}",
            ))
            added += 1

    return added


async def main():
    print("=== KG Component Stitching ===\n")

    with open(KG_PATH) as f:
        kg = KnowledgeGraph.from_dict(json.load(f))
    print(f"Loaded KG: {kg.num_entities()} entities, {kg.num_relations()} relations")

    llm = LLMClient("configs/llm_config.yaml")

    total = await stitch_components(kg, llm)
    print(f"\nTotal cross-domain edges added: {total}")
    print(f"Final KG: {kg.num_entities()} entities, {kg.num_relations()} relations")

    # Verify
    G_un = kg.graph.to_undirected()
    comps = list(nx.connected_components(G_un))
    print(f"Connected components after stitching: {len(comps)}")

    with open(KG_PATH, "w") as f:
        json.dump(kg.to_dict(), f, indent=2)
    print(f"Saved to {KG_PATH}")

    # Update ui_data
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
