"""Generate 10 sample QA datapoints using the graph-based pipeline (Track B).

Smaller scale: pick a few seed entities, build a small KG, sample subgraphs,
generate QA pairs, and apply uncertainty injection. Skips the full quality
control loop to keep it fast.
"""

import asyncio
import json
import os
import sys
from src.utils.key_setup import ensure_keys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.llm_client import LLMClient
from src.tools.tool_registry import ToolRegistry
from src.tools.search_tool import SearchTool
from src.tools.scholar_tool import ScholarTool
from src.tools.wikipedia_tool import WikipediaTool
from src.tools.calculator_tool import CalculatorTool
from src.tools.unit_converter_tool import UnitConverterTool
from src.tools.timeline_tool import TimelineTool
from src.tools.python_tool import PythonTool
from src.tools.comparison_tool import ComparisonTool
from src.tools.visit_tool import VisitTool
from src.graph.graph_builder import GraphBuilder
from src.graph.subgraph_sampler import SubgraphSampler
from src.graph.uncertainty_injector import UncertaintyInjector
from src.pipeline.stage2_graph_escalation import GraphComplexityEscalator
from src.utils.data_models import QAPair


async def main():
    print("=== Tongyi Pipeline: 10-Sample Generation ===\n")

    # Init LLM
    llm = LLMClient("configs/llm_config.yaml")

    # Init tools
    registry = ToolRegistry("configs/tool_config.yaml")
    cfg = registry.config
    registry.register(SearchTool(cfg.get("search", {})))
    registry.register(ScholarTool(cfg.get("scholar", {})))
    registry.register(WikipediaTool(cfg.get("wikipedia", {})))
    registry.register(CalculatorTool())
    registry.register(UnitConverterTool())
    registry.register(TimelineTool())
    registry.register(PythonTool(cfg.get("python", {})))
    registry.register(ComparisonTool(llm, cfg.get("comparison", {})))
    registry.register(VisitTool(llm, cfg.get("visit", {})))

    # Seed entities — diverse topics to demonstrate atomic graph construction
    seeds = [
        "Great Wall of China",
        "photosynthesis",
        "Claude Shannon information theory",
        "James Webb Space Telescope",
        "CRISPR gene editing",
        "2024 Nobel Prize in Physics",
    ]

    # Step 1: Build a small knowledge graph
    print("[1/4] Building knowledge graph from seed entities...")
    builder = GraphBuilder(llm, registry)
    kg = await builder.build_from_seeds(
        seeds,
        expansion_rounds=1,
        max_entities_per_expansion=5,
        cross_topic_edges=True,
        cross_topic_sample=3,
    )
    print(f"  KG: {kg.num_entities()} entities, {kg.num_relations()} relations\n")

    # Step 2: Sample subgraphs and generate initial QA pairs
    print("[2/4] Sampling subgraphs and generating QA pairs...")
    sampler = SubgraphSampler(kg)
    initial_qas: list[QAPair] = []

    target = 10
    attempts = 0
    max_attempts = target * 3
    qa_subgraphs: dict[str, list[str]] = {}  # qa.id -> entity_ids

    while len(initial_qas) < target and attempts < max_attempts:
        attempts += 1
        try:
            if attempts % 2 == 0:
                sample = sampler.sample_by_random_walk(min_entities=2, max_entities=5)
            else:
                sample = sampler.sample_bridging(n_components=2, entities_per_component=2)

            qa = await sampler.generate_qa_from_subgraph(sample, llm)
            initial_qas.append(qa)
            qa_subgraphs[qa.id] = sample.entity_ids
            print(f"  [{len(initial_qas)}/{target}] Q: {qa.question[:80]}...")
        except Exception as e:
            print(f"  (attempt {attempts} failed: {e})")

    print(f"\n  Generated {len(initial_qas)} initial QA pairs\n")

    # Step 3: Apply uncertainty injection to half of them
    print("[3/4] Applying uncertainty injection...")
    injector = UncertaintyInjector(llm)
    refined_qas: list[QAPair] = []

    for i, qa in enumerate(initial_qas):
        if i < len(initial_qas) // 2:
            try:
                refined = await injector.inject_multiple(qa, kg, n_operations=1)
                refined_qas.append(refined)
                # Carry subgraph mapping to the refined QA
                qa_subgraphs[refined.id] = qa_subgraphs.get(qa.id, qa.source_chunk_ids)
                print(f"  [{i+1}] Injected uncertainty -> Q: {refined.question[:80]}...")
            except Exception as e:
                print(f"  [{i+1}] Injection failed ({e}), keeping original")
                refined_qas.append(qa)
        else:
            refined_qas.append(qa)

    # Save uncertainty-injection checkpoint
    os.makedirs("data/output/checkpoints", exist_ok=True)
    with open("data/output/checkpoints/graph_refined_qas.jsonl", "w") as f:
        for qa in refined_qas:
            f.write(json.dumps(qa.model_dump()) + "\n")
    print(f"  Checkpoint saved: data/output/checkpoints/graph_refined_qas.jsonl\n")

    # Step 4: Graph-aware complexity escalation
    print("[4/5] Applying graph-aware complexity escalation...")
    graph_escalator = GraphComplexityEscalator(
        llm,
        registry,
        config={"max_tool_calls_per_round": 4},
    )
    final_qas: list[QAPair] = []

    for i, qa in enumerate(refined_qas):
        try:
            escalated = await graph_escalator.escalate_single(qa, kg, rounds=1)
            final_qas.append(escalated)
            qa_subgraphs[escalated.id] = escalated.source_chunk_ids
            print(f"  [{i+1}] Escalated -> Q: {escalated.question[:80]}...")
        except Exception as e:
            print(f"  [{i+1}] Escalation failed ({e}), keeping refined")
            final_qas.append(qa)

    # Save escalation checkpoint
    with open("data/output/checkpoints/graph_escalated_qas.jsonl", "w") as f:
        for qa in final_qas:
            f.write(json.dumps(qa.model_dump()) + "\n")
    print(f"  Checkpoint saved: data/output/checkpoints/graph_escalated_qas.jsonl\n")

    # Step 5: Save output — QA pairs, full KG, and per-QA subgraph mapping
    print(f"[5/5] Saving {len(final_qas)} QA pairs + graph data...")
    os.makedirs("data/output", exist_ok=True)

    output_path = "data/output/sample_10.jsonl"
    with open(output_path, "w") as f:
        for qa in final_qas:
            f.write(json.dumps(qa.model_dump(), indent=None) + "\n")

    # Save full KG
    kg_path = "data/output/knowledge_graph.json"
    with open(kg_path, "w") as f:
        json.dump(kg.to_dict(), f, indent=2)

    # Save combined UI data bundle (everything the viewer needs in one file)
    ui_data = {
        "knowledge_graph": kg.to_dict(),
        "qa_pairs": [],
    }
    for qa in final_qas:
        entity_ids = qa_subgraphs.get(qa.id, qa.source_chunk_ids)
        ui_data["qa_pairs"].append({
            **qa.model_dump(),
            "subgraph_entity_ids": entity_ids,
        })

    ui_path = "data/output/ui_data.json"
    with open(ui_path, "w") as f:
        json.dump(ui_data, f, indent=2)

    print(f"\n  Saved to {output_path}")
    print(f"  KG saved to {kg_path}")
    print(f"  UI bundle saved to {ui_path}\n")

    # Print results
    print("=" * 70)
    print("GENERATED QA PAIRS")
    print("=" * 70)
    for i, qa in enumerate(final_qas, 1):
        print(f"\n--- #{i} ---")
        print(f"Q: {qa.question}")
        print(f"A: {qa.answer}")
        if qa.escalation_history:
            print(f"   [history: {' -> '.join(qa.escalation_history[-2:])}]")
        if qa.metadata.get("strategy"):
            print(f"   [escalation strategy: {qa.metadata['strategy']}]")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    ensure_keys()
    asyncio.run(main())
