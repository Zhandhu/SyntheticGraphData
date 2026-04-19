"""Generate batches of 8 QA pairs until at least one passes all QC filters.

Reuses the existing knowledge graph (expensive to build) and samples fresh
subgraphs each batch. Pipeline per batch:
  1. Sample 8 subgraphs  → initial QAs
  2. Uncertainty injection
  3. Graph escalation
  4. QC: base solver → advanced solver → judge → dedup

Stops as soon as a PASS is found; prints a summary after every batch.

Optional --complexity flag (0.0–1.0) scales:
  - Subgraph walk depth   (more entities per sample)
  - Escalation rounds     (deeper research per QA)
  - QC advanced solver budget (more tool calls allowed for harder questions)
  - Escalation strategy weights (toward bridging at high complexity)
Omitting --complexity runs the original fixed pipeline unchanged.
"""

import argparse
import asyncio
import json
import os
import sys
from src.utils.key_setup import ensure_keys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.judge_agent import JudgeAgent
from src.agents.question_solver_agent import QuestionSolverAgent
from src.agents.similarity_scorer import SimilarityScorer
from src.graph.knowledge_graph import KnowledgeGraph
from src.graph.subgraph_sampler import SubgraphSampler
from src.graph.uncertainty_injector import UncertaintyInjector
from src.pipeline.stage2_graph_escalation import GraphComplexityEscalator
from src.tools.calculator_tool import CalculatorTool
from src.tools.comparison_tool import ComparisonTool
from src.tools.visit_tool import VisitTool
from src.tools.scholar_tool import ScholarTool
from src.tools.search_tool import SearchTool
from src.tools.table_tool import TableTool
from src.tools.timeline_tool import TimelineTool
from src.tools.tool_registry import ToolRegistry
from src.tools.unit_converter_tool import UnitConverterTool
from src.tools.wikipedia_tool import WikipediaTool
from src.utils.data_models import QAPair
from src.utils.llm_client import LLMClient

BATCH_SIZE = 8
MAX_BATCHES = 20  # safety ceiling

# Original fixed pipeline defaults (used when --complexity is not set)
_DEFAULT_WALK_MIN      = 2
_DEFAULT_WALK_MAX      = 6
_DEFAULT_ESC_ROUNDS    = 1
_DEFAULT_MAX_TOOL_QC   = 10
_DEFAULT_ESC_STRATEGIES = None  # means: use whatever the escalator picks

# Strategy pool and their complexity affinity (0=low, 1=high)
_STRATEGIES = ["hop_extension", "subgraph_bridging",
               "multi_entity_aggregation", "constraint_chain", "inverse_reasoning"]
_STRATEGY_AFFINITY = [0.1, 0.8, 0.7, 0.5, 0.4]


def complexity_params(c: float | None) -> dict:
    """Map a complexity value in [0, 1] to concrete pipeline parameters.

    Returns a dict of parameters. When c is None, all values are the
    original defaults — the pipeline is completely unchanged.
    """
    if c is None:
        return {
            "walk_min": _DEFAULT_WALK_MIN,
            "walk_max": _DEFAULT_WALK_MAX,
            "walk_length": 10,
            "bridge_components": 2,
            "bridge_per_component": 2,
            "esc_rounds": _DEFAULT_ESC_ROUNDS,
            "max_tool_calls_qc": _DEFAULT_MAX_TOOL_QC,
            "strategy_weights": None,
        }

    # Linear interpolation across the [0, 1] range
    walk_min = max(2, int(2 + c * 4))              # 2 → 6
    walk_max = max(walk_min + 1, int(4 + c * 8))   # 5 → 12
    walk_length = int(10 + c * 20)                 # 10 → 30: longer walks = higher diameter
    bridge_components = max(2, round(2 + c * 2))   # 2 → 4: more distant clusters at high cx
    bridge_per_comp = max(2, round(2 + c * 2))     # 2 → 4: richer local neighborhoods
    esc_rounds = 1
    max_tool_qc = int(8 + c * 4)                   # 8 → 12

    # Weight strategies toward bridging/aggregation at high complexity
    weights = [max(0.05, a * c + (1 - a) * (1 - c)) for a in _STRATEGY_AFFINITY]
    total = sum(weights)
    strategy_weights = {s: w / total for s, w in zip(_STRATEGIES, weights)}

    return {
        "walk_min": walk_min,
        "walk_max": walk_max,
        "walk_length": walk_length,
        "bridge_components": bridge_components,
        "bridge_per_component": bridge_per_comp,
        "esc_rounds": esc_rounds,
        "max_tool_calls_qc": max_tool_qc,
        "strategy_weights": strategy_weights,
    }


# ── QC helpers ────────────────────────────────────────────────────────────────

async def run_qc_single(
    qa: QAPair,
    base_solver: QuestionSolverAgent,
    advanced_solver: QuestionSolverAgent,
    judge: JudgeAgent,
    scorer: SimilarityScorer,
    max_tool_calls_qc: int = _DEFAULT_MAX_TOOL_QC,
    complexity: float | None = None,
) -> tuple[str, str, str, object]:
    """Returns (verdict, base_answer, adv_answer, trajectory|None)."""
    # Base solver is always judged strictly — easy questions should be easy
    base_resp = await base_solver.solve_base(qa.question)
    base_correct = await judge.judge(qa.answer, base_resp.answer, complexity=None)
    if base_correct:
        return "TOO_EASY", base_resp.answer, "", None

    adv_resp = await advanced_solver.solve_advanced(
        qa.question, qa_id=qa.id, max_tool_calls=max_tool_calls_qc
    )
    # Advanced solver is judged with complexity-aware leniency
    adv_correct = await judge.judge(qa.answer, adv_resp.answer, complexity=complexity)
    if not adv_correct:
        return "TOO_HARD", base_resp.answer, adv_resp.answer, None

    if scorer.is_duplicate(qa):
        return "DUPLICATE", base_resp.answer, adv_resp.answer, None

    scorer.add_to_index(qa)
    return "PASS", base_resp.answer, adv_resp.answer, adv_resp.trajectory


async def run_qc_batch(
    qas: list[QAPair],
    base_solver: QuestionSolverAgent,
    advanced_solver: QuestionSolverAgent,
    judge: JudgeAgent,
    scorer: SimilarityScorer,
    max_concurrent: int = 3,
    max_tool_calls_qc: int = _DEFAULT_MAX_TOOL_QC,
    complexity: float | None = None,
) -> list[tuple[QAPair, str, str, str, object]]:
    sem = asyncio.Semaphore(max_concurrent)

    async def _run(qa):
        async with sem:
            try:
                v, b, a, traj = await run_qc_single(
                    qa, base_solver, advanced_solver, judge, scorer,
                    max_tool_calls_qc=max_tool_calls_qc,
                    complexity=complexity,
                )
                return qa, v, b, a, traj
            except Exception as e:
                return qa, "ERROR", "", str(e), None

    return await asyncio.gather(*[_run(qa) for qa in qas])


# ── Pipeline per batch ────────────────────────────────────────────────────────

async def generate_batch(
    kg: KnowledgeGraph,
    sampler: SubgraphSampler,
    injector: UncertaintyInjector,
    escalator: GraphComplexityEscalator,
    batch_size: int,
    batch_num: int,
    params: dict,
) -> list[QAPair]:
    import random
    llm = escalator.llm
    walk_min = params["walk_min"]
    walk_max = params["walk_max"]
    walk_length = params["walk_length"]
    bridge_components = params["bridge_components"]
    bridge_per_comp = params["bridge_per_component"]
    esc_rounds = params["esc_rounds"]
    strategy_weights = params["strategy_weights"]

    # Step 1: Sample subgraphs in parallel (launch 2× to absorb failures)
    gen_sem = asyncio.Semaphore(8)  # cap concurrent LLM calls

    async def _try_one(idx: int) -> QAPair | None:
        async with gen_sem:
            try:
                sample = (
                    sampler.sample_by_random_walk(
                        min_entities=walk_min, max_entities=walk_max,
                        walk_length=walk_length,
                    )
                    if idx % 2 == 0
                    else sampler.sample_bridging(
                        n_components=bridge_components,
                        entities_per_component=bridge_per_comp,
                    )
                )
                return await sampler.generate_qa_from_subgraph(sample, llm)
            except Exception:
                return None

    candidates = await asyncio.gather(*[_try_one(i) for i in range(batch_size * 2)])
    qas = [qa for qa in candidates if qa is not None][:batch_size]

    # Fall back sequentially if parallel attempts didn't yield enough
    extra = 0
    while len(qas) < batch_size and extra < batch_size:
        extra += 1
        qa = await _try_one(extra)
        if qa is not None:
            qas.append(qa)

    # Step 2: Uncertainty injection — fully parallel
    async def _inject_one(qa: QAPair) -> QAPair:
        try:
            return await injector.inject_multiple(qa, kg, n_operations=1)
        except Exception:
            return qa

    refined = list(await asyncio.gather(*[_inject_one(qa) for qa in qas]))

    # Step 3: Graph escalation with complexity-aware rounds + strategy weights
    # Temporarily override strategy selection if weights provided
    if strategy_weights is not None:
        import src.pipeline.stage2_graph_escalation as _esc_mod
        _orig_strategies = _esc_mod.STRATEGIES
        _esc_mod.STRATEGIES = random.choices(
            list(strategy_weights.keys()),
            weights=list(strategy_weights.values()),
            k=len(_orig_strategies),
        )

    escalated = await escalator.escalate_batch(
        refined, kg, rounds=esc_rounds, max_concurrent=6
    )

    if strategy_weights is not None:
        _esc_mod.STRATEGIES = _orig_strategies

    return escalated


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs until one passes QC.")
    parser.add_argument(
        "--complexity", type=float, default=None,
        help="Complexity level 0.0–1.0. Omit to run the original fixed pipeline.",
    )
    parser.add_argument("--batches", type=int, default=MAX_BATCHES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    params = complexity_params(args.complexity)
    max_batches = args.batches
    batch_size = args.batch_size

    print("=== Generate-until-PASS ===")
    if args.complexity is not None:
        print(f"Complexity: {args.complexity:.2f} | "
              f"walk={params['walk_min']}-{params['walk_max']} | "
              f"esc_rounds={params['esc_rounds']} | "
              f"qc_tools={params['max_tool_calls_qc']}")
    else:
        print("Complexity: original pipeline (no complexity scaling)")
    print(f"Batch size: {batch_size} | Max batches: {max_batches}\n")

    # Init shared components
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

    base_solver     = QuestionSolverAgent(llm)
    advanced_solver = QuestionSolverAgent(llm, tool_registry=registry)
    judge           = JudgeAgent(llm)
    scorer          = SimilarityScorer(threshold=0.85)

    # Load existing KG (skip expensive rebuild)
    kg_path = "data/output/knowledge_graph.json"
    if not os.path.exists(kg_path):
        print("ERROR: knowledge_graph.json not found. Run generate_sample.py first.")
        sys.exit(1)

    with open(kg_path) as f:
        kg = KnowledgeGraph.from_dict(json.load(f))
    print(f"Loaded KG: {kg.num_entities()} entities, {kg.num_relations()} relations\n")

    sampler   = SubgraphSampler(kg)
    injector  = UncertaintyInjector(llm)
    escalator = GraphComplexityEscalator(
        llm, registry, config={"max_tool_calls_per_round": 4}
    )

    all_passing: list[tuple[QAPair, str]] = []  # (qa, adv_answer)
    total_generated = 0
    total_verdicts: dict[str, int] = {}

    for batch_num in range(1, max_batches + 1):
        t0 = time.time()
        print(f"── Batch {batch_num} ──────────────────────────────")

        qas = await generate_batch(
            kg, sampler, injector, escalator, batch_size, batch_num, params
        )
        # Tag each QA with the complexity level used to produce it
        for qa in qas:
            qa.metadata["complexity_level"] = args.complexity
        print(f"  Generated {len(qas)} QAs, running QC...")

        results = await run_qc_batch(
            qas, base_solver, advanced_solver, judge, scorer,
            max_concurrent=6,
            max_tool_calls_qc=params["max_tool_calls_qc"],
            complexity=args.complexity,
        )
        elapsed = time.time() - t0

        batch_verdicts: dict[str, int] = {}
        batch_passing = []
        for qa, v, b, a, traj in results:
            batch_verdicts[v] = batch_verdicts.get(v, 0) + 1
            total_verdicts[v]  = total_verdicts.get(v, 0) + 1
            if v == "PASS":
                batch_passing.append((qa, a, traj))
                all_passing.append((qa, a, traj))

        total_generated += len(qas)

        # Per-batch summary line
        vstr = "  ".join(f"{k}:{n}" for k, n in sorted(batch_verdicts.items()))
        pass_n = batch_verdicts.get("PASS", 0)
        print(f"  [{vstr}]  {elapsed:.0f}s  {'★ ' + str(pass_n) + ' PASS(es)!' if pass_n else ''}")

        # Print passing QAs immediately
        for qa, adv_answer, _ in batch_passing:
            strategy = qa.metadata.get("strategy", "")
            print(f"\n  ★ PASS ★")
            print(f"  Q: {qa.question}")
            print(f"  Ref A: {qa.answer}")
            print(f"  Solver A: {adv_answer}")
            if strategy:
                print(f"  Strategy: {strategy}")

        print()

        # Don't stop early — collect as many passing QAs as possible
        # if all_passing: break

    # ── Final summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Total generated: {total_generated} across {max_batches} batch(es)")
    print(f"Verdict totals:  {dict(sorted(total_verdicts.items()))}")
    print(f"Passing:         {len(all_passing)}")

    if all_passing:
        os.makedirs("data/output", exist_ok=True)
        out_path = "data/output/passing_qas.jsonl"
        traj_path = "data/output/passing_trajectories.jsonl"
        mode = "a" if os.path.exists(out_path) else "w"
        traj_mode = "a" if os.path.exists(traj_path) else "w"
        with open(out_path, mode) as f, open(traj_path, traj_mode) as tf:
            for qa, _, traj in all_passing:
                f.write(json.dumps(qa.model_dump()) + "\n")
                if traj is not None:
                    tf.write(json.dumps(traj.model_dump()) + "\n")
        print(f"Appended to {out_path}")
        print(f"Trajectories saved to {traj_path}")

        # ── Merge into ui_data.json so the viewer picks them up ──────────────
        ui_path = "data/output/ui_data.json"
        if os.path.exists(ui_path):
            with open(ui_path) as f:
                ui_data = json.load(f)
        else:
            ui_data = {"knowledge_graph": {}, "qa_pairs": []}

        existing_ids = {q["id"] for q in ui_data.get("qa_pairs", [])}
        added = 0
        for qa, adv_answer, _ in all_passing:
            if qa.id in existing_ids:
                continue
            entry = qa.model_dump()
            entry["subgraph_entity_ids"] = qa.source_chunk_ids
            entry["qc_verdict"] = "PASS"
            entry["advanced_solver_answer"] = adv_answer
            ui_data["qa_pairs"].append(entry)
            existing_ids.add(qa.id)
            added += 1

        with open(ui_path, "w") as f:
            json.dump(ui_data, f)
        print(f"Added {added} new QA(s) to {ui_path}")
    else:
        print(f"No QAs passed in {MAX_BATCHES} batches.")


if __name__ == "__main__":
    ensure_keys()
    asyncio.run(main())
