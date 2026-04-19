"""Run quality control on the generated sample_10.jsonl output.

Stage 3 logic (Paper 1, Section 3.3):
  1. Base solver (no tools)  → if correct: TOO_EASY
  2. Advanced solver (tools, ReAct loop) → if wrong: TOO_HARD
  3. Duplicate check (embedding similarity) → if duplicate: DUPLICATE
  4. All three pass → PASS
"""

import asyncio
import json
import os
import sys
from src.utils.key_setup import ensure_keys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.judge_agent import JudgeAgent
from src.agents.question_solver_agent import QuestionSolverAgent
from src.agents.similarity_scorer import SimilarityScorer
from src.tools.calculator_tool import CalculatorTool
from src.tools.scholar_tool import ScholarTool
from src.tools.search_tool import SearchTool
from src.tools.timeline_tool import TimelineTool
from src.tools.tool_registry import ToolRegistry
from src.tools.unit_converter_tool import UnitConverterTool
from src.tools.wikipedia_tool import WikipediaTool
from src.utils.data_models import QAPair
from src.utils.llm_client import LLMClient

VERDICT_EMOJI = {
    "PASS":      "✓ PASS",
    "TOO_EASY":  "✗ TOO_EASY",
    "TOO_HARD":  "✗ TOO_HARD",
    "DUPLICATE": "✗ DUPLICATE",
}


async def main():
    print("=== Stage 3: Quality Control ===\n")

    llm = LLMClient("configs/llm_config.yaml")

    registry = ToolRegistry("configs/tool_config.yaml")
    cfg = registry.config
    registry.register(SearchTool(cfg.get("search", {})))
    registry.register(ScholarTool(cfg.get("scholar", {})))
    registry.register(WikipediaTool(cfg.get("wikipedia", {})))
    registry.register(CalculatorTool())
    registry.register(UnitConverterTool())
    registry.register(TimelineTool())

    base_solver    = QuestionSolverAgent(llm)
    advanced_solver = QuestionSolverAgent(llm, tool_registry=registry)
    judge          = JudgeAgent(llm)
    scorer         = SimilarityScorer(threshold=0.85)

    with open("data/output/sample_10.jsonl") as f:
        qas = [QAPair(**json.loads(l)) for l in f]

    results = []

    for i, qa in enumerate(qas, 1):
        print(f"--- QA #{i} ---")
        print(f"Q: {qa.question[:120]}{'...' if len(qa.question)>120 else ''}")

        # ── Step 1: Base solver (no tools) ──────────────────────────────
        base_resp = await base_solver.solve_base(qa.question)
        base_correct = await judge.judge(qa.answer, base_resp.answer)
        print(f"  Base solver:     {'CORRECT' if base_correct else 'wrong'}")
        print(f"    answer: {base_resp.answer[:100]}")

        if base_correct:
            verdict = "TOO_EASY"
            print(f"  → {VERDICT_EMOJI[verdict]} (base solver answered without tools)\n")
            results.append((qa, verdict, base_resp.answer, None))
            continue

        # ── Step 2: Advanced solver (ReAct + tools) ──────────────────────
        adv_resp = await advanced_solver.solve_advanced(qa.question)
        adv_correct = await judge.judge(qa.answer, adv_resp.answer)
        print(f"  Advanced solver: {'CORRECT' if adv_correct else 'wrong'}")
        print(f"    answer: {adv_resp.answer[:100]}")

        if not adv_correct:
            verdict = "TOO_HARD"
            print(f"  → {VERDICT_EMOJI[verdict]} (advanced solver also failed)\n")
            results.append((qa, verdict, base_resp.answer, adv_resp.answer))
            continue

        # ── Step 3: Duplicate check ───────────────────────────────────────
        is_dup = scorer.is_duplicate(qa)
        if is_dup:
            verdict = "DUPLICATE"
            print(f"  → {VERDICT_EMOJI[verdict]}\n")
            results.append((qa, verdict, base_resp.answer, adv_resp.answer))
            continue

        scorer.add_to_index(qa)
        verdict = "PASS"
        print(f"  → {VERDICT_EMOJI[verdict]}\n")
        results.append((qa, verdict, base_resp.answer, adv_resp.answer))

    # ── Summary ───────────────────────────────────────────────────────────
    from collections import Counter
    counts = Counter(v for _, v, _, _ in results)
    passing = [(qa, adv) for qa, v, _, adv in results if v == "PASS"]

    print("=" * 70)
    print(f"RESULTS: {len(qas)} evaluated")
    for verdict in ["PASS", "TOO_EASY", "TOO_HARD", "DUPLICATE"]:
        n = counts.get(verdict, 0)
        if n:
            print(f"  {VERDICT_EMOJI[verdict]}: {n}")
    print()

    if passing:
        print(f"PASSING QA PAIRS ({len(passing)}):")
        print("=" * 70)
        for qa, adv_answer in passing:
            print(f"\nQ: {qa.question}")
            print(f"Reference A: {qa.answer}")
            print(f"Solver A:    {adv_answer}")
            strategy = qa.metadata.get("strategy", "")
            if strategy:
                print(f"Strategy:    {strategy}")
    else:
        print("No QA pairs passed all three filters.")

    # Save passing pairs
    os.makedirs("data/output", exist_ok=True)
    with open("data/output/sample_10_passed.jsonl", "w") as f:
        for qa, v, _, _ in results:
            if v == "PASS":
                f.write(json.dumps(qa.model_dump()) + "\n")
    print(f"\nPassing pairs saved to data/output/sample_10_passed.jsonl")

    # Enrich ui_data.json with QC verdicts + solver answers
    ui_path = "data/output/ui_data.json"
    if os.path.exists(ui_path):
        with open(ui_path) as f:
            ui_data = json.load(f)

        # Build lookup by qa id
        verdict_map = {qa.id: (v, base_ans, adv_ans) for qa, v, base_ans, adv_ans in results}

        for pair in ui_data["qa_pairs"]:
            qid = pair.get("id")
            if qid in verdict_map:
                v, base_ans, adv_ans = verdict_map[qid]
                pair["qc_verdict"] = v
                pair["base_solver_answer"] = base_ans or ""
                pair["advanced_solver_answer"] = adv_ans or ""

        with open(ui_path, "w") as f:
            json.dump(ui_data, f, indent=2)
        print(f"ui_data.json enriched with QC verdicts: {ui_path}")


if __name__ == "__main__":
    ensure_keys()
    asyncio.run(main())
