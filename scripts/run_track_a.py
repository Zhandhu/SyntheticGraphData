"""Track A (WebFrontier) pipeline: corpus → QA pairs via complexity escalation.

WebFrontier paper §3.1-3.3:
  1. Fetch web documents (Wikipedia articles, web pages) as the corpus
  2. SummaryAgent chunks + summarizes each document
  3. Relevance-based triplet grouping → CompositeUnits
  4. ItemWriterAgent generates seed QA pairs from each CompositeUnit
  5. Iterative complexity escalation (4 strategies × N rounds)
  6. Quality control: base solver → advanced solver → judge → dedup

Results are saved to data/output/track_a_qas.jsonl and
data/output/track_a_trajectories.jsonl.
"""

import asyncio
import json
import os
import sys
from src.utils.key_setup import ensure_keys
import time

import httpx
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.item_writer_agent import ItemWriterAgent
from src.agents.judge_agent import JudgeAgent
from src.agents.question_solver_agent import QuestionSolverAgent
from src.agents.similarity_scorer import SimilarityScorer
from src.agents.summary_agent import SummaryAgent
from src.pipeline.stage1_seed_generation import SeedGenerator
from src.pipeline.stage2_complexity_escalation import ComplexityEscalator
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
from src.utils.data_models import QAPair, Trajectory
from src.utils.llm_client import LLMClient

# ── Corpus topics ─────────────────────────────────────────────────────────────
# Wikipedia articles to use as the Track A corpus.
# These should be rich, factually dense articles that produce
# interesting multi-source QA pairs.
CORPUS_TOPICS = [
    "mRNA vaccine",
    "Lithium-ion battery",
    "Transformer (machine learning model)",
    "Silk Road",
    "Manhattan Project",
    "Carbon credit",
    "Antimicrobial resistance",
    "Black hole thermodynamics",
    "CRISPR",
    "Claude Shannon",
]

ESCALATION_ROUNDS = 2
QC_MAX_CONCURRENT = 3
OUTPUT_PATH = "data/output/track_a_qas.jsonl"
TRAJ_PATH = "data/output/track_a_trajectories.jsonl"


# ── Corpus fetching ───────────────────────────────────────────────────────────

async def fetch_wikipedia_text(topic: str, max_chars: int = 8000) -> str:
    """Fetch the introductory text of a Wikipedia article."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()
            extract = data.get("extract", "")
            return extract[:max_chars]
    except Exception as e:
        print(f"  [warn] Wikipedia fetch failed for '{topic}': {e}")
        return ""


async def build_corpus(topics: list[str]) -> list[str]:
    """Fetch Wikipedia articles for each topic."""
    print(f"Fetching corpus for {len(topics)} topics...")
    tasks = [fetch_wikipedia_text(t) for t in topics]
    texts = await asyncio.gather(*tasks)
    valid = [(t, text) for t, text in zip(topics, texts) if text and len(text) > 200]
    print(f"  Fetched {len(valid)}/{len(topics)} articles")
    return [text for _, text in valid]


# ── QC helpers ────────────────────────────────────────────────────────────────

async def run_qc_single(
    qa: QAPair,
    base_solver: QuestionSolverAgent,
    advanced_solver: QuestionSolverAgent,
    judge: JudgeAgent,
    scorer: SimilarityScorer,
) -> tuple[str, str, str, object]:
    base_resp = await base_solver.solve_base(qa.question)
    base_correct = await judge.judge(qa.answer, base_resp.answer)
    if base_correct:
        return "TOO_EASY", base_resp.answer, "", None

    adv_resp = await advanced_solver.solve_advanced(qa.question, qa_id=qa.id)
    adv_correct = await judge.judge(qa.answer, adv_resp.answer)
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
) -> list[tuple[QAPair, str, str, str, object]]:
    sem = asyncio.Semaphore(QC_MAX_CONCURRENT)

    async def _run(qa):
        async with sem:
            try:
                v, b, a, traj = await run_qc_single(qa, base_solver, advanced_solver, judge, scorer)
                return qa, v, b, a, traj
            except Exception as e:
                return qa, "ERROR", "", str(e), None

    return await asyncio.gather(*[_run(qa) for qa in qas])


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=== Track A (WebFrontier) Pipeline ===\n")

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

    # Stage 1: Build corpus and generate seeds
    print("[1/3] Building corpus and generating seed QA pairs...")
    corpus_texts = await build_corpus(CORPUS_TOPICS)

    summary_agent = SummaryAgent(llm)
    item_writer = ItemWriterAgent(llm)
    scorer = SimilarityScorer(threshold=0.85)

    seed_gen = SeedGenerator(
        summary_agent=summary_agent,
        item_writer=item_writer,
        similarity_scorer=scorer,
        config={"group_size": 3, "seeds_per_group": 2, "min_shared_entities": 0},
    )
    seed_qas = await seed_gen.generate_from_corpus(corpus_texts)
    print(f"  Generated {len(seed_qas)} seed QA pairs\n")

    # Stage 2: Iterative complexity escalation
    print(f"[2/3] Escalating complexity ({ESCALATION_ROUNDS} rounds)...")
    escalator = ComplexityEscalator(
        item_writer=item_writer,
        tool_registry=registry,
        config={"max_tool_calls_per_round": 5},
    )
    escalated_qas: list[QAPair] = []
    for i, qa in enumerate(seed_qas):
        try:
            result = await escalator.escalate_single(qa, rounds=ESCALATION_ROUNDS)
            escalated_qas.append(result)
            print(f"  [{i+1}/{len(seed_qas)}] {result.question[:80]}...")
        except Exception as e:
            print(f"  [{i+1}/{len(seed_qas)}] escalation failed ({e}), keeping seed")
            escalated_qas.append(qa)

    print(f"\n  {len(escalated_qas)} QAs ready for QC\n")

    # Stage 3: Quality control
    print("[3/3] Running quality control...")
    base_solver = QuestionSolverAgent(llm)
    advanced_solver = QuestionSolverAgent(llm, tool_registry=registry)
    judge = JudgeAgent(llm)

    t0 = time.time()
    results = await run_qc_batch(escalated_qas, base_solver, advanced_solver, judge, scorer)
    elapsed = time.time() - t0

    verdicts: dict[str, int] = {}
    passing: list[tuple[QAPair, Trajectory]] = []
    for qa, v, b, a, traj in results:
        verdicts[v] = verdicts.get(v, 0) + 1
        if v == "PASS":
            passing.append((qa, traj))

    vstr = "  ".join(f"{k}:{n}" for k, n in sorted(verdicts.items()))
    print(f"  [{vstr}]  {elapsed:.0f}s")
    print(f"  Passing: {len(passing)}\n")

    # Save results
    os.makedirs("data/output", exist_ok=True)
    qa_mode = "a" if os.path.exists(OUTPUT_PATH) else "w"
    traj_mode = "a" if os.path.exists(TRAJ_PATH) else "w"

    with open(OUTPUT_PATH, qa_mode) as f, open(TRAJ_PATH, traj_mode) as tf:
        for qa, traj in passing:
            f.write(json.dumps(qa.model_dump()) + "\n")
            if traj is not None:
                tf.write(json.dumps(traj.model_dump()) + "\n")

    print(f"Passing QAs saved to {OUTPUT_PATH}")
    print(f"Trajectories saved to {TRAJ_PATH}")

    # Summary
    print("\n" + "=" * 60)
    for i, (qa, _) in enumerate(passing, 1):
        print(f"\n--- PASS #{i} ---")
        print(f"Q: {qa.question}")
        print(f"A: {qa.answer[:300]}")


if __name__ == "__main__":
    ensure_keys()
    asyncio.run(main())
