"""Stage 3: Quality Control.

Paper 1, Section 3.3: Base solver → Advanced solver → Judge → Dedup.
"""

import asyncio

from src.agents.question_solver_agent import QuestionSolverAgent
from src.agents.judge_agent import JudgeAgent
from src.agents.similarity_scorer import SimilarityScorer
from src.utils.data_models import QAPair, QualityVerdict, FilterVerdict
from src.utils.logging_utils import get_logger, log_stage_transition

logger = get_logger(__name__)


class QualityController:
    """Full Stage 3 implementation."""

    def __init__(
        self,
        base_solver: QuestionSolverAgent,
        advanced_solver: QuestionSolverAgent,
        judge: JudgeAgent,
        similarity_scorer: SimilarityScorer,
        config: dict,
    ):
        self.base_solver = base_solver
        self.advanced_solver = advanced_solver
        self.judge = judge
        self.similarity_scorer = similarity_scorer
        self.config = config

    async def evaluate_single(self, qa: QAPair) -> QualityVerdict:
        """Paper 1, Section 3.3 quality control pipeline.

        Step 1: Base solver (no tools) — if correct, too easy.
        Step 2: Advanced solver (with tools) — if fails, too hard.
        Step 3: Duplicate check via SimilarityScorer.
        """
        # Step 1: Base solver
        base_result = await self.base_solver.solve_base(qa.question)
        base_correct = await self.judge.judge(qa.answer, base_result.answer)

        if base_correct:
            return QualityVerdict(
                qa_id=qa.id,
                base_solver_correct=True,
                verdict=FilterVerdict.TOO_EASY,
                explanation="Base solver answered correctly without tools",
            )

        # Step 2: Advanced solver
        adv_result = await self.advanced_solver.solve_advanced(qa.question)
        adv_correct = await self.judge.judge(qa.answer, adv_result.answer)

        if not adv_correct:
            return QualityVerdict(
                qa_id=qa.id,
                advanced_solver_correct=False,
                verdict=FilterVerdict.TOO_HARD,
                explanation="Advanced solver with tools could not answer",
            )

        # Step 3: Duplicate check
        is_dup = self.similarity_scorer.is_duplicate(qa)
        if is_dup:
            return QualityVerdict(
                qa_id=qa.id,
                base_solver_correct=False,
                advanced_solver_correct=True,
                verdict=FilterVerdict.DUPLICATE,
            )

        # Passed all checks
        self.similarity_scorer.add_to_index(qa)
        return QualityVerdict(
            qa_id=qa.id,
            base_solver_correct=False,
            advanced_solver_correct=True,
            verdict=FilterVerdict.PASS,
        )

    async def filter_batch(
        self, qas: list[QAPair], max_concurrent: int = 5
    ) -> list[QAPair]:
        """Filter a batch, return only PASS verdicts."""
        log_stage_transition(logger, "stage3_quality", "started", count=len(qas))

        semaphore = asyncio.Semaphore(max_concurrent)
        verdicts: list[QualityVerdict] = []

        async def evaluate_with_semaphore(qa: QAPair) -> QualityVerdict:
            async with semaphore:
                try:
                    return await self.evaluate_single(qa)
                except Exception as e:
                    logger.warning(f"QC evaluation failed for {qa.id}: {e}")
                    return QualityVerdict(
                        qa_id=qa.id,
                        verdict=FilterVerdict.INVALID,
                        explanation=f"Evaluation error: {e}",
                    )

        tasks = [evaluate_with_semaphore(qa) for qa in qas]
        verdicts = await asyncio.gather(*tasks)

        # Log statistics
        from collections import Counter
        verdict_counts = Counter(v.verdict for v in verdicts)
        logger.info(f"QC results: {dict(verdict_counts)}")

        # Return only passing QA pairs
        passing_ids = {v.qa_id for v in verdicts if v.verdict == FilterVerdict.PASS}
        passed = [qa for qa in qas if qa.id in passing_ids]

        log_stage_transition(logger, "stage3_quality", "completed", count=len(passed))
        return passed
