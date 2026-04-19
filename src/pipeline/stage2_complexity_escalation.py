"""Stage 2: Complexity Escalation.

Paper 1, Section 3.2: 'For each seed QA pair, the tool-augmented agent
iteratively evolves both the question and answer to increase their
cognitive complexity.'
"""

import asyncio

from src.agents.item_writer_agent import ItemWriterAgent
from src.tools.tool_registry import ToolRegistry
from src.utils.data_models import QAPair, QADifficulty
from src.utils.logging_utils import get_logger, log_stage_transition

logger = get_logger(__name__)


class ComplexityEscalator:
    """Full Stage 2 implementation."""

    def __init__(
        self,
        item_writer: ItemWriterAgent,
        tool_registry: ToolRegistry,
        config: dict,
    ):
        self.item_writer = item_writer
        self.tool_registry = tool_registry
        self.config = config

    async def escalate_single(self, qa: QAPair, rounds: int = 3) -> QAPair:
        """Iteratively escalate a single QA pair.

        Paper 1: 'This iterative process creates a virtuous cycle where a more
        sophisticated QA pair generated in one iteration becomes the seed for
        the next.'
        """
        max_tool_calls = self.config.get("max_tool_calls_per_round", 5)
        current = qa

        for i in range(rounds):
            try:
                result = await self.item_writer.escalate(
                    current, self.tool_registry, max_tool_calls=max_tool_calls
                )
                current = result.escalated
                current.difficulty = QADifficulty(f"escalated_{i + 1}")
                current.escalation_history.append(qa.question)
            except Exception as e:
                logger.warning(f"Escalation round {i + 1} failed: {e}")
                break

        return current

    async def escalate_batch(
        self,
        qas: list[QAPair],
        rounds: int = 3,
        max_concurrent: int = 5,
    ) -> list[QAPair]:
        """Run escalation on a batch with concurrency control via semaphore."""
        log_stage_transition(logger, "stage2_escalation", "started", count=len(qas))

        semaphore = asyncio.Semaphore(max_concurrent)

        async def escalate_with_semaphore(qa: QAPair) -> QAPair:
            async with semaphore:
                return await self.escalate_single(qa, rounds=rounds)

        tasks = [escalate_with_semaphore(qa) for qa in qas]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        escalated = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Escalation failed for QA {qas[i].id}: {result}")
                escalated.append(qas[i])  # Keep original on failure
            else:
                escalated.append(result)

        log_stage_transition(logger, "stage2_escalation", "completed", count=len(escalated))
        return escalated
