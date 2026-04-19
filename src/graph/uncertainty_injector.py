"""Uncertainty Injector — applies atomic operations to increase QA difficulty.

Paper 2, Section 3.4.1: 'The pivotal step involves strategically increasing
the uncertainty within the question to enhance its difficulty. We formally
model QA difficulty as a series of controllable "atomic operations".'
"""

import random

from pydantic import BaseModel

from src.graph.knowledge_graph import KnowledgeGraph
from src.utils.data_models import QAPair
from src.utils.llm_client import LLMClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class InjectionResponse(BaseModel):
    question: str
    answer: str
    explanation: str


OPERATIONS = {
    "entity_merging": (
        "Replace a specific entity name with a descriptive reference that could "
        "match multiple entities. E.g., 'Einstein' -> 'the physicist who received "
        "the 1921 Nobel Prize.'"
    ),
    "temporal_displacement": (
        "Add or change temporal constraints. E.g., add 'as of 2019' or "
        "'during the Cold War period.'"
    ),
    "attribute_substitution": (
        "Replace an easy-to-find attribute with a harder one. E.g., 'capital' -> "
        "'second-largest city by population.'"
    ),
    "condition_injection": (
        "Add a filtering condition that narrows the scope. E.g., 'among EU member "
        "states' or 'excluding island nations.'"
    ),
    "aggregation_requirement": (
        "Require combining data across multiple entities. E.g., 'total GDP' -> "
        "'average GDP per capita weighted by population.'"
    ),
    "indirection": (
        "Add an intermediate step. E.g., instead of asking about X directly, ask "
        "about 'the country that borders both X and Y.'"
    ),
}


class UncertaintyInjector:
    """Apply atomic uncertainty operations to increase QA difficulty."""

    def __init__(self, llm_client: LLMClient, prompt_path: str = "prompts/uncertainty_injection.txt"):
        self.llm = llm_client
        with open(prompt_path) as f:
            self.prompt_template = f.read()

    async def inject(
        self,
        qa: QAPair,
        operation: str,
        kg_context: KnowledgeGraph,
    ) -> QAPair:
        """Apply a single uncertainty operation to a QA pair."""
        if operation not in OPERATIONS:
            raise ValueError(
                f"Unknown operation: {operation}. "
                f"Available: {list(OPERATIONS.keys())}"
            )

        kg_desc = kg_context.describe_subgraph()
        prompt = self.prompt_template.format(
            operation_name=operation,
            operation_description=OPERATIONS[operation],
            question=qa.question,
            answer=qa.answer,
            kg_context=kg_desc,
        )

        messages = [{"role": "user", "content": prompt}]
        response = await self.llm.complete_json(messages, InjectionResponse)

        new_qa = qa.model_copy()
        new_qa.question = response.question
        new_qa.answer = response.answer
        new_qa.escalation_history.append(f"uncertainty:{operation}")
        new_qa.metadata["last_uncertainty_op"] = operation
        new_qa.metadata["uncertainty_explanation"] = response.explanation

        return new_qa

    async def inject_multiple(
        self,
        qa: QAPair,
        kg_context: KnowledgeGraph,
        n_operations: int = 2,
        available_operations: list[str] | None = None,
    ) -> QAPair:
        """Apply n random operations sequentially."""
        ops = available_operations or list(OPERATIONS.keys())
        selected = random.sample(ops, min(n_operations, len(ops)))

        current = qa
        for op in selected:
            try:
                current = await self.inject(current, op, kg_context)
            except Exception as e:
                logger.warning(f"Uncertainty injection '{op}' failed: {e}")

        return current
