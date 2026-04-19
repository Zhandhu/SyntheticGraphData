"""Judge Agent — assesses answer correctness against ground truth.

Paper 1, Section 3.3: 'A Judge Agent automatically assesses the
correctness of the solver's output against the ground-truth answer.'
"""

from pydantic import BaseModel

from src.agents.base_agent import BaseAgent
from src.utils.llm_client import LLMClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class JudgeResponse(BaseModel):
    correct: bool
    explanation: str


class JudgeAgent(BaseAgent):
    """Strict answer correctness judge."""

    def __init__(self, llm_client: LLMClient, prompt_path: str = "prompts/judge.txt"):
        super().__init__(llm_client, prompt_path, "judge")

    async def run(self, reference: str, predicted: str) -> JudgeResponse:
        """Judge whether predicted answer matches reference."""
        return await self.judge(reference, predicted)

    async def judge(self, reference: str, predicted: str,
                    complexity: float | None = None) -> bool:
        """Return True if predicted answer is correct.

        Strictness follows a U-curve over complexity:
          - Low  (< 0.3): lenient — small subgraphs produce obscure specific
            facts; solver may find the right answer in different phrasing.
          - Mid  (0.3–0.6): strict — well-formed questions have clear answers.
          - High (> 0.6): lenient — multi-hop answers are long; partial
            coverage of the core claim is acceptable.
        """
        if complexity is None:
            strictness = "strict"
        elif complexity < 0.45:
            # Low-to-mid: small subgraphs produce obscure specific facts;
            # solver often finds the right answer in different phrasing
            strictness = "lenient_paraphrase"
        elif complexity <= 0.55:
            # Narrow sweet spot: well-formed questions with clear single answers
            strictness = "strict"
        else:
            # High: multi-hop answers are long; partial coverage of core claim ok
            strictness = "lenient_partial"
        messages = self.format_prompt(
            reference=reference, predicted=predicted, strictness=strictness
        )
        response = await self.llm.complete_json(
            messages, JudgeResponse, params_key=self.call_params_key
        )
        return response.correct
