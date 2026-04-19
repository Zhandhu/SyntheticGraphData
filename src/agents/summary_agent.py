"""Summary Agent — preprocesses corpus text into information-dense chunks.

Paper 1, Section 3.1: 'A Summary Agent preprocesses this corpus by
paraphrasing content, removing artifacts (e.g., HTML tags), and distilling
the text into information-dense chunks.'
"""

from pydantic import BaseModel

from src.agents.base_agent import BaseAgent
from src.utils.data_models import TextChunk
from src.utils.llm_client import LLMClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SummaryResponse(BaseModel):
    summary: str
    entities: list[str]
    domain: str


class SummaryAgent(BaseAgent):
    """Process raw text into clean, entity-annotated TextChunks."""

    def __init__(self, llm_client: LLMClient, prompt_path: str = "prompts/summary_agent.txt"):
        super().__init__(llm_client, prompt_path, "summary_agent")

    async def run(self, raw_text: str, source: str = "") -> TextChunk:
        """Process a raw text passage into a TextChunk."""
        messages = self.format_prompt(raw_text=raw_text)
        response = await self.llm.complete_json(
            messages, SummaryResponse, params_key=self.call_params_key
        )

        return TextChunk(
            source=source,
            raw_text=raw_text,
            summary=response.summary,
            entities=response.entities,
            domain=response.domain,
        )
