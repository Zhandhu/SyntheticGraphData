"""ItemWriter Agent — generates seed QA pairs and escalates complexity.

Paper 1, Section 3.1 (seed mode): 'An ItemWriter Agent is then prompted
with these composite units to generate seed question-answer (QA) pairs.'

Paper 1, Section 3.2 (escalation mode): 'The tool-augmented agent iteratively
evolves both the question and answer to increase their cognitive complexity.'
"""

import json
import re
from typing import Optional

from pydantic import BaseModel

from src.agents.base_agent import BaseAgent
from src.utils.data_models import (
    QAPair, CompositeUnit, EscalationResult, ToolCallRecord, ToolType,
)
from src.tools.tool_registry import ToolRegistry
from src.utils.llm_client import LLMClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SeedQAResponse(BaseModel):
    question: str
    answer: str
    reasoning: str
    sources_used: list[str] = []


class EscalationResponse(BaseModel):
    strategy: str
    reasoning: str
    question: str
    answer: str


class ItemWriterAgent(BaseAgent):
    """Two-mode agent: seed QA generation and complexity escalation."""

    def __init__(
        self,
        llm_client: LLMClient,
        seed_prompt_path: str = "prompts/item_writer_seed.txt",
        escalate_prompt_path: str = "prompts/item_writer_escalate.txt",
    ):
        super().__init__(llm_client, seed_prompt_path, "item_writer")
        with open(escalate_prompt_path) as f:
            self.escalate_template = f.read()

    async def run(self, composite_unit: CompositeUnit) -> QAPair:
        """Generate a seed QA pair from a composite unit (Mode 1)."""
        return await self.generate_seed(composite_unit)

    async def generate_seed(self, composite_unit: CompositeUnit) -> QAPair:
        """Mode 1: Generate seed QA from composite unit."""
        # Format sources from chunks
        sources_text = ""
        for i, chunk in enumerate(composite_unit.chunks):
            sources_text += (
                f"\n--- Source {i + 1} (ID: {chunk.id}) ---\n"
                f"Domain: {chunk.domain}\n"
                f"Entities: {', '.join(chunk.entities)}\n"
                f"Content: {chunk.summary}\n"
            )

        messages = self.format_prompt(sources=sources_text)
        response = await self.llm.complete_json(
            messages, SeedQAResponse, params_key=self.call_params_key
        )

        return QAPair(
            question=response.question,
            answer=response.answer,
            source_chunk_ids=[c.id for c in composite_unit.chunks],
            domain=composite_unit.chunks[0].domain if composite_unit.chunks else "",
            metadata={
                "reasoning": response.reasoning,
                "sources_used": response.sources_used,
            },
        )

    async def escalate(
        self,
        qa: QAPair,
        tool_registry: ToolRegistry,
        max_tool_calls: int = 5,
    ) -> EscalationResult:
        """Mode 2: Escalate complexity via ReAct-style tool loop."""
        tool_descriptions = tool_registry.get_descriptions_for_prompt()
        system_msg = self.escalate_template.format(
            question=qa.question,
            answer=qa.answer,
            tool_descriptions=tool_descriptions,
        )

        conversation = [
            {"role": "system", "content": system_msg},
        ]
        tool_calls: list[ToolCallRecord] = []
        required_tools: list[ToolType] = []

        # ReAct loop
        for _ in range(max_tool_calls + 1):
            response_text = await self.llm.complete(
                conversation, params_key=self.call_params_key
            )
            conversation.append({"role": "assistant", "content": response_text})

            # Check for tool calls
            tool_call_match = re.search(
                r"TOOL_CALL:\s*(\w+)\s*\|\s*(.+)", response_text
            )

            if tool_call_match and len(tool_calls) < max_tool_calls:
                tool_name = tool_call_match.group(1).strip().lower()
                tool_query = tool_call_match.group(2).strip()

                try:
                    result = await tool_registry.execute(tool_name, query=tool_query)
                    result_text = str(result.get("result", ""))[:2000]
                except Exception as e:
                    result_text = f"Tool error: {e}"

                tool_calls.append(ToolCallRecord(
                    tool=ToolType(tool_name),
                    query=tool_query,
                    result_summary=result_text[:500],
                ))
                required_tools.append(ToolType(tool_name))

                conversation.append({
                    "role": "user",
                    "content": f"OBSERVATION: {result_text}",
                })
                continue

            # Try to parse final JSON response
            parsed = self._extract_escalation_response(response_text)
            if parsed:
                escalated_qa = QAPair(
                    question=parsed.question,
                    answer=parsed.answer,
                    domain=qa.domain,
                    source_chunk_ids=qa.source_chunk_ids,
                    required_tools=required_tools,
                    metadata={"strategy": parsed.strategy},
                )
                return EscalationResult(
                    original=qa,
                    escalated=escalated_qa,
                    strategy=parsed.strategy,
                    tool_calls=tool_calls,
                    reasoning=parsed.reasoning,
                )

        # If loop exhausted without valid response, return original
        logger.warning("Escalation loop exhausted without valid response")
        return EscalationResult(
            original=qa,
            escalated=qa,
            strategy="none",
            tool_calls=tool_calls,
            reasoning="Escalation loop exhausted",
        )

    def _extract_escalation_response(self, text: str) -> Optional[EscalationResponse]:
        """Try to extract EscalationResponse JSON from text."""
        # Try markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return EscalationResponse.model_validate_json(match.group(1))
            except Exception:
                pass

        # Try raw JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return EscalationResponse.model_validate_json(match.group(0))
            except Exception:
                pass

        return None
