"""QuestionSolver Agent — answers questions in base or advanced mode.

Paper 1, Section 3.3:
Base mode: 'A QuestionSolver Agent, operating in a baseline mode without
access to tools, attempts to answer each question.'
Advanced mode: 'The remaining challenging pairs are passed to the same
QuestionSolver Agent, now operating in an advanced, tool-augmented mode.'
"""

import re
from typing import Optional

from pydantic import BaseModel

from src.agents.base_agent import BaseAgent
from src.tools.tool_registry import ToolRegistry
from src.utils.data_models import Trajectory, TrajectoryStep
from src.utils.llm_client import LLMClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SolverResponse(BaseModel):
    answer: str
    reasoning: str
    trajectory: Optional[Trajectory] = None


class QuestionSolverAgent(BaseAgent):
    """Two-mode solver: base (no tools) and advanced (with tools)."""

    def __init__(
        self,
        llm_client: LLMClient,
        base_prompt_path: str = "prompts/question_solver_base.txt",
        advanced_prompt_path: str = "prompts/question_solver_advanced.txt",
        tool_registry: Optional[ToolRegistry] = None,
    ):
        super().__init__(llm_client, base_prompt_path, "solver_base")
        with open(advanced_prompt_path) as f:
            self.advanced_template = f.read()
        self.tool_registry = tool_registry

    async def run(self, question: str, mode: str = "base") -> SolverResponse:
        """Dispatch to base or advanced mode."""
        if mode == "base":
            return await self.solve_base(question)
        return await self.solve_advanced(question)

    async def solve_base(self, question: str) -> SolverResponse:
        """Base mode: answer using only LLM knowledge, no tools."""
        messages = self.format_prompt(question=question)
        return await self.llm.complete_json(
            messages, SolverResponse, params_key="solver_base"
        )

    async def solve_advanced(
        self, question: str, max_tool_calls: int = 10, qa_id: str = ""
    ) -> SolverResponse:
        """Advanced mode: ReAct-style tool loop to answer the question."""
        if not self.tool_registry:
            raise RuntimeError("Tool registry required for advanced mode")

        tool_descriptions = self.tool_registry.get_descriptions_for_prompt()
        system_msg = self.advanced_template.format(
            tool_descriptions=tool_descriptions,
            question=question,
        )

        conversation = [{"role": "system", "content": system_msg}]
        tool_call_count = 0
        traj_steps: list[TrajectoryStep] = []
        tools_used: list[str] = []

        for _ in range(max_tool_calls + 1):
            response_text = await self.llm.complete(
                conversation, params_key="solver_advanced"
            )
            conversation.append({"role": "assistant", "content": response_text})

            # Check for tool calls
            tool_call_match = re.search(
                r"TOOL_CALL:\s*(\w+)\s*\|\s*(.+)", response_text
            )

            if tool_call_match and tool_call_count < max_tool_calls:
                tool_name = tool_call_match.group(1).strip().lower()
                tool_query = tool_call_match.group(2).strip()
                tool_call_count += 1

                try:
                    result = await self.tool_registry.execute(
                        tool_name, query=tool_query
                    )
                    result_text = str(result.get("result", ""))[:2000]
                except Exception as e:
                    result_text = f"Tool error: {e}"

                obs_text = f"OBSERVATION: {result_text}"
                traj_steps.append(TrajectoryStep(
                    role="assistant",
                    content=response_text,
                    tool=tool_name,
                    tool_query=tool_query,
                    observation=result_text,
                ))
                tools_used.append(tool_name)
                conversation.append({"role": "user", "content": obs_text})
                continue

            # Try to parse final answer
            parsed = self._extract_response(response_text)
            if parsed:
                traj_steps.append(TrajectoryStep(role="assistant", content=response_text))
                parsed.trajectory = Trajectory(
                    qa_id=qa_id,
                    mode="react",
                    steps=traj_steps,
                    final_answer=parsed.answer,
                    tool_calls_used=tools_used,
                )
                return parsed

        # Fallback
        logger.warning("Solver loop exhausted without valid JSON response")
        return SolverResponse(
            answer="",
            reasoning="Failed to produce answer",
            trajectory=Trajectory(qa_id=qa_id, mode="react", steps=traj_steps, tool_calls_used=tools_used),
        )

    def _extract_response(self, text: str) -> Optional[SolverResponse]:
        """Try to extract SolverResponse JSON from text."""
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return SolverResponse.model_validate_json(match.group(1))
            except Exception:
                pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return SolverResponse.model_validate_json(match.group(0))
            except Exception:
                pass

        return None
