"""Comparison tool — systematically compare entities across attributes.

Useful for questions that require comparing multiple entities on specific
dimensions (e.g., comparing reactor types, isotope properties, etc.).
"""

import json
from typing import Any

from src.tools.tool_registry import BaseTool
from src.utils.llm_client import LLMClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ComparisonTool(BaseTool):
    name = "comparison"
    description = (
        "Compare two or more entities across specified attributes using LLM knowledge. "
        "Input: a comparison request like 'compare <entity1> and <entity2> on <attributes>'. "
        "Returns a structured comparison table."
    )

    def __init__(self, llm_client: LLMClient, config: dict | None = None):
        self.llm = llm_client
        cfg = config or {}
        self.max_entities = cfg.get("max_entities", 5)

    async def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Generate a structured comparison."""
        if not query.strip():
            return {"result": "No comparison query provided.", "raw": ""}

        prompt = (
            "You are a factual comparison engine. Given the following comparison request, "
            "produce a structured comparison.\n\n"
            f"Request: {query}\n\n"
            "Return a JSON object with this structure:\n"
            "{\n"
            '  "entities": ["entity1", "entity2", ...],\n'
            '  "attributes": ["attr1", "attr2", ...],\n'
            '  "comparison": {\n'
            '    "entity1": {"attr1": "value", "attr2": "value"},\n'
            '    "entity2": {"attr1": "value", "attr2": "value"}\n'
            "  },\n"
            '  "summary": "brief summary of key differences"\n'
            "}"
        )

        messages = [{"role": "user", "content": prompt}]
        raw_response = await self.llm.complete(messages, params_key="summary_agent")

        # Try to format as a readable table
        try:
            # Extract JSON
            import re
            match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                result = self._format_table(data)
                return {"result": result, "raw": data}
        except Exception:
            pass

        return {"result": raw_response[:3000], "raw": raw_response}

    @staticmethod
    def _format_table(data: dict) -> str:
        """Format comparison data as a readable table."""
        entities = data.get("entities", [])
        attributes = data.get("attributes", [])
        comparison = data.get("comparison", {})
        summary = data.get("summary", "")

        if not entities or not attributes:
            return json.dumps(data, indent=2)

        # Build text table
        lines = []

        # Header
        header = f"{'Attribute':<25}" + "".join(f"{e:<25}" for e in entities)
        lines.append(header)
        lines.append("-" * len(header))

        # Rows
        for attr in attributes:
            row = f"{attr:<25}"
            for entity in entities:
                val = comparison.get(entity, {}).get(attr, "N/A")
                row += f"{str(val):<25}"
            lines.append(row)

        if summary:
            lines.append(f"\nSummary: {summary}")

        return "\n".join(lines)
