"""Central tool dispatch. Each tool implements a common interface."""

from abc import ABC, abstractmethod
from typing import Any

import yaml
import os
import re

from src.utils.logging_utils import get_logger, log_tool_call, Timer

logger = get_logger(__name__)


class BaseTool(ABC):
    """Base class for all tools."""
    name: str
    description: str

    @abstractmethod
    async def execute(self, **kwargs: Any) -> dict:
        """Execute the tool. Returns {"result": ..., "raw": ...}"""
        ...


class ToolRegistry:
    """Registry and dispatcher for all tools."""

    def __init__(self, config_path: str = "configs/tool_config.yaml"):
        if os.path.exists(config_path):
            with open(config_path) as f:
                raw = f.read()
            for match in re.findall(r"\$\{(\w+)\}", raw):
                raw = raw.replace(f"${{{match}}}", os.environ.get(match, ""))
            self.config = yaml.safe_load(raw) or {}
        else:
            self.config = {}

        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool by its name."""
        self._tools[tool.name] = tool

    async def execute(self, tool_name: str, **kwargs: Any) -> dict:
        """Execute a registered tool by name.

        When called from ReAct loops, the agent typically passes a single
        'query' kwarg. This method normalises that into the tool-specific
        parameter (e.g., code= for python, url= for visit) so every tool
        gets what it expects.
        """
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}. Available: {list(self._tools.keys())}")

        # Normalise generic 'query' kwarg into tool-specific params
        if "query" in kwargs:
            q = kwargs["query"]
            if tool_name == "python" and "code" not in kwargs:
                kwargs["code"] = q
            elif tool_name == "visit" and "url" not in kwargs:
                # Try splitting "url | goal"
                if "|" in q:
                    parts = q.split("|", 1)
                    kwargs["url"] = parts[0].strip()
                    kwargs["goal"] = parts[1].strip()
                else:
                    kwargs["url"] = q

        tool = self._tools[tool_name]
        with Timer() as timer:
            result = await tool.execute(**kwargs)

        log_tool_call(
            logger,
            tool_name=tool_name,
            query=str(kwargs.get("query", kwargs.get("url", kwargs.get("code", ""))))[:200],
            cached=result.get("cached", False),
            latency_ms=timer.elapsed_ms,
        )
        return result

    def get_descriptions_for_prompt(self) -> str:
        """Format all tool descriptions for inclusion in LLM prompts."""
        lines = []
        for name, tool in self._tools.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)

    def get_tool_names(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())
