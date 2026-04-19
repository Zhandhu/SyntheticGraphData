"""Page fetcher + goal-oriented summarizer."""

import hashlib
import json
import os
from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.tools.tool_registry import BaseTool
from src.utils.llm_client import LLMClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VisitTool(BaseTool):
    name = "visit"
    description = (
        "Visit a web page and extract information relevant to a specific goal. "
        "Input: a URL and an extraction goal. Returns a focused summary."
    )

    def __init__(self, llm_client: LLMClient, config: dict | None = None):
        cfg = config or {}
        self.llm = llm_client
        self.timeout = cfg.get("timeout", 15)
        self.max_content_tokens = cfg.get("max_content_tokens", 8000)
        self.cache_enabled = cfg.get("cache_enabled", True)
        self.cache_dir = cfg.get("cache_dir", "data/.cache/visit/")
        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    async def execute(self, url: str = "", goal: str = "", **kwargs: Any) -> dict:
        """Fetch a page and summarize it with respect to a goal."""
        # Check cache
        if self.cache_enabled:
            cached = self._load_cache(url, goal)
            if cached is not None:
                return {"result": cached, "raw": cached, "cached": True}

        # Step 1: Fetch page content
        content = await self._fetch_page(url)
        if not content:
            return {"result": "Failed to fetch page content.", "raw": "", "cached": False}

        # Step 2: Truncate if too long (~4 chars per token)
        max_chars = self.max_content_tokens * 4
        if len(content) > max_chars:
            content = content[:max_chars]

        # Step 3: Summarize with LLM
        summary = await self._summarize(content, goal)

        if self.cache_enabled:
            self._save_cache(url, goal, summary)

        return {"result": summary, "raw": content[:500], "cached": False}

    async def _fetch_page(self, url: str) -> str:
        """Fetch and extract main text from a URL."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                html = response.text

            soup = BeautifulSoup(html, "html.parser")
            # Remove scripts, styles, nav elements
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Collapse multiple newlines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return ""

    async def _summarize(self, content: str, goal: str) -> str:
        """Use LLM to extract goal-relevant information."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Given this webpage content, extract information relevant "
                    f"to the following goal: {goal}\n\nContent:\n{content}"
                ),
            }
        ]
        return await self.llm.complete(messages, params_key="summary_agent")

    def _cache_key(self, url: str, goal: str) -> str:
        key = f"{url}|{goal}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _load_cache(self, url: str, goal: str) -> str | None:
        path = os.path.join(self.cache_dir, f"{self._cache_key(url, goal)}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                return data.get("summary", "")
        return None

    def _save_cache(self, url: str, goal: str, summary: str) -> None:
        path = os.path.join(self.cache_dir, f"{self._cache_key(url, goal)}.json")
        with open(path, "w") as f:
            json.dump({"url": url, "goal": goal, "summary": summary}, f)
