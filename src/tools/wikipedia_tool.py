"""Wikipedia lookup tool — direct Wikipedia API access for factual retrieval."""

import hashlib
import json
import os
from typing import Any

import httpx

from src.tools.tool_registry import BaseTool
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class WikipediaTool(BaseTool):
    name = "wikipedia"
    description = (
        "Look up a Wikipedia article by topic. Returns the article summary "
        "and key sections. Input: a topic or article title."
    )

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.language = cfg.get("language", "en")
        self.max_chars = cfg.get("max_chars", 8000)
        self.cache_enabled = cfg.get("cache_enabled", True)
        self.cache_dir = cfg.get("cache_dir", "data/.cache/wikipedia/")
        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    async def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Fetch a Wikipedia article summary."""
        if self.cache_enabled:
            cached = self._load_cache(query)
            if cached is not None:
                return {"result": cached, "raw": cached, "cached": True}

        result = await self._fetch_article(query)

        if self.cache_enabled:
            self._save_cache(query, result)

        return {"result": result, "raw": result, "cached": False}

    async def _fetch_article(self, topic: str) -> str:
        """Fetch article via Wikipedia REST API."""
        url = f"https://{self.language}.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"

        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    title = data.get("title", topic)
                    extract = data.get("extract", "")
                    description = data.get("description", "")

                    # Also try to get more content from the full article
                    full_url = f"https://{self.language}.wikipedia.org/w/api.php"
                    params = {
                        "action": "query",
                        "titles": topic,
                        "prop": "extracts",
                        "explaintext": True,
                        "exlimit": 1,
                        "format": "json",
                    }
                    full_response = await client.get(full_url, params=params)
                    full_text = ""
                    if full_response.status_code == 200:
                        pages = full_response.json().get("query", {}).get("pages", {})
                        for page in pages.values():
                            full_text = page.get("extract", "")

                    content = full_text if full_text else extract
                    if len(content) > self.max_chars:
                        content = content[:self.max_chars] + "..."

                    return f"# {title}\n{description}\n\n{content}"
                else:
                    return f"Wikipedia article not found for: {topic}"
            except Exception as e:
                logger.warning(f"Wikipedia lookup failed for '{topic}': {e}")
                return f"Wikipedia lookup failed: {e}"

    def _cache_key(self, query: str) -> str:
        return hashlib.sha256(query.lower().encode()).hexdigest()[:16]

    def _load_cache(self, query: str) -> str | None:
        path = os.path.join(self.cache_dir, f"{self._cache_key(query)}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f).get("content", "")
        return None

    def _save_cache(self, query: str, content: str) -> None:
        path = os.path.join(self.cache_dir, f"{self._cache_key(query)}.json")
        with open(path, "w") as f:
            json.dump({"query": query, "content": content}, f)
