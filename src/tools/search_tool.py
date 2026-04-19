"""Web search tool — Brave Search API with DuckDuckGo fallback.

Brave Search API (https://api.search.brave.com) is the primary backend:
  - No aggressive rate-limiting
  - Free tier: 2000 queries/month
  - Requires BRAVE_API_KEY env var or config key

Falls back to DuckDuckGo if no Brave key is configured.
"""

import hashlib
import json
import os
from typing import Any

import httpx

from src.tools.tool_registry import BaseTool
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


class SearchTool(BaseTool):
    name = "search"
    description = (
        "Search the web. Input: a search query string. "
        "Returns top results with title, snippet, and URL."
    )

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.max_results = cfg.get("max_results", 10)
        self.cache_enabled = cfg.get("cache_enabled", True)
        self.cache_dir = cfg.get("cache_dir", "data/.cache/search/")
        # Brave API key: config > env var
        self.brave_api_key = cfg.get("api_key") or os.environ.get("BRAVE_API_KEY", "")
        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    async def execute(self, query: str = "", queries: list[str] | None = None, **kwargs: Any) -> dict:
        if queries:
            all_results = []
            for q in queries:
                all_results.append(await self._search_single(q))
            return {"result": all_results, "raw": all_results, "cached": False}

        results = await self._search_single(query)
        return {"result": results, "raw": results, "cached": False}

    async def _search_single(self, query: str) -> list[dict]:
        if self.cache_enabled:
            cached = self._load_cache(query)
            if cached is not None:
                return cached

        if self.brave_api_key:
            results = await self._call_brave(query)
        else:
            logger.warning("No BRAVE_API_KEY set — falling back to DuckDuckGo")
            results = await self._call_ddg(query)

        if self.cache_enabled and results:
            self._save_cache(query, results)

        return results

    async def _call_brave(self, query: str) -> list[dict]:
        """Call Brave Search API."""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_api_key,
        }
        params = {
            "q": query,
            "count": min(self.max_results, 20),
            "search_lang": "en",
            "safesearch": "moderate",
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(BRAVE_API_URL, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()

            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "url": item.get("url", ""),
                })
            logger.info(f"  brave_search('{query}') -> {len(results)} results")
            return results

        except httpx.HTTPStatusError as e:
            logger.error(f"Brave Search API error {e.response.status_code}: {e.response.text[:200]}")
            return []
        except Exception as e:
            logger.error(f"Brave Search request failed: {e}")
            return []

    async def _call_ddg(self, query: str) -> list[dict]:
        """DuckDuckGo fallback (rate-limited — use only when no Brave key)."""
        import asyncio
        from duckduckgo_search import DDGS

        def _sync_search() -> list[dict]:
            results = []
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=self.max_results):
                        results.append({
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "url": r.get("href", ""),
                        })
            except Exception as e:
                logger.warning(f"DuckDuckGo search failed: {e}")
            return results

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_search)

    def _cache_key(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _load_cache(self, query: str) -> list[dict] | None:
        path = os.path.join(self.cache_dir, f"{self._cache_key(query)}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, query: str, results: list[dict]) -> None:
        path = os.path.join(self.cache_dir, f"{self._cache_key(query)}.json")
        with open(path, "w") as f:
            json.dump(results, f)
