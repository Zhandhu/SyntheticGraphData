"""Academic search tool using Semantic Scholar API."""

import hashlib
import json
import os
from typing import Any

import httpx

from src.tools.tool_registry import BaseTool
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ScholarTool(BaseTool):
    name = "scholar"
    description = (
        "Search academic literature. Input: a search query. "
        "Returns top-5 papers with title, authors, year, venue, abstract, "
        "citation count, and URL."
    )

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.api_key = cfg.get("api_key", os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""))
        self.max_results = cfg.get("max_results", 5)
        self.cache_enabled = cfg.get("cache_enabled", True)
        self.cache_dir = cfg.get("cache_dir", "data/.cache/scholar/")
        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    async def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Execute academic search."""
        # Check cache
        if self.cache_enabled:
            cached = self._load_cache(query)
            if cached is not None:
                return {"result": cached, "raw": cached, "cached": True}

        results = await self._call_semantic_scholar(query)

        if self.cache_enabled:
            self._save_cache(query, results)

        return {"result": results, "raw": results, "cached": False}

    async def _call_semantic_scholar(self, query: str) -> list[dict]:
        """Call Semantic Scholar API."""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": self.max_results,
            "fields": "title,authors,year,venue,abstract,citationCount,url",
        }

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

        results = []
        for paper in data.get("data", [])[:self.max_results]:
            authors = [a.get("name", "") for a in paper.get("authors", [])]
            results.append({
                "title": paper.get("title", ""),
                "authors": authors,
                "year": paper.get("year"),
                "venue": paper.get("venue", ""),
                "abstract": paper.get("abstract", ""),
                "citation_count": paper.get("citationCount", 0),
                "url": paper.get("url", ""),
            })
        return results

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
