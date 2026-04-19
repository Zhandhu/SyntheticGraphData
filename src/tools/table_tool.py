"""Table extraction tool — fetches and parses HTML tables from web pages.

Tongyi paper §3.4.1 mentions 'isomorphic tables from real-world websites'
as a parallel knowledge source alongside the text-based KG.  This tool:
  1. Fetches a URL with httpx
  2. Extracts all <table> elements via BeautifulSoup
  3. Converts each table to a list of {header: value} row dicts
  4. Returns the structured rows (+ a text representation for atomisation)

Tables produce naturally structured facts — counts, dates, rankings,
measurements — which generate different question types than prose text.
"""

import hashlib
import json
import os
from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.tools.tool_registry import BaseTool
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TableTool(BaseTool):
    name = "table"
    description = (
        "Extract structured tables from a web page. "
        "Input: a URL and an optional topic filter. "
        "Returns table rows as structured data with headers."
    )

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.timeout = cfg.get("timeout", 15)
        self.max_tables = cfg.get("max_tables", 3)
        self.max_rows = cfg.get("max_rows", 20)
        self.cache_enabled = cfg.get("cache_enabled", True)
        self.cache_dir = cfg.get("cache_dir", "data/.cache/table/")
        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    async def execute(self, url: str = "", query: str = "", topic: str = "", **kwargs: Any) -> dict:
        """Fetch and parse tables from a URL.

        Accepts `url` directly, or falls back to `query` if url looks like a
        search query (no http prefix) — in that case returns empty (callers
        should use search + visit to resolve URLs first).
        """
        target_url = url or query
        if not target_url.startswith("http"):
            return {"result": [], "raw": [], "cached": False, "text": ""}

        if self.cache_enabled:
            cached = self._load_cache(target_url)
            if cached is not None:
                return {"result": cached["rows"], "raw": cached["rows"],
                        "text": cached["text"], "cached": True}

        html = await self._fetch_html(target_url)
        if not html:
            return {"result": [], "raw": [], "cached": False, "text": ""}

        tables = self._extract_tables(html, topic or query)
        text_repr = self._tables_to_text(tables)

        if self.cache_enabled and tables:
            self._save_cache(target_url, tables, text_repr)

        logger.info(f"  table('{target_url}') -> {len(tables)} tables, "
                    f"{sum(len(t['rows']) for t in tables)} total rows")

        return {"result": tables, "raw": tables, "text": text_repr, "cached": False}

    async def _fetch_html(self, url: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                return resp.text
        except Exception as e:
            logger.warning(f"TableTool fetch failed for {url}: {e}")
            return ""

    def _extract_tables(self, html: str, topic: str = "") -> list[dict]:
        """Parse all tables from HTML, returning structured row dicts."""
        soup = BeautifulSoup(html, "html.parser")
        results = []
        topic_lower = topic.lower()

        for table in soup.find_all("table")[:self.max_tables * 3]:
            # Extract headers
            headers = []
            header_row = table.find("tr")
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

            if not headers:
                continue

            # Extract data rows
            rows = []
            for tr in table.find_all("tr")[1:self.max_rows + 1]:
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if len(cells) == len(headers) and any(c for c in cells):
                    rows.append(dict(zip(headers, cells)))

            if not rows:
                continue

            # Score relevance to topic if provided
            table_text = " ".join(str(r) for r in rows[:5]).lower()
            if topic_lower and topic_lower not in table_text:
                # Allow if topic words partially match
                topic_words = topic_lower.split()
                if not any(w in table_text for w in topic_words if len(w) > 3):
                    continue

            # Infer a caption
            caption_el = table.find("caption")
            caption = caption_el.get_text(strip=True) if caption_el else ""
            if not caption:
                # Look for a heading just before the table
                prev = table.find_previous(["h1", "h2", "h3", "h4"])
                caption = prev.get_text(strip=True)[:80] if prev else ""

            results.append({
                "caption": caption,
                "headers": headers,
                "rows": rows[:self.max_rows],
            })

            if len(results) >= self.max_tables:
                break

        return results

    @staticmethod
    def _tables_to_text(tables: list[dict]) -> str:
        """Convert table list to a text representation for LLM atomisation."""
        lines = []
        for t in tables:
            if t.get("caption"):
                lines.append(f"Table: {t['caption']}")
            lines.append("  " + " | ".join(t["headers"]))
            for row in t["rows"][:10]:
                lines.append("  " + " | ".join(str(v) for v in row.values()))
            lines.append("")
        return "\n".join(lines)

    def _cache_key(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def _load_cache(self, url: str) -> dict | None:
        path = os.path.join(self.cache_dir, f"{self._cache_key(url)}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, url: str, rows: list, text: str) -> None:
        path = os.path.join(self.cache_dir, f"{self._cache_key(url)}.json")
        with open(path, "w") as f:
            json.dump({"url": url, "rows": rows, "text": text}, f)
