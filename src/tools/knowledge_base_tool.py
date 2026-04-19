"""Knowledge Base tool — query structured data from local JSON/CSV knowledge bases.

Useful for domain-specific lookups (e.g., nuclear isotope data, element properties,
reactor specifications) where the data is stored as local structured files.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any

from src.tools.tool_registry import BaseTool
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class KnowledgeBaseTool(BaseTool):
    name = "knowledge_base"
    description = (
        "Query a structured knowledge base. Searches across local JSON and "
        "CSV data files for matching records. Input: a search query. "
        "The tool performs fuzzy text matching across all fields."
    )

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.data_dirs = cfg.get("data_dirs", ["data/knowledge_bases/"])
        self.max_results = cfg.get("max_results", 10)
        self._records: list[dict] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load all knowledge base files."""
        if self._loaded:
            return
        self._loaded = True

        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                continue

            for filepath in Path(data_dir).rglob("*.json"):
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for record in data:
                            if isinstance(record, dict):
                                record["_source_file"] = str(filepath)
                                self._records.append(record)
                    elif isinstance(data, dict):
                        # Could be a dict of lists or nested structure
                        for key, val in data.items():
                            if isinstance(val, list):
                                for item in val:
                                    if isinstance(item, dict):
                                        item["_source_file"] = str(filepath)
                                        item["_category"] = key
                                        self._records.append(item)
                except Exception as e:
                    logger.warning(f"Failed to load KB file {filepath}: {e}")

            for filepath in Path(data_dir).rglob("*.csv"):
                try:
                    with open(filepath) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            row["_source_file"] = str(filepath)
                            self._records.append(dict(row))
                except Exception as e:
                    logger.warning(f"Failed to load CSV file {filepath}: {e}")

        logger.info(f"Knowledge base loaded: {len(self._records)} records")

    async def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Search the knowledge base for matching records."""
        self._ensure_loaded()

        if not query.strip():
            return {"result": "No query provided.", "raw": ""}

        if not self._records:
            return {
                "result": "Knowledge base is empty. Add JSON/CSV files to data/knowledge_bases/.",
                "raw": "",
            }

        # Score each record by fuzzy text match
        query_lower = query.lower()
        query_terms = query_lower.split()
        scored = []

        for record in self._records:
            record_text = " ".join(
                str(v).lower() for k, v in record.items() if not k.startswith("_")
            )
            # Count matching terms
            score = sum(1 for term in query_terms if term in record_text)
            # Bonus for exact substring match
            if query_lower in record_text:
                score += len(query_terms)
            if score > 0:
                scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_results = [r for _, r in scored[:self.max_results]]

        if not top_results:
            return {"result": f"No results found for: {query}", "raw": ""}

        # Format results
        lines = [f"Found {len(top_results)} matching records:\n"]
        for i, record in enumerate(top_results, 1):
            display = {k: v for k, v in record.items() if not k.startswith("_")}
            lines.append(f"{i}. {json.dumps(display, ensure_ascii=False)}")

        result_text = "\n".join(lines)
        return {"result": result_text, "raw": top_results}
