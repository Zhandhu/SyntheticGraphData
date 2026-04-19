"""Document Reader tool — extract text from PDFs and technical documents."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.tools.tool_registry import BaseTool
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DocumentReaderTool(BaseTool):
    name = "document_reader"
    description = (
        "Read and extract text from a local document (PDF, TXT, or Markdown). "
        "Input: a file path to the document, optionally with a page range "
        "(e.g., 'path/to/doc.pdf pages=1-5'). Returns extracted text content."
    )

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.max_chars = cfg.get("max_chars", 15000)
        self.cache_enabled = cfg.get("cache_enabled", True)
        self.cache_dir = cfg.get("cache_dir", "data/.cache/documents/")
        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    async def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Read a document and return its text content."""
        # Parse query for path and optional page range
        parts = query.strip().split("pages=")
        filepath = parts[0].strip()
        page_range = parts[1].strip() if len(parts) > 1 else None

        if not filepath:
            return {"result": "No file path provided.", "raw": ""}

        if not os.path.exists(filepath):
            return {"result": f"File not found: {filepath}", "raw": ""}

        # Check cache
        cache_key = f"{filepath}|{page_range}"
        if self.cache_enabled:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return {"result": cached, "raw": cached, "cached": True}

        ext = Path(filepath).suffix.lower()
        try:
            if ext == ".pdf":
                text = self._read_pdf(filepath, page_range)
            elif ext in (".txt", ".text"):
                text = self._read_text(filepath)
            elif ext in (".md", ".markdown"):
                text = self._read_text(filepath)
            else:
                text = self._read_text(filepath)
        except Exception as e:
            return {"result": f"Failed to read document: {e}", "raw": ""}

        if len(text) > self.max_chars:
            text = text[:self.max_chars] + "\n\n[... truncated]"

        if self.cache_enabled:
            self._save_cache(cache_key, text)

        return {"result": text, "raw": text, "cached": False}

    def _read_pdf(self, filepath: str, page_range: str | None = None) -> str:
        """Extract text from PDF using PyPDF2 or pdfplumber."""
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(filepath) as pdf:
                start, end = self._parse_page_range(page_range, len(pdf.pages))
                for i in range(start, end):
                    page = pdf.pages[i]
                    text = page.extract_text() or ""
                    if text:
                        text_parts.append(f"--- Page {i + 1} ---\n{text}")
            return "\n\n".join(text_parts)
        except ImportError:
            pass

        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            text_parts = []
            start, end = self._parse_page_range(page_range, len(reader.pages))
            for i in range(start, end):
                text = reader.pages[i].extract_text() or ""
                if text:
                    text_parts.append(f"--- Page {i + 1} ---\n{text}")
            return "\n\n".join(text_parts)
        except ImportError:
            return "PDF reading requires 'pdfplumber' or 'PyPDF2'. Install with: pip install pdfplumber"

    def _read_text(self, filepath: str) -> str:
        """Read a plain text file."""
        with open(filepath, encoding="utf-8", errors="replace") as f:
            return f.read()

    @staticmethod
    def _parse_page_range(page_range: str | None, total_pages: int) -> tuple[int, int]:
        """Parse 'start-end' into 0-indexed (start, end)."""
        if not page_range:
            return 0, total_pages
        try:
            if "-" in page_range:
                s, e = page_range.split("-", 1)
                start = max(0, int(s) - 1)
                end = min(total_pages, int(e))
            else:
                page = int(page_range) - 1
                start = max(0, page)
                end = min(total_pages, page + 1)
            return start, end
        except ValueError:
            return 0, total_pages

    def _cache_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _load_cache(self, key: str) -> str | None:
        path = os.path.join(self.cache_dir, f"{self._cache_key(key)}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f).get("content", "")
        return None

    def _save_cache(self, key: str, content: str) -> None:
        path = os.path.join(self.cache_dir, f"{self._cache_key(key)}.json")
        with open(path, "w") as f:
            json.dump({"key": key, "content": content}, f)
