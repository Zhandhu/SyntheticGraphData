"""Timeline tool — temporal reasoning and date arithmetic.

Useful for questions involving chronological ordering, date differences,
and historical period overlap.
"""

import re
from datetime import datetime, timedelta
from typing import Any

from src.tools.tool_registry import BaseTool
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TimelineTool(BaseTool):
    name = "timeline"
    description = (
        "Perform temporal reasoning and date arithmetic. Supported operations:\n"
        "  - 'diff <date1> <date2>': Calculate difference between two dates (YYYY-MM-DD or YYYY)\n"
        "  - 'order <event1:date1>, <event2:date2>, ...': Sort events chronologically\n"
        "  - 'overlap <start1>-<end1> <start2>-<end2>': Check if two periods overlap\n"
        "  - 'add <date> <N> days|months|years': Add duration to a date\n"
        "Input: one of the operations above."
    )

    async def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Execute a temporal reasoning operation."""
        query = query.strip()
        if not query:
            return {"result": "No query provided.", "raw": ""}

        lower = query.lower()

        if lower.startswith("diff"):
            return self._date_diff(query[4:].strip())
        elif lower.startswith("order"):
            return self._chronological_order(query[5:].strip())
        elif lower.startswith("overlap"):
            return self._check_overlap(query[7:].strip())
        elif lower.startswith("add"):
            return self._date_add(query[3:].strip())
        else:
            return {"result": f"Unknown operation. Use: diff, order, overlap, or add.", "raw": ""}

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse a date string in various formats."""
        date_str = date_str.strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        # Try year-only with BCE
        match = re.match(r"(\d+)\s*BC(?:E)?", date_str, re.IGNORECASE)
        if match:
            year = -int(match.group(1))
            return datetime(1, 1, 1).replace(year=max(1, abs(year)))
        return None

    def _date_diff(self, args: str) -> dict:
        """Calculate difference between two dates."""
        # Split on common separators
        parts = re.split(r"\s+(?:and|to|vs|,)\s+|\s{2,}", args, maxsplit=1)
        if len(parts) < 2:
            parts = args.split(maxsplit=1)
        if len(parts) < 2:
            return {"result": "Provide two dates separated by space or 'and'.", "raw": ""}

        d1 = self._parse_date(parts[0])
        d2 = self._parse_date(parts[1])
        if not d1 or not d2:
            return {"result": f"Could not parse dates: '{parts[0]}' and/or '{parts[1]}'", "raw": ""}

        delta = abs((d2 - d1).days)
        years = delta // 365
        months = (delta % 365) // 30
        days = (delta % 365) % 30

        result = f"Difference: {delta} days ({years} years, {months} months, {days} days)"
        return {"result": result, "raw": result}

    def _chronological_order(self, args: str) -> dict:
        """Sort events by date."""
        events = []
        for item in re.split(r",\s*", args):
            match = re.match(r"(.+?)\s*:\s*(.+)", item.strip())
            if match:
                name = match.group(1).strip()
                date = self._parse_date(match.group(2).strip())
                if date:
                    events.append((name, date))

        if not events:
            return {"result": "No valid events found. Use format: 'event1:date1, event2:date2'", "raw": ""}

        events.sort(key=lambda x: x[1])
        lines = ["Chronological order:"]
        for i, (name, date) in enumerate(events, 1):
            lines.append(f"  {i}. {name} ({date.strftime('%Y-%m-%d')})")

        result = "\n".join(lines)
        return {"result": result, "raw": result}

    def _check_overlap(self, args: str) -> dict:
        """Check if two time periods overlap."""
        ranges = re.findall(r"(\S+)\s*-\s*(\S+)", args)
        if len(ranges) < 2:
            return {"result": "Provide two ranges: '<start1>-<end1> <start2>-<end2>'", "raw": ""}

        s1, e1 = self._parse_date(ranges[0][0]), self._parse_date(ranges[0][1])
        s2, e2 = self._parse_date(ranges[1][0]), self._parse_date(ranges[1][1])

        if not all([s1, e1, s2, e2]):
            return {"result": "Could not parse all dates.", "raw": ""}

        overlaps = s1 <= e2 and s2 <= e1
        if overlaps:
            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            days = (overlap_end - overlap_start).days
            result = f"Periods OVERLAP by {days} days ({overlap_start.strftime('%Y-%m-%d')} to {overlap_end.strftime('%Y-%m-%d')})"
        else:
            gap = abs((min(s1, s2) - max(e1, e2)).days)
            result = f"Periods do NOT overlap. Gap: {gap} days."

        return {"result": result, "raw": result}

    def _date_add(self, args: str) -> dict:
        """Add a duration to a date."""
        match = re.match(r"(\S+)\s+(\d+)\s+(day|month|year|week)s?", args, re.IGNORECASE)
        if not match:
            return {"result": "Use format: '<date> <N> days|months|years'", "raw": ""}

        base = self._parse_date(match.group(1))
        if not base:
            return {"result": f"Could not parse date: {match.group(1)}", "raw": ""}

        n = int(match.group(2))
        unit = match.group(3).lower()

        if unit == "day":
            result_date = base + timedelta(days=n)
        elif unit == "week":
            result_date = base + timedelta(weeks=n)
        elif unit == "month":
            month = base.month + n
            year = base.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            day = min(base.day, 28)
            result_date = base.replace(year=year, month=month, day=day)
        elif unit == "year":
            result_date = base.replace(year=base.year + n)
        else:
            return {"result": f"Unknown unit: {unit}", "raw": ""}

        result = f"{base.strftime('%Y-%m-%d')} + {n} {unit}s = {result_date.strftime('%Y-%m-%d')}"
        return {"result": result, "raw": result}
