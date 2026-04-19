"""Code execution sandbox for computational tasks."""

import ast
import os
import subprocess
import tempfile
from typing import Any

from src.tools.tool_registry import BaseTool
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

ALLOWED_IMPORTS = {
    "math", "numpy", "scipy", "statistics", "collections", "itertools",
    "datetime", "json", "csv", "re", "fractions", "decimal",
}

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "requests",
    "urllib", "pathlib",
}


def _validate_imports(code: str) -> str | None:
    """Check that code only imports allowed modules. Returns error message or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module in BLOCKED_MODULES:
                    return f"Blocked import: {module}"
                if module not in ALLOWED_IMPORTS:
                    return f"Disallowed import: {module}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                if module in BLOCKED_MODULES:
                    return f"Blocked import: {module}"
                if module not in ALLOWED_IMPORTS:
                    return f"Disallowed import: {module}"
    return None


class PythonTool(BaseTool):
    name = "python"
    description = (
        "Execute Python code in a sandboxed environment for computational tasks. "
        "Supports standard libraries for data analysis: math, numpy, scipy, "
        "statistics, collections, itertools, datetime, json, csv, re, fractions, decimal. "
        "Input: Python code as a string. Returns stdout output."
    )

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.timeout = cfg.get("timeout", 30)

    async def execute(self, code: str = "", **kwargs: Any) -> dict:
        """Execute Python code in a subprocess sandbox."""
        if not code.strip():
            return {"result": "No code provided.", "raw": ""}

        # Validate imports
        error = _validate_imports(code)
        if error:
            return {"result": f"Import validation failed: {error}", "raw": ""}

        # Execute in temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)

            try:
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                )

                if result.returncode == 0:
                    output = result.stdout.strip()
                    return {"result": output, "raw": output}
                else:
                    error_output = result.stderr.strip()
                    return {"result": f"Execution error:\n{error_output}", "raw": error_output}

            except subprocess.TimeoutExpired:
                return {"result": f"Execution timed out after {self.timeout}s.", "raw": ""}
            except Exception as e:
                return {"result": f"Execution failed: {e}", "raw": ""}
