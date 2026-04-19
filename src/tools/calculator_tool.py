"""Calculator tool — lightweight math evaluation without a full Python sandbox."""

import ast
import math
import operator
from typing import Any

from src.tools.tool_registry import BaseTool
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Safe operators for AST evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    # math module
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "pow": pow,
}

SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "tau": math.tau,
}


def _safe_eval(node: ast.AST) -> Any:
    """Recursively evaluate an AST node with only safe operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.Name):
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return SAFE_OPERATORS[op_type](_safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            args = [_safe_eval(arg) for arg in node.args]
            return SAFE_FUNCTIONS[node.func.id](*args)
        raise ValueError(f"Unsupported function call")
    elif isinstance(node, ast.List):
        return [_safe_eval(el) for el in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_safe_eval(el) for el in node.elts)
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression safely. Supports arithmetic "
        "(+, -, *, /, **, %), math functions (sqrt, log, sin, cos, etc.), "
        "and constants (pi, e). Input: a math expression as a string. "
        "Examples: '2**10', 'sqrt(144) + log(100)', 'factorial(10)'."
    )

    async def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Evaluate a math expression."""
        expression = query.strip()
        if not expression:
            return {"result": "No expression provided.", "raw": ""}

        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            result_str = str(result)
            return {"result": result_str, "raw": result_str}
        except Exception as e:
            return {"result": f"Calculation error: {e}", "raw": ""}
