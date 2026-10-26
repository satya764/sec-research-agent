"""Safe arithmetic. The agent sends every calculation here instead of doing math in its head."""

from __future__ import annotations

import ast
import operator

BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
FUNCTIONS = {"abs": abs, "round": round, "min": min, "max": max}
MAX_LENGTH = 300


class CalculatorError(ValueError):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise CalculatorError("exponent too large")
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise CalculatorError("division by zero")
        return BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return UNARY[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
        and not node.keywords
    ):
        return FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Tuple):
        raise CalculatorError("write numbers without thousands separators, e.g. 391035000000")
    raise CalculatorError(f"unsupported expression: {ast.dump(node)[:60]}")


def evaluate(expression: str) -> float:
    if len(expression) > MAX_LENGTH:
        raise CalculatorError("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as err:
        raise CalculatorError(f"invalid expression: {err.msg}") from err
    return float(_eval(tree))
