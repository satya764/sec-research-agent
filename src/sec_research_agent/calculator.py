"""Safe arithmetic. The agent sends every calculation here instead of doing math in its head."""

from __future__ import annotations

import ast
import math
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
        # Floats, not Python ints: an int can grow without limit, so nested powers like
        # ((9**10)**10)**10... would run for minutes. A float overflows at once instead.
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise CalculatorError("exponent too large")
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise CalculatorError("division by zero")
        try:
            result = BINARY[type(node.op)](left, right)
        except OverflowError:
            raise CalculatorError("result too large") from None
        if isinstance(result, complex):  # e.g. (-8) ** 0.5
            raise CalculatorError("result is not a real number")
        if not math.isfinite(result):
            raise CalculatorError("result too large")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return UNARY[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
        and not node.keywords
    ):
        args = [_eval(a) for a in node.args]
        name = node.func.id
        if not args or (name == "round" and len(args) > 2) or (name == "abs" and len(args) != 1):
            raise CalculatorError(f"wrong number of arguments for {name}")
        if name == "round" and len(args) == 2:
            args[1] = int(args[1])  # round() needs an int for the number of digits
        return FUNCTIONS[name](*args)
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
