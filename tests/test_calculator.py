import time

import pytest

from sec_research_agent.calculator import CalculatorError, evaluate


def test_basic_math():
    assert evaluate("1 + 2 * 3") == 7
    assert evaluate("(391035000000 - 383285000000) / 383285000000 * 100") == pytest.approx(2.0220, rel=1e-3)
    assert evaluate("-5 + abs(-3)") == -2
    assert evaluate("round(2.456, 2)") == 2.46
    assert evaluate("max(1, 7, 3)") == 7


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')", "open('x')", "1 / 0", "2 ** 100", "1,000 + 2", "a + 1", "x" * 400,
        "max()", "abs(1, 2)", "1e308 * 10", "(-8) ** 0.5",
    ],
)
def test_rejects_unsafe_or_invalid(expr):
    with pytest.raises(CalculatorError):
        evaluate(expr)


def test_nested_powers_fail_fast():
    # Each exponent is within the limit, but nested they would build a number with millions of digits.
    expr = "(" * 8 + "9**10" + ")**10" * 8
    started = time.perf_counter()
    with pytest.raises(CalculatorError):
        evaluate(expr)
    assert time.perf_counter() - started < 1
