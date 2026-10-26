"""Pick the cheapest model likely to get the question right."""

from __future__ import annotations

from .models import ParsedQuestion

CHEAP_TYPES = {"lookup", "narrative"}


def route(parsed: ParsedQuestion, cheap_model: str, strong_model: str) -> tuple[str, str]:
    if len(parsed.companies) > 1:
        return strong_model, "multiple companies"
    if len(parsed.fiscal_years) > 1:
        return strong_model, "multiple periods"
    if parsed.question_type not in CHEAP_TYPES:
        return strong_model, f"{parsed.question_type} question"
    return cheap_model, f"single {parsed.question_type}"
