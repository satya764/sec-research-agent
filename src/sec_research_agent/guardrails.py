"""Input checks: redact personal data before it reaches a model, catch prompt injection."""

from __future__ import annotations

import re

MAX_QUESTION_CHARS = 1000

# Applied in this order. Cards go first, so a shorter pattern can't redact part of a card number
# and leave the rest of its digits behind.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"(?<!\w)(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}\b"),
}

INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
        r"disregard (all |the |your )?(previous|prior|system|above)",
        r"reveal (your |the )?(system prompt|instructions|hidden)",
        r"you are now (a|an|in) ",
        r"new instructions\s*:",
        r"</?(system|assistant)>",
    )
]


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []

    def replace(kind: str):
        def _sub(match: re.Match[str]) -> str:
            if kind == "card":
                digits = re.sub(r"\D", "", match.group(0))
                if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                    return match.group(0)
            found.append(kind)
            return f"[{kind.upper()} REDACTED]"

        return _sub

    for kind, pattern in PII_PATTERNS.items():
        text = pattern.sub(replace(kind), text)
    return text, sorted(set(found))


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


def clean_retrieved(text: str) -> str:
    """Drop lines in retrieved filing text that try to give the model instructions."""
    return "\n".join(line for line in text.splitlines() if not looks_like_injection(line))
