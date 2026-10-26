"""Golden-set evals: retrieval recall@5 and MRR, answer accuracy, citation correctness,
refusal accuracy, cost per question, p95 latency.

Golden rows (evals/golden.jsonl), one JSON object per line:
  id, question, type (numeric | narrative | unanswerable), company,
  expected_value + tolerance_pct or tolerance_abs   (numeric)
  expected_answer, expected_section, fiscal_year      (narrative)
  verified: true only after you checked the answer against the filing yourself
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from .config import get_settings
from .facts import METRICS, extract_series
from .llm import LLM, StructuredOutputError, structured
from .models import AskResult, Filters


class Verdict(BaseModel):
    correct: bool
    reason: str


def load_golden(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def within(value: float, expected: float, tolerance_pct: float | None, tolerance_abs: float | None) -> bool:
    if tolerance_abs is not None:
        return abs(value - expected) <= tolerance_abs
    pct = tolerance_pct if tolerance_pct is not None else 0.5
    return abs(value - expected) <= abs(expected) * pct / 100


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _graded(fn, *args) -> bool | None:
    """Run a judge; None means the judge model failed to return a verdict (not counted)."""
    try:
        return fn(*args)
    except StructuredOutputError:
        return None


def judge_answer(llm: LLM, model: str, question: str, reference: str, answer: str) -> bool:
    verdict, _ = structured(
        llm, model=model,
        system="You grade answers to questions about company filings. Be strict about facts, lenient about wording.",
        prompt=f"Question: {question}\nReference answer: {reference}\nCandidate answer: {answer}\n"
               "Is the candidate answer factually consistent with the reference?",
        schema=Verdict, name="verdict", description="Grading result",
    )
    return verdict.correct


def judge_citations(llm: LLM, model: str, result: AskResult) -> bool:
    if not result.citations:
        return False
    sources = "\n\n".join(f"[{c.source_id}] {c.snippet}" for c in result.citations)
    verdict, _ = structured(
        llm, model=model,
        system="You check whether cited sources support an answer.",
        prompt=f"Answer: {result.answer}\n\nCited sources:\n{sources}\n\n"
               "Is every factual claim in the answer supported by these sources? "
               "Arithmetic derived from cited numbers counts as supported.",
        schema=Verdict, name="verdict", description="Grading result",
    )
    return verdict.correct


def _mean(values: list) -> float | None:
    values = [float(v) for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize(records: list[dict]) -> dict:
    return {
        "questions": len(records),
        "retrieval_recall_at_5": _mean([r.get("retrieval_hit") for r in records]),
        "retrieval_mrr": _mean([r.get("reciprocal_rank") for r in records]),
        "answer_accuracy": _mean([r.get("correct") for r in records]),
        "citation_correctness": _mean([r.get("citations_ok") for r in records]),
        "refusal_accuracy": _mean([r.get("refusal_correct") for r in records]),
        "avg_cost_usd": _mean([r["cost_usd"] for r in records]),
        "p95_latency_ms": p95([r["latency_ms"] for r in records]),
        "ungraded_by_judge": sum(1 for r in records if r.get("judge_failed")),
    }


def to_markdown(summary: dict, mode: str) -> str:
    def fmt(key: str, value) -> str:
        if value is None:
            return "n/a"
        if key == "avg_cost_usd":
            return f"${value:.4f}"
        if key == "p95_latency_ms":
            return f"{value / 1000:.1f} s"
        if key in ("questions", "ungraded_by_judge"):
            return str(value)
        return f"{value:.0%}"

    lines = [f"| Metric ({mode}) | Value |", "| --- | --- |"]
    lines += [f"| {k} | {fmt(k, v)} |" for k, v in summary.items()]
    return "\n".join(lines)


def run_evals(args: argparse.Namespace) -> int:
    from .factory import build_agent, build_llm, build_retriever, build_sec

    settings = get_settings()
    mode = args.mode or settings.retrieval_mode
    rows = load_golden(args.golden)
    if not args.include_unverified:
        rows = [r for r in rows if r.get("verified")]
    rows = rows[: args.limit] if args.limit else rows
    if not rows:
        print("No verified rows. Check answers by hand and set \"verified\": true, "
              "or pass --include-unverified.")
        return 0

    agent = build_agent(settings, mode)
    retriever = build_retriever(settings, mode)
    sec = build_sec(settings)
    judge = build_llm(settings)

    records = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['id']} ...", flush=True)
        result = agent.ask(row["question"])
        rec: dict = {"id": row["id"], "type": row["type"], "answer": result.answer,
                     "found": result.found, "value": result.value,
                     "cost_usd": result.cost_usd, "latency_ms": result.latency_ms,
                     "model": result.model,
                     "cited": [c.source_id for c in result.citations]}
        answerable = row["type"] != "unanswerable"
        rec["refusal_correct"] = result.found == answerable

        if row.get("expected_section"):
            ticker = sec.lookup(row["company"]).ticker
            hits = retriever.search(row["question"], Filters(ticker=ticker, fiscal_year=row.get("fiscal_year")), k=5)
            sections = [h.chunk.section for h in hits]
            rank = sections.index(row["expected_section"]) + 1 if row["expected_section"] in sections else None
            rec["retrieval_hit"] = rank is not None
            rec["reciprocal_rank"] = 1 / rank if rank else 0.0

        if not answerable:
            rec["correct"] = not result.found
        elif row["type"] == "numeric":
            rec["correct"] = (
                result.found and result.value is not None
                and within(result.value, row["expected_value"], row.get("tolerance_pct"), row.get("tolerance_abs"))
            )
        elif row.get("expected_answer"):
            rec["correct"] = _graded(judge_answer, judge, settings.judge_model, row["question"], row["expected_answer"], result.answer)

        if answerable and result.found:
            rec["citations_ok"] = _graded(judge_citations, judge, settings.judge_model, result)
            if rec["citations_ok"] is None:
                rec["judge_failed"] = True
        records.append(rec)
        print(f"{row['id']:30} correct={rec.get('correct')} found={result.found} "
              f"${result.cost_usd:.4f} {result.latency_ms}ms")

    summary = summarize(records)
    md = to_markdown(summary, mode)
    print("\n" + md)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}-{mode}.json").write_text(json.dumps({"summary": summary, "records": records}, indent=2, default=str))
    (out / f"{stamp}-{mode}.md").write_text(md + "\n")

    accuracy = summary["answer_accuracy"] or 0.0
    if accuracy < args.min_accuracy:
        print(f"\nFAIL: answer accuracy {accuracy:.0%} is below {args.min_accuracy:.0%}")
        return 1
    return 0


def make_candidates(args: argparse.Namespace) -> None:
    """Draft numeric golden rows from XBRL. Every row starts unverified: open the filing,
    check the number, then set verified to true."""
    from .factory import build_sec

    sec = build_sec()
    rows = []
    for query in args.companies:
        co = sec.lookup(query)
        facts = sec.company_facts(co.cik)
        for metric in args.metrics:
            series = extract_series(facts, metric)[-args.years :]
            label = METRICS[metric].label.lower()
            for p in series:
                rows.append({
                    "id": f"{co.ticker.lower()}-{metric}-{p.fiscal_year}",
                    "question": f"What was {co.name}'s {label} in fiscal {p.fiscal_year}?",
                    "type": "numeric", "company": co.ticker, "fiscal_year": p.fiscal_year,
                    "expected_value": p.value, "tolerance_pct": 0.5,
                    "notes": f"XBRL {p.concept}, {p.form} filed {p.filed}, accession {p.accession}",
                    "verified": False,
                })
            if metric == "revenue" and len(series) >= 2:
                a, b = series[-2], series[-1]
                rows.append({
                    "id": f"{co.ticker.lower()}-revenue-growth-{b.fiscal_year}",
                    "question": f"By what percent did {co.name}'s revenue change from fiscal "
                                f"{a.fiscal_year} to fiscal {b.fiscal_year}?",
                    "type": "numeric", "company": co.ticker, "fiscal_year": b.fiscal_year,
                    "expected_value": round((b.value - a.value) / a.value * 100, 4),
                    "tolerance_abs": 0.1,
                    "notes": f"computed from {a.accession} and {b.accession}",
                    "verified": False,
                })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} candidate rows to {args.out}. Verify each one, then copy into golden.jsonl.")
