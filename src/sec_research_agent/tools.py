"""Tools the agent can call, plus the bookkeeping that turns tool output into citations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from . import calculator, guardrails
from .facts import METRICS, extract_series, find_year
from .models import Citation, Company, FactPoint, Filters, SearchHit

if TYPE_CHECKING:
    from .retrieval import Retriever

SECTIONS = ["business", "risk_factors", "mdna"]


class FactsSource(Protocol):
    def lookup(self, query: str) -> Company: ...
    def company_facts(self, cik: str) -> dict: ...


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_filings",
        "description": (
            "Search the narrative text of 10-K/10-Q filings (Business, Risk Factors, MD&A). "
            "Use for descriptive questions. Returns passages with source_ids."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in plain words"},
                "company": {"type": "string", "description": "Ticker or company name"},
                "fiscal_year": {"type": "integer"},
                "section": {"type": "string", "enum": SECTIONS},
                "form": {"type": "string", "enum": ["10-K", "10-Q"]},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_financial_data",
        "description": (
            "Reported values from the company's XBRL financial data. The only allowed source "
            "for numbers. Values are in base units (USD, not millions). Returns source_ids."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Ticker or company name"},
                "metric": {"type": "string", "enum": list(METRICS)},
                "fiscal_year": {
                    "type": "integer",
                    "description": (
                        "Only if the question names a year. For latest / most recent / current, "
                        "omit it: the result lists the newest years first. Never guess the year."
                    ),
                },
                "period": {"type": "string", "enum": ["annual", "quarterly"]},
            },
            "required": ["company", "metric"],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate arithmetic such as (391035000000 - 383285000000) / 383285000000 * 100. "
            "Plain numbers only, no commas or units."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "final_answer",
        "description": "Give the final answer. Call exactly once, as the last step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "1-4 sentences with units and years"},
                "found": {
                    "type": "boolean",
                    "description": "false if the filings do not contain the answer",
                },
                "citations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "source_ids of every source the answer relies on",
                },
                "value": {
                    "type": "number",
                    "description": (
                        "The number that directly answers the question. For a percent or growth "
                        "question, the percent itself (e.g. 6.43, not the revenue). For an amount, "
                        "base units exactly as returned by get_financial_data. Omit if none."
                    ),
                },
                "unit": {"type": "string", "description": "percent, USD, USD/shares, ..."},
            },
            "required": ["answer", "found", "citations"],
        },
    },
]


def _fmt(value: float, unit: str) -> str:
    if unit == "USD" and abs(value) >= 1e6:
        return f"${value:,.0f}"
    return f"{value:,.2f} {unit}"


@dataclass
class ToolContext:
    sec: FactsSource
    retriever: Retriever | None
    sources: dict[str, Citation] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[SearchHit] = field(default_factory=list)
    _fact_counter: int = 0

    def run(self, name: str, args: dict[str, Any]) -> str:
        method = getattr(self, f"_tool_{name}", None)
        if method is None:
            result: dict[str, Any] = {"error": f"unknown tool {name}"}
        else:
            try:
                result = method(**args)
            except Exception as err:  # tool errors go back to the model so it can recover
                result = {"error": f"{type(err).__name__}: {err}"}
        self.calls.append({"tool": name, "input": args, "ok": "error" not in result})
        return json.dumps(result, default=str)

    # ---- tools ------------------------------------------------------------

    def _tool_search_filings(self, query, company=None, fiscal_year=None, section=None, form=None):
        if self.retriever is None:
            return {"error": "filing search is not configured"}
        ticker = self.sec.lookup(company).ticker if company else None
        filters = Filters(ticker=ticker, fiscal_year=fiscal_year, section=section, form=form)
        hits = self.retriever.search(query, filters, k=5)
        note = None
        if not hits and fiscal_year is not None:
            hits = self.retriever.search(query, filters.model_copy(update={"fiscal_year": None}), k=5)
            note = f"No passages for fiscal {fiscal_year}; showing other years."
        self.retrieved.extend(hits)
        passages = []
        for hit in hits:
            c = hit.chunk
            text = guardrails.clean_retrieved(c.content)
            self.sources[hit.source_id] = Citation(
                source_id=hit.source_id, kind="filing", company=c.company, form=c.form,
                filing_date=c.filing_date, section=c.section, url=c.url, snippet=text,
            )
            passages.append({
                "source_id": hit.source_id, "company": c.company, "form": c.form,
                "fiscal_year": c.fiscal_year, "section": c.section, "text": text,
            })
        return {"passages": passages, "note": note} if note else {"passages": passages}

    def _tool_get_financial_data(self, company, metric, fiscal_year=None, period="annual"):
        if metric not in METRICS:
            return {
                "error": f"unknown metric '{metric}'. Use one of: {', '.join(METRICS)}. "
                "For a margin or ratio, get each metric (e.g. gross_profit and revenue) "
                "and divide with calculator."
            }
        note = None
        co = self.sec.lookup(company)
        series = extract_series(self.sec.company_facts(co.cik), metric, period)
        if not series:
            return {"error": f"{co.name} does not report {metric} in XBRL"}
        if fiscal_year is not None and period == "annual":
            point = find_year(series, fiscal_year)
            if point is None:
                years = sorted({p.fiscal_year for p in series})
                return {"error": f"no {metric} for fiscal {fiscal_year}; available: {years}"}
            points = [point]
            if fiscal_year < series[-1].fiscal_year:
                note = f"Fiscal {fiscal_year} is not the latest year; the latest is {series[-1].fiscal_year}."
        elif fiscal_year is not None:
            # Quarters are matched by the calendar year they end in, which can differ from the company's
            # fiscal year (Microsoft's fiscal Q1 ends in September of the year before). Say so.
            points = list(reversed([p for p in series if p.end.year == fiscal_year]))
            if not points:
                years = sorted({p.end.year for p in series})
                return {"error": f"no quarterly {metric} ending in {fiscal_year}; available: {years}"}
            note = f"Quarters ending in calendar {fiscal_year}. Check period_end for the exact quarter."
        else:
            points = list(reversed(series[-5:]))  # newest first: small models read the top entry
        result = {
            "company": co.name,
            "metric": METRICS[metric].label,
            "latest_fiscal_year": series[-1].fiscal_year,
            "values": [self._register_fact(co, p) for p in points],
        }
        if note:
            result["note"] = note
        return result

    def _register_fact(self, co: Company, p: FactPoint) -> dict[str, Any]:
        self._fact_counter += 1
        source_id = f"F{self._fact_counter}"
        accn = p.accession.replace("-", "")
        label = METRICS[p.metric].label
        snippet = f"{label} for period ending {p.end}: {_fmt(p.value, p.unit)} ({p.concept}, {p.form})"
        self.sources[source_id] = Citation(
            source_id=source_id, kind="fact", company=co.name, form=p.form, filing_date=p.filed,
            section=None, url=f"https://www.sec.gov/Archives/edgar/data/{co.cik_int}/{accn}/",
            snippet=snippet,
        )
        return {
            "source_id": source_id, "fiscal_year": p.fiscal_year, "period_end": str(p.end),
            "value": p.value, "unit": p.unit, "form": p.form, "filed": str(p.filed),
        }

    def _tool_calculator(self, expression):
        return {"expression": expression, "result": calculator.evaluate(expression)}
