"""Pull clean annual or quarterly series out of an SEC companyfacts JSON document.

Numbers come from XBRL tags, not from text. Tables in filings get split by chunkers,
so the agent never reads numbers from chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from .models import FactPoint

Period = Literal["annual", "quarterly"]


@dataclass(frozen=True)
class MetricSpec:
    label: str
    concepts: tuple[str, ...]  # tried in order; companies switch tags over the years
    unit: str = "USD"
    kind: Literal["duration", "instant"] = "duration"


METRICS: dict[str, MetricSpec] = {
    "revenue": MetricSpec(
        "Revenue",
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ),
    ),
    "cost_of_revenue": MetricSpec(
        "Cost of revenue", ("CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold")
    ),
    "gross_profit": MetricSpec("Gross profit", ("GrossProfit",)),
    "research_and_development": MetricSpec(
        "Research and development expense", ("ResearchAndDevelopmentExpense",)
    ),
    "operating_income": MetricSpec("Operating income", ("OperatingIncomeLoss",)),
    "net_income": MetricSpec("Net income", ("NetIncomeLoss", "ProfitLoss")),
    "eps_diluted": MetricSpec("Diluted EPS", ("EarningsPerShareDiluted",), unit="USD/shares"),
    "operating_cash_flow": MetricSpec(
        "Operating cash flow", ("NetCashProvidedByUsedInOperatingActivities",)
    ),
    "capital_expenditures": MetricSpec(
        "Capital expenditures", ("PaymentsToAcquirePropertyPlantAndEquipment",)
    ),
    "total_assets": MetricSpec("Total assets", ("Assets",), kind="instant"),
    "total_liabilities": MetricSpec("Total liabilities", ("Liabilities",), kind="instant"),
    "stockholders_equity": MetricSpec(
        "Stockholders' equity",
        ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        kind="instant",
    ),
    "cash": MetricSpec(
        "Cash and cash equivalents",
        ("CashAndCashEquivalentsAtCarryingValue",),
        kind="instant",
    ),
    "long_term_debt": MetricSpec(
        "Long-term debt", ("LongTermDebtNoncurrent", "LongTermDebt"), kind="instant"
    ),
}

ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "40-F"}
QUARTERLY_FORMS = {"10-Q", "10-Q/A"}


def _duration_ok(start: date, end: date, period: Period) -> bool:
    days = (end - start).days
    return 350 <= days <= 380 if period == "annual" else 80 <= days <= 100


def fiscal_year_for(end: date) -> int:
    # 52/53-week years that end in the first days of January belong to the prior year.
    return end.year - 1 if end.month == 1 and end.day <= 7 else end.year


def extract_series(company_facts: dict, metric: str, period: Period = "annual") -> list[FactPoint]:
    """Return one value per period end, oldest first. Restated values win (latest filing)."""
    if metric not in METRICS:
        raise KeyError(f"Unknown metric '{metric}'. Known: {', '.join(METRICS)}")
    spec = METRICS[metric]
    gaap = company_facts.get("facts", {}).get("us-gaap", {})
    forms = ANNUAL_FORMS if period == "annual" else QUARTERLY_FORMS

    by_end: dict[date, tuple[int, FactPoint]] = {}
    for priority, concept in enumerate(spec.concepts):
        for entry in gaap.get(concept, {}).get("units", {}).get(spec.unit, []):
            if entry.get("form") not in forms:
                continue
            end = date.fromisoformat(entry["end"])
            start = date.fromisoformat(entry["start"]) if entry.get("start") else None
            if spec.kind == "duration":
                if start is None or not _duration_ok(start, end, period):
                    continue
            elif start is not None:
                continue
            point = FactPoint(
                metric=metric,
                concept=concept,
                unit=spec.unit,
                value=float(entry["val"]),
                start=start,
                end=end,
                fiscal_year=fiscal_year_for(end),
                period=period,
                form=entry["form"],
                filed=date.fromisoformat(entry["filed"]),
                accession=entry["accn"],
            )
            current = by_end.get(end)
            if (
                current is None
                or priority < current[0]
                or (priority == current[0] and point.filed > current[1].filed)
            ):
                by_end[end] = (priority, point)

    return [p for _, (_, p) in sorted(by_end.items())]


def find_year(series: list[FactPoint], fiscal_year: int) -> FactPoint | None:
    """Annual series: the point whose period ends in that fiscal year."""
    matches = [p for p in series if p.fiscal_year == fiscal_year]
    return matches[-1] if matches else None
