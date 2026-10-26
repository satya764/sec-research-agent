from datetime import date

import pytest

from sec_research_agent.facts import extract_series, find_year, fiscal_year_for


def row(start, end, val, form="10-K", fp="FY", filed="2024-11-01", accn="0000320193-24-000123"):
    r = {"end": end, "val": val, "form": form, "fp": fp, "filed": filed, "accn": accn}
    if start:
        r["start"] = start
    return r


FACTS = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {"USD": [
                    row("2022-09-25", "2023-09-30", 383285000000, filed="2023-11-03"),
                    row("2023-10-01", "2024-09-28", 391035000000),
                    # comparative value repeated in the next 10-K: same period, later filing
                    row("2022-09-25", "2023-09-30", 383285000000, filed="2024-11-01"),
                    # a quarter inside a 10-K must not count as annual
                    row("2024-06-30", "2024-09-28", 94930000000),
                    row("2024-06-30", "2024-09-28", 94930000000, form="10-Q", fp="Q4"),
                ]}
            },
            "SalesRevenueNet": {
                "units": {"USD": [row("2009-09-27", "2010-09-25", 65225000000, filed="2010-10-27")]}
            },
            "Assets": {
                "units": {"USD": [
                    row(None, "2023-09-30", 352583000000),
                    row(None, "2024-09-28", 364980000000),
                ]}
            },
        }
    }
}


def test_annual_revenue_merges_concepts_and_dedupes():
    series = extract_series(FACTS, "revenue")
    assert [p.fiscal_year for p in series] == [2010, 2023, 2024]
    assert series[-1].value == 391035000000
    assert series[0].concept == "SalesRevenueNet"
    assert series[1].filed == date(2024, 11, 1)  # latest filing wins


def test_quarterly_series():
    series = extract_series(FACTS, "revenue", "quarterly")
    assert len(series) == 1 and series[0].value == 94930000000


def test_instant_metric():
    series = extract_series(FACTS, "total_assets")
    assert find_year(series, 2024).value == 364980000000
    assert find_year(series, 2019) is None


def test_unknown_metric():
    with pytest.raises(KeyError):
        extract_series(FACTS, "vibes")


def test_fiscal_year_for_52_week_years():
    assert fiscal_year_for(date(2024, 9, 28)) == 2024
    assert fiscal_year_for(date(2023, 1, 1)) == 2022
    assert fiscal_year_for(date(2025, 1, 31)) == 2025
