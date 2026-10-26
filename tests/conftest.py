from datetime import date

import pytest

from sec_research_agent.models import Chunk, Company, SearchHit


def make_hit(chunk_id: int, section="risk_factors", content="text", score=1.0, ticker="AAPL") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        score=score,
        chunk=Chunk(
            ticker=ticker, cik="0000320193", company="Apple Inc.", form="10-K",
            filing_date=date(2024, 11, 1), fiscal_year=2024, accession="0000320193-24-000123",
            section=section, chunk_index=chunk_id, url="https://www.sec.gov/x.htm", content=content,
        ),
    )


class FakeSec:
    def __init__(self, facts: dict):
        self.facts = facts

    def lookup(self, query: str) -> Company:
        return Company(cik="0000320193", ticker="AAPL", name="Apple Inc.")

    def company_facts(self, cik: str) -> dict:
        return self.facts


@pytest.fixture
def apple_facts() -> dict:
    def row(start, end, val, filed):
        return {"start": start, "end": end, "val": val, "form": "10-K", "fp": "FY",
                "filed": filed, "accn": "0000320193-24-000123"}

    return {"facts": {"us-gaap": {"RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        row("2022-09-25", "2023-09-30", 383285000000, "2023-11-03"),
        row("2023-10-01", "2024-09-28", 391035000000, "2024-11-01"),
    ]}}}}}
