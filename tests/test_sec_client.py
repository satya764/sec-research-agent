import json
import threading
import time
from itertools import pairwise

import pytest

httpx = pytest.importorskip("httpx")

from sec_research_agent.sec_client import CompanyNotFound, SecClient  # noqa: E402

TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1418121, "ticker": "APLE", "title": "Apple Hospitality REIT, Inc."},
    "2": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
}
SUBMISSIONS = {"filings": {"recent": {
    "form": ["8-K", "10-K", "10-Q", "10-K"],
    "accessionNumber": ["a-1", "0000320193-24-000123", "a-3", "0000320193-23-000106"],
    "filingDate": ["2024-12-01", "2024-11-01", "2024-08-02", "2023-11-03"],
    "reportDate": ["", "2024-09-28", "2024-06-29", "2023-09-30"],
    "primaryDocument": ["x.htm", "aapl-20240928.htm", "y.htm", "aapl-20230930.htm"],
}}}


def make_client():
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        if "submissions" in request.url.path:
            return httpx.Response(200, json=SUBMISSIONS)
        return httpx.Response(404)

    return SecClient("Test Person test@example.com", transport=httpx.MockTransport(handler), min_interval=0), seen


def test_requires_user_agent():
    with pytest.raises(ValueError):
        SecClient("")


def test_lookup_by_ticker_name_and_cik():
    sec, seen = make_client()
    assert sec.lookup("aapl").name == "Apple Inc."
    assert sec.lookup("Apple").ticker == "AAPL"
    assert sec.lookup("microsoft").ticker == "MSFT"
    assert sec.lookup("320193").ticker == "AAPL"
    with pytest.raises(CompanyNotFound):
        sec.lookup("not a company")
    assert seen[0].headers["User-Agent"] == "Test Person test@example.com"
    assert len(seen) == 1  # ticker map fetched once


def test_list_filings():
    sec, _ = make_client()
    filings = sec.list_filings(sec.lookup("AAPL"), ("10-K",), limit=5)
    assert [f.fiscal_year for f in filings] == [2024, 2023]
    assert filings[0].url == "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"


def test_cache_writes_whole_files(tmp_path):
    sec = SecClient("Test Person test@example.com", cache_dir=tmp_path, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=TICKERS)), min_interval=0)
    sec.tickers()
    assert json.loads((tmp_path / "company_tickers.json").read_text()) == TICKERS
    assert not list(tmp_path.glob("*.tmp"))  # no temp files left behind


def test_requests_spaced_across_threads():
    times = []

    def handler(request):
        times.append(time.monotonic())
        return httpx.Response(200, json=TICKERS)

    sec = SecClient("Test Person test@example.com", transport=httpx.MockTransport(handler), min_interval=0.05)
    threads = [threading.Thread(target=sec._get, args=("https://www.sec.gov/x",)) for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    gaps = [b - a for a, b in pairwise(sorted(times))]
    assert min(gaps) >= 0.03  # without the lock, parallel requests land almost together
