"""Small client for SEC EDGAR: ticker lookup, company facts (XBRL), filing lists and documents.

SEC rules: send a User-Agent with your name and email, stay under 10 requests per second.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path

import httpx

from .models import Company, FilingRef

log = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

DAY = 24 * 3600


class CompanyNotFound(LookupError):
    pass


class SecClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: Path | None = None,
        transport: httpx.BaseTransport | None = None,
        min_interval: float = 0.12,
    ) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError("Set SEC_USER_AGENT to 'Your Name your@email.com' (SEC requires it).")
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )
        self._cache_dir = cache_dir
        self._min_interval = min_interval
        self._last_request = 0.0
        self._tickers: dict[str, Company] | None = None

    # ---- HTTP -------------------------------------------------------------

    def _get(self, url: str) -> httpx.Response:
        for attempt in range(4):
            wait = self._min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            resp = self._http.get(url)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                backoff = 2**attempt
                log.warning("SEC returned %s for %s, retrying in %ss", resp.status_code, url, backoff)
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError("unreachable")

    def _cache_path(self, name: str) -> Path | None:
        if self._cache_dir is None:
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir / name

    def _cached(self, name: str, max_age: float) -> str | None:
        path = self._cache_path(name)
        if path and path.exists() and time.time() - path.stat().st_mtime < max_age:
            return path.read_text()
        return None

    def _store(self, name: str, text: str) -> None:
        path = self._cache_path(name)
        if path:
            path.write_text(text)

    def _get_json(self, url: str, cache_name: str, max_age: float = DAY) -> dict:
        text = self._cached(cache_name, max_age)
        if text is None:
            text = self._get(url).text
            self._store(cache_name, text)
        return json.loads(text)

    # ---- Companies --------------------------------------------------------

    def tickers(self) -> dict[str, Company]:
        if self._tickers is None:
            raw = self._get_json(TICKERS_URL, "company_tickers.json", max_age=7 * DAY)
            self._tickers = {
                row["ticker"].upper(): Company(
                    cik=str(row["cik_str"]).zfill(10), ticker=row["ticker"].upper(), name=row["title"]
                )
                for row in raw.values()
            }
        return self._tickers

    def lookup(self, query: str) -> Company:
        """Find a company by ticker, CIK, or name."""
        q = query.strip()
        tickers = self.tickers()
        if q.upper() in tickers:
            return tickers[q.upper()]
        if q.isdigit():
            cik = q.zfill(10)
            for company in tickers.values():
                if company.cik == cik:
                    return company
        name = q.lower()
        matches = [c for c in tickers.values() if c.name.lower().startswith(name)]
        matches = matches or [c for c in tickers.values() if name in c.name.lower()]
        if not matches:
            raise CompanyNotFound(f"No SEC registrant matches '{query}'")
        return min(matches, key=lambda c: len(c.name))

    # ---- Data -------------------------------------------------------------

    def company_facts(self, cik: str) -> dict:
        return self._get_json(FACTS_URL.format(cik=cik), f"facts_{cik}.json")

    def submissions(self, cik: str) -> dict:
        return self._get_json(SUBMISSIONS_URL.format(cik=cik), f"submissions_{cik}.json")

    def list_filings(
        self, company: Company, forms: tuple[str, ...] = ("10-K",), limit: int = 3
    ) -> list[FilingRef]:
        recent = self.submissions(company.cik)["filings"]["recent"]
        filings: list[FilingRef] = []
        for i, form in enumerate(recent["form"]):
            if form not in forms:
                continue
            report = recent["reportDate"][i]
            filings.append(
                FilingRef(
                    cik=company.cik,
                    ticker=company.ticker,
                    company=company.name,
                    form=form,
                    accession=recent["accessionNumber"][i],
                    filing_date=date.fromisoformat(recent["filingDate"][i]),
                    report_date=date.fromisoformat(report) if report else None,
                    primary_document=recent["primaryDocument"][i],
                )
            )
            if len(filings) >= limit:
                break
        return filings

    def fetch_document(self, filing: FilingRef) -> str:
        name = f"doc_{filing.accession}.html"
        text = self._cached(name, max_age=365 * DAY)
        if text is None:
            text = self._get(filing.url).text
            self._store(name, text)
        return text

    def close(self) -> None:
        self._http.close()
