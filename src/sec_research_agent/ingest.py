"""Download filings, extract narrative sections, chunk, embed, store."""

from __future__ import annotations

import logging

from .chunking import chunk_text
from .db import ChunkStore
from .embeddings import Embedder
from .filings import html_to_text, split_sections
from .models import Chunk
from .sec_client import SecClient

log = logging.getLogger(__name__)


def ingest_company(
    query: str,
    sec: SecClient,
    store: ChunkStore,
    embedder: Embedder,
    forms: tuple[str, ...] = ("10-K",),
    limit: int = 3,
    force: bool = False,
) -> int:
    company = sec.lookup(query)
    sec.company_facts(company.cik)  # warm the facts cache
    total = 0
    for filing in sec.list_filings(company, forms, limit):
        if not force and store.has_filing(filing.accession):
            log.info("skip %s %s %s (already stored)", company.ticker, filing.form, filing.filing_date)
            continue
        text = html_to_text(sec.fetch_document(filing))
        sections = split_sections(text, filing.form)
        if not sections:
            log.warning("no sections found in %s", filing.url)
            continue
        chunks = [
            Chunk(
                ticker=company.ticker, cik=company.cik, company=company.name, form=filing.form,
                filing_date=filing.filing_date, fiscal_year=filing.fiscal_year,
                accession=filing.accession, section=section, chunk_index=i, url=filing.url,
                content=piece,
            )
            for section, body in sections.items()
            for i, piece in enumerate(chunk_text(body))
        ]
        embeddings = embedder.embed([c.embedding_text() for c in chunks])
        total += store.upsert(chunks, embeddings)
        log.info(
            "%s %s FY%s: %s chunks (%s)",
            company.ticker, filing.form, filing.fiscal_year, len(chunks), ", ".join(sections),
        )
    return total
