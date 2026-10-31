"""Postgres + pgvector storage for filing chunks: vector search and full-text keyword search."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .models import Chunk, Filters, SearchHit

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    cik TEXT NOT NULL,
    company TEXT NOT NULL,
    form TEXT NOT NULL,
    filing_date DATE NOT NULL,
    fiscal_year INT NOT NULL,
    accession TEXT NOT NULL,
    section TEXT NOT NULL,
    chunk_index INT NOT NULL,
    url TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector({dim}) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (accession, section, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_meta_idx ON chunks (ticker, fiscal_year, form, section);
"""

COLUMNS = (
    "id, ticker, cik, company, form, filing_date, fiscal_year, accession, "
    "section, chunk_index, url, content"
)


def _where(filters: Filters) -> tuple[str, list]:
    clauses, params = [], []
    for column in ("ticker", "fiscal_year", "form", "section"):
        value = getattr(filters, column)
        if value is not None:
            clauses.append(f"{column} = %s")
            params.append(value)
    return (" AND ".join(clauses) or "TRUE"), params


def _hit(row: Sequence, score: float) -> SearchHit:
    (id_, ticker, cik, company, form, filing_date, fiscal_year, accession,
     section, chunk_index, url, content) = row
    return SearchHit(
        chunk_id=id_,
        score=float(score),
        chunk=Chunk(
            ticker=ticker, cik=cik, company=company, form=form, filing_date=filing_date,
            fiscal_year=fiscal_year, accession=accession, section=section,
            chunk_index=chunk_index, url=url, content=content,
        ),
    )


class ChunkStore:
    """Thread-safe: each call borrows a connection from a pool, so API workers can search in parallel."""

    def __init__(self, url: str, dim: int, min_size: int = 1, max_size: int = 10):
        self._url = url
        self._dim = dim
        self._min_size = min_size
        self._max_size = max_size
        self._pool: ConnectionPool | None = None
        self._lock = Lock()

    @property
    def pool(self) -> ConnectionPool:
        if self._pool is None:
            with self._lock:
                if self._pool is None:
                    with psycopg.connect(self._url, autocommit=True) as conn:
                        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    self._pool = ConnectionPool(
                        self._url,
                        min_size=self._min_size,
                        max_size=self._max_size,
                        kwargs={"autocommit": True},
                        configure=register_vector,
                        open=True,
                    )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _execute(self, sql: str, params=None, fetch: str | None = None):
        with self.pool.connection() as conn:
            cur = conn.execute(sql, params)
            if fetch == "one":
                return cur.fetchone()
            if fetch == "all":
                return cur.fetchall()
            return None

    def init_schema(self) -> None:
        self._execute(SCHEMA.format(dim=self._dim))
        row = self._execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'",
            fetch="one",
        )
        if row and row[0] != self._dim:
            raise RuntimeError(
                f"chunks table stores {row[0]}-dim vectors but EMBEDDING_DIM is {self._dim}. "
                "Run: uv run sec-agent init-db --reset (deletes stored chunks)"
            )

    def drop(self) -> None:
        self._execute("DROP TABLE IF EXISTS chunks")

    def has_filing(self, accession: str) -> bool:
        row = self._execute(
            "SELECT 1 FROM chunks WHERE accession = %s LIMIT 1", (accession,), fetch="one"
        )
        return row is not None

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        rows = [
            (c.ticker, c.cik, c.company, c.form, c.filing_date, c.fiscal_year, c.accession,
             c.section, c.chunk_index, c.url, c.content, np.array(e, dtype=np.float32))
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (ticker, cik, company, form, filing_date, fiscal_year, accession,
                                    section, chunk_index, url, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (accession, section, chunk_index)
                DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
                """,
                rows,
            )
        return len(rows)

    def vector_search(self, embedding: list[float], filters: Filters, k: int) -> list[SearchHit]:
        where, params = _where(filters)
        vec = np.array(embedding, dtype=np.float32)
        rows = self._execute(
            f"SELECT {COLUMNS}, 1 - (embedding <=> %s) AS score FROM chunks "
            f"WHERE {where} ORDER BY embedding <=> %s LIMIT %s",
            [vec, *params, vec, k],
            fetch="all",
        )
        return [_hit(r[:-1], r[-1]) for r in rows]

    def keyword_search(self, query: str, filters: Filters, k: int) -> list[SearchHit]:
        where, params = _where(filters)
        rows = self._execute(
            f"SELECT {COLUMNS}, ts_rank_cd(tsv, q) AS score "
            f"FROM chunks, websearch_to_tsquery('english', %s) q "
            f"WHERE tsv @@ q AND {where} ORDER BY score DESC LIMIT %s",
            [query, *params, k],
            fetch="all",
        )
        return [_hit(r[:-1], r[-1]) for r in rows]

    def stats(self) -> list[tuple]:
        return self._execute(
            "SELECT ticker, form, fiscal_year, section, count(*) FROM chunks "
            "GROUP BY 1, 2, 3, 4 ORDER BY 1, 3 DESC, 2, 4",
            fetch="all",
        )
