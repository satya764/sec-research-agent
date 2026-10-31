"""Shared API state: rate limits, answer cache, chat history.

memory   = one process only (tests, local dev)
postgres = shared by every API instance, so the API can scale horizontally
"""

from __future__ import annotations

import json
import time
from collections import deque
from threading import Lock
from typing import Protocol

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_rate (
    client TEXT NOT NULL,
    ts DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS api_rate_client_ts ON api_rate (client, ts);
CREATE TABLE IF NOT EXISTS api_cache (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    created DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS api_turns (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS api_turns_session ON api_turns (session_id, id);
"""


class StateStore(Protocol):
    def allow(self, client: str, per_minute: int) -> bool: ...
    def cache_get(self, key: str, max_age: float) -> dict | None: ...
    def cache_put(self, key: str, value: dict) -> None: ...
    def history(self, session_id: str, limit: int = 5) -> list[tuple[str, str]]: ...
    def add_turn(self, session_id: str, question: str, answer: str) -> None: ...


class MemoryState:
    def __init__(self, max_cache: int = 500):
        self._lock = Lock()
        self._hits: dict[str, deque[float]] = {}
        self._cache: dict[str, tuple[float, dict]] = {}
        self._turns: dict[str, deque[tuple[str, str]]] = {}
        self._max_cache = max_cache

    def allow(self, client: str, per_minute: int) -> bool:
        now = time.time()
        with self._lock:
            hits = self._hits.setdefault(client, deque())
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= per_minute:
                return False
            hits.append(now)
            return True

    def cache_get(self, key: str, max_age: float) -> dict | None:
        with self._lock:
            item = self._cache.get(key)
            if item and time.time() - item[0] < max_age:
                return item[1]
            self._cache.pop(key, None)
            return None

    def cache_put(self, key: str, value: dict) -> None:
        with self._lock:
            if len(self._cache) >= self._max_cache:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (time.time(), value)

    def history(self, session_id: str, limit: int = 5) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._turns.get(session_id, []))[-limit:]

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            self._turns.setdefault(session_id, deque(maxlen=20)).append((question, answer))


class PostgresState:
    def __init__(self, url: str, max_size: int = 5):
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(url, min_size=1, max_size=max_size, kwargs={"autocommit": True}, open=True)
        with self._pool.connection() as conn:
            conn.execute(POSTGRES_SCHEMA)

    def allow(self, client: str, per_minute: int) -> bool:
        now = time.time()
        with self._pool.connection() as conn, conn.transaction():
            # serialize per client so two instances can't both let request N+1 through
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (client,))
            conn.execute("DELETE FROM api_rate WHERE client = %s AND ts < %s", (client, now - 60))
            (count,) = conn.execute("SELECT count(*) FROM api_rate WHERE client = %s", (client,)).fetchone()
            if count >= per_minute:
                return False
            conn.execute("INSERT INTO api_rate (client, ts) VALUES (%s, %s)", (client, now))
            return True

    def cache_get(self, key: str, max_age: float) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT value FROM api_cache WHERE key = %s AND created > %s", (key, time.time() - max_age)
            ).fetchone()
        return row[0] if row else None

    def cache_put(self, key: str, value: dict) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO api_cache (key, value, created) VALUES (%s, %s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, created = EXCLUDED.created",
                (key, json.dumps(value, default=str), time.time()),
            )

    def history(self, session_id: str, limit: int = 5) -> list[tuple[str, str]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT question, answer FROM api_turns WHERE session_id = %s ORDER BY id DESC LIMIT %s",
                (session_id, limit),
            ).fetchall()
        return [(q, a) for q, a in reversed(rows)]

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO api_turns (session_id, question, answer, created) VALUES (%s, %s, %s, %s)",
                (session_id, question, answer, time.time()),
            )


def build_state(backend: str, database_url: str) -> StateStore:
    if backend == "postgres":
        return PostgresState(database_url)
    if backend == "memory":
        return MemoryState()
    raise ValueError("STATE_BACKEND must be 'memory' or 'postgres'")
