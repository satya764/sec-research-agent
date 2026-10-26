"""HTTP API. Run: uv run sec-agent serve  (docs at http://localhost:8000/docs)"""

from __future__ import annotations

import time
from collections import deque
from functools import lru_cache
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .agent import Agent
from .config import get_settings
from .models import AskResult, FactPoint
from .tracing import setup_tracing

app = FastAPI(title="SEC Filings Research Agent", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    session_id: str | None = Field(None, max_length=64)


class TTLCache:
    def __init__(self, seconds: int, max_items: int = 500):
        self.seconds, self.max_items = seconds, max_items
        self._items: dict[str, tuple[float, AskResult]] = {}

    def get(self, key: str) -> AskResult | None:
        item = self._items.get(key)
        if item and time.time() - item[0] < self.seconds:
            return item[1]
        self._items.pop(key, None)
        return None

    def put(self, key: str, value: AskResult) -> None:
        if len(self._items) >= self.max_items:
            self._items.pop(next(iter(self._items)))
        self._items[key] = (time.time(), value)


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, client: str) -> bool:
        now = time.time()
        with self._lock:
            hits = self._hits.setdefault(client, deque())
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            return True


settings = get_settings()
setup_tracing()
cache = TTLCache(settings.answer_cache_seconds)
limiter = RateLimiter(settings.rate_limit_per_minute)
sessions: dict[str, deque[tuple[str, str]]] = {}


@lru_cache
def get_agent() -> Agent:
    from .factory import build_agent

    return build_agent()


@lru_cache
def get_sec():
    from .factory import build_sec

    return build_sec()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResult)
def ask(body: AskRequest, request: Request, agent: Agent = Depends(get_agent)) -> AskResult:
    client = request.client.host if request.client else "unknown"
    if not limiter.allow(client):
        raise HTTPException(status_code=429, detail="Too many requests, try again in a minute.")

    history = list(sessions.get(body.session_id, [])) if body.session_id else []
    key = " ".join(body.question.lower().split())
    if not history and (hit := cache.get(key)):
        return hit

    result = agent.ask(body.question, history=history)
    if not history and result.found and not result.blocked:
        cache.put(key, result)
    if body.session_id:
        sessions.setdefault(body.session_id, deque(maxlen=5)).append((body.question, result.answer))
    return result


@app.get("/facts/{company}/{metric}", response_model=list[FactPoint])
def facts(company: str, metric: str, sec=Depends(get_sec)) -> list[FactPoint]:
    from .facts import METRICS, extract_series
    from .sec_client import CompanyNotFound

    if metric not in METRICS:
        raise HTTPException(status_code=404, detail=f"Unknown metric. Known: {', '.join(METRICS)}")
    try:
        co = sec.lookup(company)
    except CompanyNotFound as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return extract_series(sec.company_facts(co.cik), metric)
