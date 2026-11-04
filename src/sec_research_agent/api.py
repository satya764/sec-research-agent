"""HTTP API. Run: uv run sec-agent serve  (docs at http://localhost:8000/docs)

Stateless workers: rate limits, cached answers and chat history live in the state store
(Postgres in production), so any number of API instances can run behind a load balancer.
"""

from __future__ import annotations

import hmac
import json
import logging
import time
import uuid
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import Agent
from .config import get_settings
from .models import AskResult, FactPoint
from .state import StateStore, build_state
from .tracing import setup_tracing

log = logging.getLogger("sec_research_agent.api")
settings = get_settings()
setup_tracing()

app = FastAPI(title="SEC Filings Research Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    # At least 16 random characters (the UI sends a random hex id), so one user can't guess another's id
    # and read into their conversation.
    session_id: str | None = Field(None, min_length=16, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


@app.middleware("http")
async def request_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    log.info(json.dumps({
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "ms": round((time.perf_counter() - started) * 1000),
    }))
    return response


@lru_cache
def get_state() -> StateStore:
    return build_state(settings.state_backend, settings.database_url)


@lru_cache
def get_agent() -> Agent:
    from .factory import build_agent

    return build_agent()


@lru_cache
def get_sec():
    from .factory import build_sec

    return build_sec()


def require_key(x_api_key: str | None = Header(None)) -> None:
    if settings.api_key and not hmac.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResult, dependencies=[Depends(require_key)])
def ask(
    body: AskRequest,
    request: Request,
    agent: Agent = Depends(get_agent),
    state: StateStore = Depends(get_state),
) -> AskResult:
    client = request.client.host if request.client else "unknown"
    if not state.allow(client, settings.rate_limit_per_minute):
        raise HTTPException(status_code=429, detail="Too many requests, try again in a minute.")

    history = state.history(body.session_id) if body.session_id else []
    key = " ".join(body.question.lower().split())
    if not history and (hit := state.cache_get(key, settings.answer_cache_seconds)):
        return AskResult.model_validate(hit)

    result = agent.ask(body.question, history=history)
    if not history and result.found and not result.blocked:
        state.cache_put(key, result.model_dump(mode="json"))
    if body.session_id:
        state.add_turn(body.session_id, body.question, result.answer)
    return result


@app.get("/facts/{company}/{metric}", response_model=list[FactPoint], dependencies=[Depends(require_key)])
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
