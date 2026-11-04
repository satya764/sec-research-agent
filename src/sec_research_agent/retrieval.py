"""Hybrid retrieval: vector + keyword search, merged with reciprocal rank fusion, then reranked."""

from __future__ import annotations

import time
from typing import Literal, Protocol

import httpx

from .embeddings import Embedder
from .models import Filters, SearchHit

Mode = Literal["vector", "keyword", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("vector", "keyword", "hybrid", "hybrid_rerank")

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


class Store(Protocol):
    def vector_search(self, embedding: list[float], filters: Filters, k: int) -> list[SearchHit]: ...
    def keyword_search(self, query: str, filters: Filters, k: int) -> list[SearchHit]: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]: ...


class NoopReranker:
    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        return hits[:top_n]


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        self._model = model
        self._http = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=30.0)

    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        if not hits:
            return []
        body = {
            "model": self._model,
            "query": query,
            "documents": [h.chunk.embedding_text() for h in hits],
            "top_n": min(top_n, len(hits)),
        }
        for attempt in range(3):
            try:
                resp = self._http.post(COHERE_RERANK_URL, json=body)
            except httpx.TransportError:  # timeouts and dropped connections, retried like a 503
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise
            if resp.status_code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            break
        return [
            hits[r["index"]].model_copy(update={"score": float(r["relevance_score"])})
            for r in resp.json()["results"]
        ]


def reciprocal_rank_fusion(result_lists: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    scores: dict[int, float] = {}
    hits: dict[int, SearchHit] = {}
    for results in result_lists:
        for rank, hit in enumerate(results):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            hits.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [hits[cid].model_copy(update={"score": scores[cid]}) for cid in ordered]


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        reranker: Reranker | None = None,
        mode: Mode = "hybrid_rerank",
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.store = store
        self.embedder = embedder
        self.reranker = reranker or NoopReranker()
        self.mode = mode

    def search(
        self, query: str, filters: Filters | None = None, k: int = 5, candidates: int = 30
    ) -> list[SearchHit]:
        filters = filters or Filters()
        if self.mode == "keyword":
            return self.store.keyword_search(query, filters, k)
        vector_hits = self.store.vector_search(self.embedder.embed_query(query), filters, candidates)
        if self.mode == "vector":
            return vector_hits[:k]
        keyword_hits = self.store.keyword_search(query, filters, candidates)
        fused = reciprocal_rank_fusion([vector_hits, keyword_hits])
        if self.mode == "hybrid":
            return fused[:k]
        return self.reranker.rerank(query, fused[:candidates], top_n=k)
