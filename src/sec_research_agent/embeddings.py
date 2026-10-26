from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Protocol

OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dim: int = 1536):
        import httpx

        if not api_key:
            raise ValueError("Set OPENAI_API_KEY to create embeddings.")
        self.dim = dim
        self._model = model
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0
        )

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [t[:8000] for t in texts[i : i + batch_size]]
            out.extend(self._embed_batch(batch))
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(4):
            resp = self._http.post(
                OPENAI_EMBEDDINGS_URL,
                json={"model": self._model, "input": batch, "dimensions": self.dim},
            )
            if resp.status_code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2**attempt)
                continue
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        raise RuntimeError("unreachable")


class HashEmbedder:
    """Deterministic bag-of-words vectors. Only for tests and offline runs: no semantic meaning."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors
