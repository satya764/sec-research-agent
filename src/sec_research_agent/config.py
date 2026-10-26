"""Settings come from environment variables, with an optional .env file in the project root."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

# USD per million tokens (input, output) for prompts under 100K tokens.
# Check https://www.anthropic.com/pricing before trusting cost numbers.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines into os.environ without overriding variables already set."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Settings(BaseModel):
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    cohere_api_key: str = ""
    sec_user_agent: str = ""
    database_url: str = "postgresql://postgres:postgres@localhost:5432/sec_agent"

    # anthropic = Claude (paid). openai_compatible = any OpenAI-style endpoint:
    # Ollama on your machine (free, the default base URL), Groq, OpenRouter, ...
    llm_provider: str = "anthropic"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"

    cheap_model: str = "claude-haiku-5-5"
    strong_model: str = "claude-sonnet-5-5"
    judge_model: str = "claude-haiku-5-5"

    # local = free, runs on your machine. openai = needs OPENAI_API_KEY
    # (then also set EMBEDDING_MODEL=text-embedding-3-small and EMBEDDING_DIM=1536).
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    rerank_model: str = "rerank-v3.5"
    retrieval_mode: str = "hybrid_rerank"

    cache_dir: Path = Path("data/cache")
    max_agent_steps: int = 8
    rate_limit_per_minute: int = 20
    answer_cache_seconds: int = 3600

    @classmethod
    def from_env(cls) -> Settings:
        values = {
            name: os.environ[name.upper()]
            for name in cls.model_fields
            if os.environ.get(name.upper())
        }
        return cls.model_validate(values)


@lru_cache
def get_settings() -> Settings:
    load_dotenv()
    return Settings.from_env()
