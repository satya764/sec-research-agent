"""Wire real dependencies together from settings."""

from __future__ import annotations

from .agent import Agent
from .config import Settings, get_settings
from .tools import ToolContext


def build_sec(settings: Settings | None = None):
    from .sec_client import SecClient

    settings = settings or get_settings()
    return SecClient(settings.sec_user_agent, cache_dir=settings.cache_dir)


def build_retriever(settings: Settings | None = None, mode: str | None = None):
    from .db import ChunkStore
    from .embeddings import OpenAIEmbedder
    from .retrieval import CohereReranker, NoopReranker, Retriever

    settings = settings or get_settings()
    store = ChunkStore(settings.database_url, settings.embedding_dim)
    embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model, settings.embedding_dim)
    reranker = (
        CohereReranker(settings.cohere_api_key, settings.rerank_model)
        if settings.cohere_api_key
        else NoopReranker()
    )
    return Retriever(store, embedder, reranker, mode=mode or settings.retrieval_mode)


def build_agent(settings: Settings | None = None, mode: str | None = None) -> Agent:
    from .llm import AnthropicLLM

    settings = settings or get_settings()
    sec = build_sec(settings)
    retriever = build_retriever(settings, mode)
    llm = AnthropicLLM(settings.anthropic_api_key, fallback_model=settings.strong_model)
    return Agent(
        llm=llm,
        make_context=lambda: ToolContext(sec=sec, retriever=retriever),
        cheap_model=settings.cheap_model,
        strong_model=settings.strong_model,
        max_steps=settings.max_agent_steps,
    )
