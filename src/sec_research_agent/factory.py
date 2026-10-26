"""Wire real dependencies together from settings."""

from __future__ import annotations

from .agent import Agent
from .config import Settings, get_settings
from .tools import ToolContext


def build_sec(settings: Settings | None = None):
    from .sec_client import SecClient

    settings = settings or get_settings()
    return SecClient(settings.sec_user_agent, cache_dir=settings.cache_dir)


def build_embedder(settings: Settings | None = None):
    from .embeddings import LocalEmbedder, OpenAIEmbedder

    settings = settings or get_settings()
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings.openai_api_key, settings.embedding_model, settings.embedding_dim)
    if settings.embedding_provider == "local":
        return LocalEmbedder(
            settings.embedding_model, settings.embedding_dim, str(settings.cache_dir.parent / "models")
        )
    raise ValueError("EMBEDDING_PROVIDER must be 'local' or 'openai'")


def build_retriever(settings: Settings | None = None, mode: str | None = None):
    from .db import ChunkStore
    from .retrieval import CohereReranker, NoopReranker, Retriever

    settings = settings or get_settings()
    store = ChunkStore(settings.database_url, settings.embedding_dim)
    embedder = build_embedder(settings)
    reranker = (
        CohereReranker(settings.cohere_api_key, settings.rerank_model)
        if settings.cohere_api_key
        else NoopReranker()
    )
    return Retriever(store, embedder, reranker, mode=mode or settings.retrieval_mode)


def build_llm(settings: Settings | None = None):
    from .llm import AnthropicLLM, OpenAICompatibleLLM

    settings = settings or get_settings()
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.anthropic_api_key, fallback_model=settings.strong_model)
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleLLM(settings.llm_base_url, settings.llm_api_key)
    raise ValueError("LLM_PROVIDER must be 'anthropic' or 'openai_compatible'")


def build_agent(settings: Settings | None = None, mode: str | None = None) -> Agent:
    settings = settings or get_settings()
    sec = build_sec(settings)
    retriever = build_retriever(settings, mode)
    llm = build_llm(settings)
    return Agent(
        llm=llm,
        make_context=lambda: ToolContext(sec=sec, retriever=retriever),
        cheap_model=settings.cheap_model,
        strong_model=settings.strong_model,
        max_steps=settings.max_agent_steps,
    )
