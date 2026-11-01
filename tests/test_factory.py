from sec_research_agent.config import Settings
from sec_research_agent.factory import effective_mode


def test_rerank_without_cohere_key_runs_as_hybrid():
    assert effective_mode(Settings(cohere_api_key="", retrieval_mode="hybrid_rerank")) == "hybrid"


def test_rerank_with_cohere_key_stays():
    assert effective_mode(Settings(cohere_api_key="k", retrieval_mode="hybrid_rerank")) == "hybrid_rerank"


def test_explicit_mode_wins():
    assert effective_mode(Settings(), "vector") == "vector"
