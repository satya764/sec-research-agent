from conftest import make_hit

from sec_research_agent.embeddings import HashEmbedder
from sec_research_agent.models import Filters
from sec_research_agent.retrieval import Retriever, reciprocal_rank_fusion


def test_rrf_rewards_agreement():
    vector = [make_hit(1), make_hit(2), make_hit(3)]
    keyword = [make_hit(3), make_hit(4), make_hit(1)]
    fused = reciprocal_rank_fusion([vector, keyword])
    assert [h.chunk_id for h in fused][:2] in ([1, 3], [3, 1])
    assert {h.chunk_id for h in fused} == {1, 2, 3, 4}


class FakeStore:
    def __init__(self):
        self.calls = []

    def vector_search(self, embedding, filters, k):
        self.calls.append(("vector", filters, k))
        return [make_hit(1), make_hit(2)]

    def keyword_search(self, query, filters, k):
        self.calls.append(("keyword", filters, k))
        return [make_hit(2), make_hit(3)]


class ReverseReranker:
    def rerank(self, query, hits, top_n):
        return list(reversed(hits))[:top_n]


def test_modes():
    store = FakeStore()
    f = Filters(ticker="AAPL")
    assert [h.chunk_id for h in Retriever(store, HashEmbedder(), mode="vector").search("q", f, k=1)] == [1]
    assert [h.chunk_id for h in Retriever(store, HashEmbedder(), mode="keyword").search("q", f)] == [2, 3]
    hybrid = Retriever(store, HashEmbedder(), mode="hybrid").search("q", f, k=3)
    assert hybrid[0].chunk_id == 2  # in both lists
    reranked = Retriever(store, HashEmbedder(), ReverseReranker(), mode="hybrid_rerank").search("q", f, k=3)
    assert [h.chunk_id for h in reranked] == list(reversed([h.chunk_id for h in hybrid]))
    assert all(call[1] == f for call in store.calls)


def test_hash_embedder_is_normalized():
    vec = HashEmbedder(dim=64).embed(["supply chain risk"])[0]
    assert abs(sum(v * v for v in vec) - 1.0) < 1e-9
