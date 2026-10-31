import time

from sec_research_agent.state import MemoryState


def test_rate_limit_window():
    s = MemoryState()
    assert [s.allow("a", 2) for _ in range(3)] == [True, True, False]
    assert s.allow("b", 2)
    s._hits["a"][0] -= 61
    assert s.allow("a", 2)


def test_cache_expiry():
    s = MemoryState()
    s.cache_put("k", {"answer": "x"})
    assert s.cache_get("k", max_age=60) == {"answer": "x"}
    s._cache["k"] = (time.time() - 120, {"answer": "x"})
    assert s.cache_get("k", max_age=60) is None


def test_history_keeps_latest():
    s = MemoryState()
    for i in range(8):
        s.add_turn("s", f"q{i}", f"a{i}")
    assert s.history("s", limit=3) == [("q5", "a5"), ("q6", "a6"), ("q7", "a7")]
    assert s.history("other") == []
