import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sec_research_agent import api  # noqa: E402
from sec_research_agent.models import AskResult  # noqa: E402


class FakeAgent:
    def __init__(self):
        self.calls = []

    def ask(self, question, history=None):
        self.calls.append((question, history))
        return AskResult(question=question, answer=f"answer {len(self.calls)}", found=True)


@pytest.fixture
def client():
    agent = FakeAgent()
    api.app.dependency_overrides[api.get_agent] = lambda: agent
    api.cache._items.clear()
    api.sessions.clear()
    api.limiter._hits.clear()
    yield TestClient(api.app), agent
    api.app.dependency_overrides.clear()


def test_health(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok"}


def test_ask_and_cache(client):
    c, agent = client
    first = c.post("/ask", json={"question": "Apple revenue 2024?"}).json()
    second = c.post("/ask", json={"question": "apple  revenue 2024?"}).json()
    assert first["answer"] == second["answer"] == "answer 1"
    assert len(agent.calls) == 1


def test_session_memory(client):
    c, agent = client
    c.post("/ask", json={"question": "Apple revenue 2024?", "session_id": "s1"})
    c.post("/ask", json={"question": "And 2023?", "session_id": "s1"})
    assert agent.calls[1][1] == [("Apple revenue 2024?", "answer 1")]


def test_validation_and_rate_limit(client):
    c, _ = client
    assert c.post("/ask", json={"question": "x"}).status_code == 422
    api.limiter.per_minute = 2
    try:
        codes = [c.post("/ask", json={"question": f"question {i}"}).status_code for i in range(3)]
    finally:
        api.limiter.per_minute = api.settings.rate_limit_per_minute
    assert codes == [200, 200, 429]
