import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sec_research_agent import api  # noqa: E402
from sec_research_agent.models import AskResult  # noqa: E402
from sec_research_agent.state import MemoryState  # noqa: E402


class FakeAgent:
    def __init__(self):
        self.calls = []

    def ask(self, question, history=None):
        self.calls.append((question, history))
        return AskResult(question=question, answer=f"answer {len(self.calls)}", found=True)


@pytest.fixture
def client():
    agent, state = FakeAgent(), MemoryState()
    api.app.dependency_overrides[api.get_agent] = lambda: agent
    api.app.dependency_overrides[api.get_state] = lambda: state
    yield TestClient(api.app), agent
    api.app.dependency_overrides.clear()
    api.settings.api_key = ""
    api.settings.rate_limit_per_minute = 20


def test_health_and_request_id(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.json() == {"status": "ok"}
    assert resp.headers["X-Request-ID"]


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
    api.settings.rate_limit_per_minute = 2
    codes = [c.post("/ask", json={"question": f"question {i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_api_key(client):
    c, _ = client
    api.settings.api_key = "secret"
    assert c.post("/ask", json={"question": "Apple revenue?"}).status_code == 401
    assert c.post("/ask", json={"question": "Apple revenue?"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/ask", json={"question": "Apple revenue?"}, headers={"X-API-Key": "secret"}).status_code == 200
