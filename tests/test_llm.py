from sec_research_agent.llm import cost_usd, from_openai_message, to_openai_messages


def test_round_trip_to_openai_format():
    messages = [
        {"role": "user", "content": "Apple revenue?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "t1", "name": "calculator", "input": {"expression": "1+1"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{\"result\": 2}"}]},
    ]
    out = to_openai_messages(messages)
    assert out[0] == {"role": "user", "content": "Apple revenue?"}
    assert out[1]["tool_calls"][0]["function"] == {"name": "calculator", "arguments": "{\"expression\": \"1+1\"}"}
    assert out[2] == {"role": "tool", "tool_call_id": "t1", "content": "{\"result\": 2}"}


def test_parses_tool_calls_back():
    blocks = from_openai_message({
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "calculator", "arguments": "{\"expression\": \"2*3\"}"}}],
    })
    assert blocks == [{"type": "tool_use", "id": "c1", "name": "calculator", "input": {"expression": "2*3"}}]


def test_json_text_becomes_forced_tool_call():
    blocks = from_openai_message({"content": "Sure: {\"companies\": [\"AAPL\"]}"}, forced_tool="parsed_question")
    assert blocks[0]["type"] == "tool_use" and blocks[0]["input"] == {"companies": ["AAPL"]}


def test_local_models_cost_nothing():
    assert cost_usd("qwen3:8b", 10_000, 1_000) == 0.0
    assert cost_usd("claude-haiku-5-5", 1_000_000, 0) == 0.10


def test_structured_retries_then_raises():
    import pytest
    from pydantic import BaseModel

    from sec_research_agent.llm import ChatResponse, StructuredOutputError, structured

    class V(BaseModel):
        correct: bool

    class Flaky:
        def __init__(self, replies):
            self.replies, self.prompts = list(replies), []

        def chat(self, *, model, system, messages, tools=None, tool_choice=None, max_tokens=1024):
            self.prompts.append(messages[0]["content"])
            return ChatResponse(content=self.replies.pop(0), stop_reason="", model=model)

    ok = Flaky([[{"type": "text", "text": "hmm"}], [{"type": "tool_use", "id": "a", "name": "v", "input": {"correct": True}}]])
    verdict, _ = structured(ok, model="m", system="s", prompt="p", schema=V, name="v", description="d")
    assert verdict.correct and "JSON object" in ok.prompts[1]

    bad = Flaky([[{"type": "text", "text": "no"}], [{"type": "text", "text": "still no"}]])
    with pytest.raises(StructuredOutputError):
        structured(bad, model="m", system="s", prompt="p", schema=V, name="v", description="d")


def test_anthropic_falls_back_to_auto_when_forced_tools_unsupported():
    import sys
    import types

    calls = []

    class BadRequestError(Exception):
        pass

    class Usage:
        input_tokens, output_tokens = 10, 5

    class Resp:
        content, stop_reason, usage = [], "end_turn", Usage()

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("tool_choice", {}).get("type") in ("any", "tool"):
                raise BadRequestError('tool_choice: type "tool" and "any" are not supported for this model.')
            return Resp()

    fake = types.SimpleNamespace(
        Anthropic=lambda **_: types.SimpleNamespace(messages=Messages()),
        BadRequestError=BadRequestError,
        RateLimitError=type("R", (Exception,), {}),
        InternalServerError=type("I", (Exception,), {}),
        APIConnectionError=type("C", (Exception,), {}),
    )
    sys.modules["anthropic"] = fake
    try:
        from sec_research_agent.llm import AnthropicLLM

        llm = AnthropicLLM("key")
        tool = {"name": "final_answer", "description": "", "input_schema": {"type": "object"}}
        for _ in range(2):
            llm.chat(model="m", system="s", messages=[], tools=[tool], tool_choice={"type": "tool", "name": "final_answer"})
    finally:
        del sys.modules["anthropic"]
    # first call rejected, retried with auto; second call goes straight to auto
    assert [c["tool_choice"]["type"] for c in calls] == ["tool", "auto", "auto"]
    assert "Respond only by calling the final_answer tool." in calls[1]["system"]
    assert calls[0]["max_tokens"] == 4096
