"""Thin wrapper over the Anthropic Messages API: cost tracking, fallback model, structured output."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from .config import MODEL_PRICES

log = logging.getLogger(__name__)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = MODEL_PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class ChatResponse:
    content: list[dict[str, Any]]
    stop_reason: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)

    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, resp: ChatResponse) -> None:
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.cost_usd += resp.cost_usd


class LLM(Protocol):
    def chat(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> ChatResponse: ...


FORCED_CHOICES = {"any", "tool"}


def soften_tool_choice(system: str, tool_choice: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """For models without forced tool use: tool_choice auto, plus the rule stated in the prompt."""
    if tool_choice.get("type") == "tool":
        system += f"\n\nRespond only by calling the {tool_choice['name']} tool."
    else:
        system += "\n\nAlways respond by calling one of the tools."
    return system, {"type": "auto"}


class AnthropicLLM:
    def __init__(self, api_key: str, fallback_model: str | None = None, max_retries: int = 3):
        import anthropic

        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY.")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
        self._fallback = fallback_model
        self._no_forced_tools: set[str] = set()  # models that reject tool_choice any/tool

    def chat(self, *, model, system, messages, tools=None, tool_choice=None, max_tokens=1024):
        try:
            return self._call(model, system, messages, tools, tool_choice, max_tokens)
        except (
            self._anthropic.RateLimitError,
            self._anthropic.InternalServerError,
            self._anthropic.APIConnectionError,
        ) as err:
            # fall back only on capacity / outage errors; a bad request would fail on any model
            if not self._fallback or self._fallback == model:
                raise
            log.warning("%s failed (%s); falling back to %s", model, err, self._fallback)
            return self._call(self._fallback, system, messages, tools, tool_choice, max_tokens)

    def _call(self, model, system, messages, tools, tool_choice, max_tokens) -> ChatResponse:
        forced = bool(tool_choice and tool_choice.get("type") in FORCED_CHOICES)
        if forced and model in self._no_forced_tools:
            system, tool_choice = soften_tool_choice(system, tool_choice)
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": messages,
            # room for thinking tokens on models that think by default; billing is per token used
            "max_tokens": max(max_tokens, 4096),
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        try:
            resp = self._client.messages.create(**kwargs)
        except self._anthropic.BadRequestError as err:
            if not (forced and "tool_choice" in str(err)):
                raise
            log.info("%s does not support forced tool use; using tool_choice auto", model)
            self._no_forced_tools.add(model)
            kwargs["system"], kwargs["tool_choice"] = soften_tool_choice(system, tool_choice)
            resp = self._client.messages.create(**kwargs)
        return ChatResponse(
            content=[block.model_dump(exclude_none=True) for block in resp.content],
            stop_reason=resp.stop_reason or "",
            model=model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


class OpenAICompatibleLLM:
    """Same interface, for OpenAI-style chat APIs: Ollama (free, local), Groq, OpenRouter.

    The agent speaks Anthropic's message format, so this class converts on the way out
    (tool_use -> tool_calls, tool_result -> role "tool") and back on the way in.
    Local models follow tool rules less reliably than Claude; the agent nudges them.
    """

    def __init__(self, base_url: str, api_key: str = "ollama", timeout: float = 300.0):
        import httpx

        self._url = base_url.rstrip("/") + "/chat/completions"
        self._http = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)

    def chat(self, *, model, system, messages, tools=None, tool_choice=None, max_tokens=1024):
        sys_text = system
        if tool_choice and tool_choice.get("type") in FORCED_CHOICES:
            sys_text, _ = soften_tool_choice(system, tool_choice)
        if model.startswith("qwen3"):
            sys_text += " /no_think"  # Qwen3 skips its long reasoning trace with this switch
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": sys_text}, *to_openai_messages(messages)],
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = [
                {"type": "function", "function": {
                    "name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"],
                }}
                for t in tools
            ]
            if tool_choice and tool_choice.get("type") == "tool":
                body["tool_choice"] = {"type": "function", "function": {"name": tool_choice["name"]}}
            elif tool_choice and tool_choice.get("type") == "any":
                body["tool_choice"] = "required"
        resp = self._http.post(self._url, json=body)
        resp.raise_for_status()
        data = resp.json()
        forced = tool_choice["name"] if tool_choice and tool_choice.get("type") == "tool" else None
        content = from_openai_message(data["choices"][0]["message"], forced)
        usage = data.get("usage") or {}
        return ChatResponse(
            content=content,
            stop_reason=data["choices"][0].get("finish_reason") or "",
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        texts = [b["text"] for b in content if b.get("type") == "text"]
        if role == "assistant":
            calls = [
                {"id": b["id"], "type": "function",
                 "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                for b in content if b.get("type") == "tool_use"
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
        else:
            for b in content:
                if b.get("type") == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]})
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
    return out


def _parse_json(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def from_openai_message(message: dict[str, Any], forced_tool: str | None = None) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    text = (message.get("content") or "").strip()
    calls = message.get("tool_calls") or []
    if text and not calls:
        # Some local models write the tool call as JSON text instead of a real call.
        parsed = _parse_json(text)
        if forced_tool and parsed is not None:
            calls = [{"function": {"name": forced_tool, "arguments": parsed}}]
            text = ""
    if text:
        blocks.append({"type": "text", "text": text})
    for call in calls:
        args = call["function"].get("arguments") or {}
        if isinstance(args, str):
            args = _parse_json(args) or {}
        blocks.append({
            "type": "tool_use",
            "id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            "name": call["function"]["name"],
            "input": args,
        })
    return blocks


class StructuredOutputError(ValueError):
    pass


def structured[T: BaseModel](
    llm: LLM, *, model: str, system: str, prompt: str, schema: type[T], name: str, description: str
) -> tuple[T, ChatResponse]:
    """Force the model to call one tool whose input schema is the Pydantic model.
    Retries once with the fields spelled out, since small local models sometimes miss the format."""
    json_schema = schema.model_json_schema()
    tool = {"name": name, "description": description, "input_schema": json_schema}
    content = prompt
    for _ in range(2):
        resp = llm.chat(
            model=model,
            system=system,
            messages=[{"role": "user", "content": content}],
            tools=[tool],
            tool_choice={"type": "tool", "name": name},
            max_tokens=1024,
        )
        uses = [u for u in resp.tool_uses() if u["name"] == name] or resp.tool_uses()
        if uses:
            try:
                return schema.model_validate(uses[0]["input"]), resp
            except ValidationError:
                pass
        content = (
            f"{prompt}\n\nCall the {name} tool, or reply with only a JSON object "
            f"with these fields: {json.dumps(json_schema.get('properties', {}))}"
        )
    raise StructuredOutputError(f"{model} did not return valid {name} output")
