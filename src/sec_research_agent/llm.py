"""Thin wrapper over the Anthropic Messages API: cost tracking, fallback model, structured output."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .config import MODEL_PRICES

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


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


class AnthropicLLM:
    def __init__(self, api_key: str, fallback_model: str | None = None, max_retries: int = 3):
        import anthropic

        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY.")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
        self._fallback = fallback_model

    def chat(self, *, model, system, messages, tools=None, tool_choice=None, max_tokens=1024):
        try:
            return self._call(model, system, messages, tools, tool_choice, max_tokens)
        except (self._anthropic.APIStatusError, self._anthropic.APIConnectionError) as err:
            if not self._fallback or self._fallback == model:
                raise
            log.warning("%s failed (%s); falling back to %s", model, err, self._fallback)
            return self._call(self._fallback, system, messages, tools, tool_choice, max_tokens)

    def _call(self, model, system, messages, tools, tool_choice, max_tokens) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        resp = self._client.messages.create(**kwargs)
        return ChatResponse(
            content=[block.model_dump(exclude_none=True) for block in resp.content],
            stop_reason=resp.stop_reason or "",
            model=model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


def structured(
    llm: LLM, *, model: str, system: str, prompt: str, schema: type[T], name: str, description: str
) -> tuple[T, ChatResponse]:
    """Force the model to call one tool whose input schema is the Pydantic model."""
    tool = {"name": name, "description": description, "input_schema": schema.model_json_schema()}
    resp = llm.chat(
        model=model,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        tools=[tool],
        tool_choice={"type": "tool", "name": name},
        max_tokens=512,
    )
    uses = resp.tool_uses()
    if not uses:
        raise ValueError(f"Model did not return structured output for {name}")
    return schema.model_validate(uses[0]["input"]), resp
