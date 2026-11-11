"""Chat backends, behind one interface, with token and cost accounting.

Two backends matter here. `AnthropicBackend` talks to the Claude API and is the
reference system the small model is measured against. `OpenAIBackend` talks to
anything speaking the OpenAI chat API, which covers vLLM on Modal -- the small model
being trained, and any open model standing in as the big one.

Both report token usage, because cost per datasheet is a headline number and an agent
that quietly burns ten tool calls is not cheaper than one big call just because the
model is smaller.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# USD per million tokens. Open models hosted on Modal are priced from GPU time
# instead, in `pricing.py`; these are the hosted-API list prices.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (15.0, 75.0),
    "claude-sonnet-5-5": (3.0, 15.0),
    "claude-haiku-5-5": (1.0, 5.0),
}


@dataclass
class Usage:
    """Tokens and wall-clock accumulated over one or more calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    seconds: float = 0.0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.seconds += other.seconds

    def cost_usd(self, model: str) -> float:
        rate_in, rate_out = PRICES.get(model, (0.0, 0.0))
        return (self.input_tokens * rate_in + self.output_tokens * rate_out) / 1e6

    def to_dict(self, model: str = "") -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": self.calls,
            "seconds": round(self.seconds, 3),
            "cost_usd": round(self.cost_usd(model), 6) if model else None,
        }


@dataclass
class ToolRequest:
    """A tool call the model asked for."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    """One assistant turn: free text, tool requests, and what it cost."""

    text: str
    tool_requests: list[ToolRequest] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_requests)


class Backend(ABC):
    """A chat model that can call tools."""

    model: str

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Reply:
        """One assistant turn."""

    @abstractmethod
    def tool_result_message(self, request: ToolRequest, result: str) -> dict[str, Any]:
        """Wrap a tool result in this backend's message shape."""

    @abstractmethod
    def assistant_message(self, reply: Reply) -> dict[str, Any]:
        """Echo an assistant turn back into the transcript."""

    @abstractmethod
    def format_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert the canonical tool schemas into this backend's wire shape.

        The two APIs disagree: Anthropic takes `{name, description, input_schema}`
        flat, OpenAI nests it under `{"type": "function", "function": {...}}`. Sending
        one shape to the other API is a 400, so the backend owns the conversion rather
        than every caller having to remember which it is talking to.
        """


class AnthropicBackend(Backend):
    """Claude via the Messages API."""

    def __init__(self, model: str = "claude-sonnet-5-5", api_key: str | None = None) -> None:
        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Reply:
        started = time.monotonic()
        response = self._client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            tools=tools or [],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.monotonic() - started

        text_parts: list[str] = []
        requests: list[ToolRequest] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                requests.append(ToolRequest(block.id, block.name, dict(block.input)))

        return Reply(
            text="\n".join(text_parts),
            tool_requests=requests,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                calls=1,
                seconds=elapsed,
            ),
            raw=response,
        )

    def tool_result_message(self, request: ToolRequest, result: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": request.id, "content": result}],
        }

    def assistant_message(self, reply: Reply) -> dict[str, Any]:
        return {"role": "assistant", "content": reply.raw.content}

    def format_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The canonical schemas are already in Anthropic shape."""
        return schemas


class OpenAIBackend(Backend):
    """Any OpenAI-compatible endpoint: vLLM on Modal, or a hosted provider."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
            timeout=timeout,
        )

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Reply:
        started = time.monotonic()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            tools=tools or None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.monotonic() - started

        choice = response.choices[0].message
        requests = [
            ToolRequest(call.id, call.function.name, _loads(call.function.arguments))
            for call in (choice.tool_calls or [])
        ]
        usage = response.usage
        return Reply(
            text=choice.content or "",
            tool_requests=requests,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                calls=1,
                seconds=elapsed,
            ),
            raw=choice,
        )

    def tool_result_message(self, request: ToolRequest, result: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": request.id, "content": result}

    def assistant_message(self, reply: Reply) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": reply.text or None}
        if reply.tool_requests:
            message["tool_calls"] = [
                {
                    "id": r.id,
                    "type": "function",
                    "function": {"name": r.name, "arguments": json.dumps(r.arguments)},
                }
                for r in reply.tool_requests
            ]
        return message

    def format_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["input_schema"],
                },
            }
            for s in schemas
        ]


def _loads(raw: str | None) -> dict[str, Any]:
    """Parse tool arguments. Small models emit malformed JSON often enough to matter."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
