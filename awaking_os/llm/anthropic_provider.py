"""Anthropic SDK wrapper with prompt caching enabled by default.

Defaults: ``claude-opus-4-7`` (latest Opus, per Anthropic guidance), adaptive
thinking on, top-level ``cache_control={"type": "ephemeral"}`` so the system
prompt is auto-cached. Override per-call as needed.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic

from awaking_os.llm.provider import CompletionResult, LLMProvider


class AnthropicProvider(LLMProvider):
    DEFAULT_MODEL = "claude-opus-4-7"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        adaptive_thinking: bool = True,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "AnthropicProvider requires an API key (api_key= or ANTHROPIC_API_KEY env)"
            )
        self._client = anthropic.AsyncAnthropic(api_key=key)
        self._model = model
        self._adaptive_thinking = adaptive_thinking

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> CompletionResult:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if cache_system:
            # Top-level auto-caching — caches the last cacheable block
            # (the system prompt, since render order is tools → system → messages
            # and we have no tools here).
            kwargs["cache_control"] = {"type": "ephemeral"}
        if self._adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        response = await self._client.messages.create(**kwargs)

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = response.usage
        return CompletionResult(
            text=text,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            stop_reason=response.stop_reason,
        )
