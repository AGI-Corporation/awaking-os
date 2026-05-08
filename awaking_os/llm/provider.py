"""LLM provider ABC + a deterministic fake for tests."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompletionResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    stop_reason: str | None = None


class LLMProvider(ABC):
    """Async LLM completion interface.

    Implementations should treat ``cache_system=True`` as a hint to enable
    prompt caching for the system prompt. The ABC does not mandate caching;
    it just documents the surface so agents can request it.
    """

    @abstractmethod
    async def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> CompletionResult: ...


@dataclass
class _CallRecord:
    system: str
    messages: list[dict[str, Any]]
    max_tokens: int
    cache_system: bool


class FakeLLMProvider(LLMProvider):
    """Deterministic LLM for tests.

    Returns a canned response if the (system, messages) hash matches an
    entry in ``responses``; otherwise ``default_response``. Tracks calls
    and simulates prompt caching: the first call with a given system
    prompt registers a cache write; subsequent calls register a cache
    read.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str = "[fake llm response]",
        model: str = "fake-model",
    ) -> None:
        self._responses = responses or {}
        self._default = default_response
        self._model = model
        self.calls: list[_CallRecord] = []
        self._cached_systems: set[str] = set()

    async def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        cache_system: bool = True,
    ) -> CompletionResult:
        self.calls.append(_CallRecord(system, list(messages), max_tokens, cache_system))
        text = self._responses.get(self._key(system, messages), self._default)

        sys_tokens = max(1, len(system.split()))
        msg_tokens = sum(len(json.dumps(m).split()) for m in messages) or 1
        cache_creation = 0
        cache_read = 0
        if cache_system:
            if system in self._cached_systems:
                cache_read = sys_tokens
            else:
                cache_creation = sys_tokens
                self._cached_systems.add(system)

        return CompletionResult(
            text=text,
            model=self._model,
            input_tokens=msg_tokens,
            output_tokens=max(1, len(text.split())),
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            stop_reason="end_turn",
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @staticmethod
    def _key(system: str, messages: list[dict[str, Any]]) -> str:
        payload = system + "|" + json.dumps(messages, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()
