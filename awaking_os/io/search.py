"""Search tool ABC + a deterministic stub implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


class SearchTool(ABC):
    @abstractmethod
    async def search(self, query: str, k: int = 5) -> list[SearchHit]: ...


class StubSearchTool(SearchTool):
    """Deterministic, in-memory search.

    Returns canned hits for known queries (case-insensitive substring
    match against the registered query). Falls back to ``default_hits``.
    """

    def __init__(
        self,
        responses: dict[str, list[SearchHit]] | None = None,
        default_hits: list[SearchHit] | None = None,
    ) -> None:
        self._responses = {q.lower(): hits for q, hits in (responses or {}).items()}
        self._default = default_hits or []
        self.calls: list[str] = []

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        self.calls.append(query)
        q_lower = query.lower()
        for registered, hits in self._responses.items():
            if registered in q_lower:
                return list(hits[:k])
        return list(self._default[:k])
