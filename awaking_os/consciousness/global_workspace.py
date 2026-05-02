"""Global Workspace — broadcast salient agent outputs.

Loosely modeled on Global Workspace Theory: the workspace holds the
most recent agent outputs, and ``salient(k)`` returns the top-k by
salience score (phi contribution + nodes created), breaking ties by
recency.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from awaking_os.kernel.task import AgentResult


class GlobalWorkspace:
    def __init__(self, capacity: int = 16) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._history: deque[AgentResult] = deque(maxlen=capacity)

    def broadcast(self, result: AgentResult) -> None:
        self._history.append(result)

    def broadcast_many(self, results: Iterable[AgentResult]) -> None:
        for r in results:
            self.broadcast(r)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._history)

    def history(self) -> list[AgentResult]:
        return list(self._history)

    @staticmethod
    def _salience(result: AgentResult, recency_index: int) -> tuple[float, int]:
        # Higher salience first, then more recent.
        return (
            result.phi_contribution + len(result.knowledge_nodes_created),
            recency_index,
        )

    def salient(self, k: int = 5) -> list[AgentResult]:
        if k <= 0 or not self._history:
            return []
        # ``recency_index`` is larger for newer items
        ranked = sorted(
            ((r, i) for i, r in enumerate(self._history)),
            key=lambda pair: self._salience(pair[0], pair[1]),
            reverse=True,
        )
        return [r for r, _ in ranked[:k]]
