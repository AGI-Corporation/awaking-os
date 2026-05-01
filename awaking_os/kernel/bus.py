"""IAC Bus — asyncio pub/sub used by agents and the consciousness layer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from awaking_os.memory.node import KnowledgeNode


class IACBus:
    """Pub/sub bus over asyncio queues.

    Topics are arbitrary strings; messages are pydantic models. Each
    subscriber gets its own queue, so no message is dropped between
    subscribers and slow consumers don't block others.
    """

    def __init__(self, max_queue_size: int = 1024) -> None:
        self._max_queue_size = max_queue_size
        self._subscribers: dict[str, list[asyncio.Queue[BaseModel]]] = {}
        self._memory_provider: object | None = None

    def attach_memory(self, agi_ram: object) -> None:
        """Wire AGI-RAM into the bus so agents can request memory by task id."""
        self._memory_provider = agi_ram

    async def publish(self, topic: str, message: BaseModel) -> None:
        for q in self._subscribers.get(topic, []):
            await q.put(message)

    async def subscribe(self, topic: str) -> AsyncIterator[BaseModel]:
        queue: asyncio.Queue[BaseModel] = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers.setdefault(topic, []).append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers[topic].remove(queue)

    def subscriber_count(self, topic: str) -> int:
        return len(self._subscribers.get(topic, []))

    async def query_memory(self, task_id: str, k: int = 5) -> list[KnowledgeNode]:
        if self._memory_provider is None:
            return []
        retrieve = getattr(self._memory_provider, "retrieve", None)
        if retrieve is None:
            return []
        return await retrieve(task_id, k=k)
