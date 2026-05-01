"""IACBus tests."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from awaking_os.kernel import IACBus


class _Msg(BaseModel):
    text: str


async def test_publish_with_no_subscribers_is_noop(bus: IACBus) -> None:
    await bus.publish("topic.x", _Msg(text="hello"))
    assert bus.subscriber_count("topic.x") == 0


async def test_subscriber_receives_published_message(bus: IACBus) -> None:
    received: list[_Msg] = []

    async def consume() -> None:
        async for msg in bus.subscribe("topic.a"):
            assert isinstance(msg, _Msg)
            received.append(msg)
            if len(received) == 2:
                return

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)  # let the subscriber register
    await bus.publish("topic.a", _Msg(text="one"))
    await bus.publish("topic.a", _Msg(text="two"))
    await asyncio.wait_for(consumer, timeout=1.0)

    assert [m.text for m in received] == ["one", "two"]


async def test_multiple_subscribers_each_receive_message(bus: IACBus) -> None:
    out_a: list[str] = []
    out_b: list[str] = []

    async def consume(out: list[str]) -> None:
        async for msg in bus.subscribe("topic.b"):
            out.append(msg.text)  # type: ignore[attr-defined]
            return

    a = asyncio.create_task(consume(out_a))
    b = asyncio.create_task(consume(out_b))
    await asyncio.sleep(0)
    assert bus.subscriber_count("topic.b") == 2

    await bus.publish("topic.b", _Msg(text="broadcast"))
    await asyncio.wait_for(asyncio.gather(a, b), timeout=1.0)

    assert out_a == ["broadcast"]
    assert out_b == ["broadcast"]


async def test_query_memory_returns_empty_when_unattached(bus: IACBus) -> None:
    assert await bus.query_memory("any-id") == []


async def test_query_memory_uses_attached_provider(bus: IACBus, in_memory_agi_ram) -> None:
    from awaking_os.memory.node import KnowledgeNode

    node_id = await in_memory_agi_ram.store(KnowledgeNode(content="alpha bravo", created_by="test"))
    bus.attach_memory(in_memory_agi_ram)
    results = await bus.query_memory("alpha")
    assert any(n.id == node_id for n in results)


@pytest.mark.parametrize("count", [1, 5])
async def test_subscribe_is_cleaned_up_when_consumer_exits(bus: IACBus, count: int) -> None:
    async def consume_once() -> None:
        async for _ in bus.subscribe("topic.cleanup"):
            return

    tasks = [asyncio.create_task(consume_once()) for _ in range(count)]
    await asyncio.sleep(0)
    assert bus.subscriber_count("topic.cleanup") == count

    for _ in range(count):
        await bus.publish("topic.cleanup", _Msg(text="x"))
    await asyncio.gather(*tasks)
    assert bus.subscriber_count("topic.cleanup") == 0
