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


async def test_publish_does_not_block_on_slow_subscribers(bus: IACBus) -> None:
    """A full subscriber queue must not delay delivery to other subscribers
    on the same topic (the bus's stated 'slow consumers don't block others'
    guarantee)."""
    fast_received: list[_Msg] = []

    # Pre-fill a tiny queue to make one subscriber 'slow'.
    slow_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    await slow_queue.put(_Msg(text="filler"))
    bus._subscribers.setdefault("topic.slow", []).append(slow_queue)

    async def fast_consume() -> None:
        async for msg in bus.subscribe("topic.slow"):
            fast_received.append(msg)
            return

    fast = asyncio.create_task(fast_consume())
    await asyncio.sleep(0)

    publish = asyncio.create_task(bus.publish("topic.slow", _Msg(text="hi")))
    # Give the fast consumer time to receive its copy.
    await asyncio.wait_for(fast, timeout=1.0)
    assert fast_received == [_Msg(text="hi")]

    # Drain the slow queue so publish() resolves before the test ends.
    await slow_queue.get()
    await slow_queue.get()
    await asyncio.wait_for(publish, timeout=1.0)


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
