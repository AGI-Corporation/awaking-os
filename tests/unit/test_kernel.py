"""AKernel tests."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from awaking_os.agents.base import Agent
from awaking_os.kernel import AgentRegistry, AKernel, IACBus
from awaking_os.kernel.kernel import RESULT_TOPIC
from awaking_os.kernel.task import AgentContext, AgentResult, AgentTask
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.types import AgentType


def _task(priority: int = 50, payload: dict | None = None) -> AgentTask:
    return AgentTask(
        id=str(uuid4()),
        priority=priority,
        agent_type=AgentType.SEMANTIC,
        payload=payload or {"q": "ping"},
    )


async def test_dispatch_runs_registered_agent(kernel: AKernel) -> None:
    result = await kernel.dispatch(_task())
    assert result.agent_id == "echo-1"
    assert result.output["echo"] == {"q": "ping"}
    assert len(result.knowledge_nodes_created) == 1
    assert result.elapsed_ms >= 0


async def test_dispatch_publishes_result_on_bus(kernel: AKernel, bus: IACBus) -> None:
    received: list[AgentResult] = []

    async def consume() -> None:
        async for msg in bus.subscribe(RESULT_TOPIC):
            assert isinstance(msg, AgentResult)
            received.append(msg)
            return

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    task = _task()
    await kernel.dispatch(task)
    await asyncio.wait_for(consumer, timeout=1.0)
    assert received[0].task_id == task.id


async def test_build_context_uses_memory(kernel: AKernel, agi_ram: AGIRam) -> None:
    from awaking_os.memory.node import KnowledgeNode

    await agi_ram.store(KnowledgeNode(content="ping pong", created_by="test"))
    # Empty payload → kernel falls back to task.id as the memory query.
    task = _task(payload={})
    task.id = "ping"
    ctx = await kernel.build_context(task)
    assert any("ping" in n.content for n in ctx.memory)


async def test_build_context_keys_off_payload_content(kernel: AKernel, agi_ram: AGIRam) -> None:
    from awaking_os.memory.node import KnowledgeNode

    # Regression: querying memory by random task UUID never matches anything;
    # the kernel must derive the query from the payload's q/query/topic/goal.
    await agi_ram.store(KnowledgeNode(content="alpha bravo", created_by="test"))
    task = _task(payload={"q": "alpha"})
    ctx = await kernel.build_context(task)
    assert any("alpha" in n.content for n in ctx.memory)


async def test_build_context_payload_topic_used_for_memory_query(
    kernel: AKernel, agi_ram: AGIRam
) -> None:
    from awaking_os.memory.node import KnowledgeNode

    await agi_ram.store(KnowledgeNode(content="cetacean signaling", created_by="test"))
    task = _task(payload={"topic": "cetacean"})
    ctx = await kernel.build_context(task)
    assert any("cetacean" in n.content for n in ctx.memory)


async def test_dispatch_unknown_agent_type_raises(bus: IACBus, agi_ram: AGIRam) -> None:
    registry = AgentRegistry()  # empty
    kernel = AKernel(registry=registry, bus=bus, agi_ram=agi_ram)
    with pytest.raises(KeyError):
        await kernel.dispatch(_task())


async def test_run_loop_processes_in_priority_order(bus: IACBus, agi_ram: AGIRam) -> None:
    seen: list[int] = []

    class RecordingAgent(Agent):
        agent_id = "recording"
        agent_type = AgentType.SEMANTIC

        async def execute(self, context: AgentContext) -> AgentResult:
            seen.append(context.task.priority)
            return AgentResult(task_id=context.task.id, agent_id=self.agent_id)

    registry = AgentRegistry()
    registry.register(RecordingAgent())
    kernel = AKernel(registry=registry, bus=bus, agi_ram=agi_ram)

    await kernel.submit(_task(priority=10))
    await kernel.submit(_task(priority=90))
    await kernel.submit(_task(priority=50))

    kernel.start()
    # Wait for the queue to drain
    while kernel.pending_count > 0:
        await asyncio.sleep(0.01)
    # Give the loop one more tick to dispatch the last item
    await asyncio.sleep(0.05)
    await kernel.shutdown()

    assert seen == [90, 50, 10]


async def test_dispatch_timeout_returns_error_result(bus: IACBus, agi_ram: AGIRam) -> None:
    class SlowAgent(Agent):
        agent_id = "slow"
        agent_type = AgentType.SEMANTIC

        async def execute(self, context: AgentContext) -> AgentResult:
            await asyncio.sleep(10)
            return AgentResult(task_id=context.task.id, agent_id=self.agent_id)

    registry = AgentRegistry()
    registry.register(SlowAgent())
    kernel = AKernel(registry=registry, bus=bus, agi_ram=agi_ram, dispatch_timeout_s=0.05)
    result = await kernel.dispatch(_task())
    assert result.output == {"error": "timeout"}
