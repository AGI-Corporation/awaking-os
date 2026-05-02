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


# --- integration_matrix richness ---------------------------------------------


def _build_two_agent_kernel(bus: IACBus, agi_ram: AGIRam) -> AKernel:
    class TaggedAgent(Agent):
        def __init__(self, agent_id: str, agent_type: AgentType) -> None:
            self.agent_id = agent_id
            self.agent_type = agent_type

        async def execute(self, context: AgentContext) -> AgentResult:
            return AgentResult(task_id=context.task.id, agent_id=self.agent_id)

    registry = AgentRegistry()
    registry.register(TaggedAgent("semantic-1", AgentType.SEMANTIC))
    registry.register(TaggedAgent("research-1", AgentType.RESEARCH))
    return AKernel(registry=registry, bus=bus, agi_ram=agi_ram)


async def test_snapshot_matrix_uses_consecutive_heuristic_without_parent(
    bus: IACBus, agi_ram: AGIRam
) -> None:
    kernel = _build_two_agent_kernel(bus, agi_ram)
    await kernel.dispatch(AgentTask(id="t1", agent_type=AgentType.SEMANTIC, payload={"q": "a"}))
    await kernel.dispatch(AgentTask(id="t2", agent_type=AgentType.RESEARCH, payload={"topic": "b"}))

    snap = kernel._build_snapshot()
    # Two agents, one consecutive edge semantic→research weighted 1.0.
    assert snap.agent_ids == ["semantic-1", "research-1"]
    assert snap.integration_matrix == [[0.0, 1.0], [0.0, 0.0]]


async def test_snapshot_matrix_weights_parent_chain_higher(bus: IACBus, agi_ram: AGIRam) -> None:
    kernel = _build_two_agent_kernel(bus, agi_ram)
    # First task is the "parent"; second carries parent_task_id pointing to it.
    await kernel.dispatch(
        AgentTask(id="parent-1", agent_type=AgentType.SEMANTIC, payload={"q": "a"})
    )
    await kernel.dispatch(
        AgentTask(
            id="child-1",
            agent_type=AgentType.RESEARCH,
            payload={"topic": "b", "parent_task_id": "parent-1"},
        )
    )

    snap = kernel._build_snapshot()
    # Same edge gets both the parent weight (2.0) and the consecutive weight (1.0).
    assert snap.integration_matrix == [[0.0, 3.0], [0.0, 0.0]]


async def test_parent_chain_works_for_non_consecutive_tasks(bus: IACBus, agi_ram: AGIRam) -> None:
    kernel = _build_two_agent_kernel(bus, agi_ram)
    await kernel.dispatch(
        AgentTask(id="parent-2", agent_type=AgentType.SEMANTIC, payload={"q": "a"})
    )
    # Interleave a self-edge that the consecutive heuristic ignores (i==j).
    await kernel.dispatch(AgentTask(id="middle", agent_type=AgentType.SEMANTIC, payload={"q": "b"}))
    await kernel.dispatch(
        AgentTask(
            id="child-2",
            agent_type=AgentType.RESEARCH,
            payload={"topic": "c", "parent_task_id": "parent-2"},
        )
    )

    snap = kernel._build_snapshot()
    # Parent chain still contributes 2.0 even though the parent isn't directly
    # adjacent to the child in dispatch order.
    sem_idx = snap.agent_ids.index("semantic-1")
    res_idx = snap.agent_ids.index("research-1")
    assert snap.integration_matrix[sem_idx][res_idx] >= 2.0
