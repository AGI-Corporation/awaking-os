"""ExecutiveAgent tests — uses a recording submit fn (no real kernel needed)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from awaking_os.agents.executive import ExecutiveAgent
from awaking_os.kernel.task import AgentContext, AgentTask
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.types import AgentType


class _Recorder:
    def __init__(self) -> None:
        self.submitted: list[AgentTask] = []

    async def submit(self, task: AgentTask) -> str:
        self.submitted.append(task)
        return task.id


def _task(payload: dict | None = None) -> AgentTask:
    return AgentTask(
        id=str(uuid4()),
        agent_type=AgentType.EXECUTIVE,
        payload=payload or {"goal": "research the nature of Phi"},
    )


def _ctx(task: AgentTask) -> AgentContext:
    return AgentContext(task=task, memory=[], ethical_boundary=[])


async def test_default_decomposition_spawns_research_and_semantic(
    semantic_agi_ram: AGIRam,
) -> None:
    rec = _Recorder()
    agent = ExecutiveAgent(agi_ram=semantic_agi_ram, submit=rec.submit)
    result = await agent.execute(_ctx(_task()))

    assert len(rec.submitted) == 2
    types = {st.agent_type for st in rec.submitted}
    assert types == {AgentType.RESEARCH, AgentType.SEMANTIC}
    assert result.output["subtask_ids"] == [st.id for st in rec.submitted]


async def test_include_biotic_spawns_third_subtask(semantic_agi_ram: AGIRam) -> None:
    rec = _Recorder()
    agent = ExecutiveAgent(agi_ram=semantic_agi_ram, submit=rec.submit)
    payload = {
        "goal": "investigate dolphin acoustics",
        "include_biotic": True,
        "biotic_signal_type": "cetacean",
        "biotic_samples": 256,
    }
    await agent.execute(_ctx(_task(payload)))

    assert len(rec.submitted) == 3
    biotic = next(st for st in rec.submitted if st.agent_type == AgentType.BIOTIC)
    assert biotic.payload["signal_type"] == "cetacean"
    assert biotic.payload["samples"] == 256


async def test_subtasks_carry_parent_id_and_constraints(semantic_agi_ram: AGIRam) -> None:
    rec = _Recorder()
    agent = ExecutiveAgent(agi_ram=semantic_agi_ram, submit=rec.submit)
    parent = _task({"goal": "G"})
    parent.ethical_constraints = ["no harm"]
    await agent.execute(_ctx(parent))

    for st in rec.submitted:
        assert st.payload["parent_task_id"] == parent.id
        assert st.ethical_constraints == ["no harm"]


async def test_research_priority_higher_than_semantic(semantic_agi_ram: AGIRam) -> None:
    rec = _Recorder()
    agent = ExecutiveAgent(agi_ram=semantic_agi_ram, submit=rec.submit)
    await agent.execute(_ctx(_task()))

    by_type = {st.agent_type: st for st in rec.submitted}
    assert by_type[AgentType.RESEARCH].priority > by_type[AgentType.SEMANTIC].priority


async def test_persists_plan_node(semantic_agi_ram: AGIRam) -> None:
    rec = _Recorder()
    agent = ExecutiveAgent(agi_ram=semantic_agi_ram, submit=rec.submit)
    result = await agent.execute(_ctx(_task({"goal": "test goal"})))

    assert len(result.knowledge_nodes_created) == 1
    plan_node = await semantic_agi_ram.get(result.knowledge_nodes_created[0])
    assert plan_node is not None
    assert plan_node.type == "event"
    assert plan_node.metadata["goal"] == "test goal"
    assert len(plan_node.metadata["subtask_ids"]) == 2


async def test_missing_goal_raises(semantic_agi_ram: AGIRam) -> None:
    rec = _Recorder()
    agent = ExecutiveAgent(agi_ram=semantic_agi_ram, submit=rec.submit)
    with pytest.raises(ValueError, match="goal"):
        await agent.execute(_ctx(_task({"foo": "bar"})))


async def test_goal_extraction_via_q_key(semantic_agi_ram: AGIRam) -> None:
    rec = _Recorder()
    agent = ExecutiveAgent(agi_ram=semantic_agi_ram, submit=rec.submit)
    await agent.execute(_ctx(_task({"q": "via q"})))
    research = next(st for st in rec.submitted if st.agent_type == AgentType.RESEARCH)
    assert research.payload["topic"] == "via q"


async def test_subtasks_are_actually_kernel_dispatchable(
    semantic_agi_ram: AGIRam, fake_llm
) -> None:
    """End-to-end: Executive's subtasks really run when submitted to a kernel."""
    import asyncio

    from awaking_os.agents.research import ResearchAgent
    from awaking_os.agents.semantic import SemanticAgent
    from awaking_os.io.search import StubSearchTool
    from awaking_os.kernel import AgentRegistry, AKernel, IACBus
    from awaking_os.kernel.kernel import RESULT_TOPIC
    from awaking_os.kernel.task import AgentResult

    bus = IACBus()
    registry = AgentRegistry()
    kernel = AKernel(registry=registry, bus=bus, agi_ram=semantic_agi_ram)
    registry.register(SemanticAgent(llm=fake_llm, agi_ram=semantic_agi_ram))
    registry.register(
        ResearchAgent(llm=fake_llm, search=StubSearchTool(), agi_ram=semantic_agi_ram)
    )
    registry.register(ExecutiveAgent(agi_ram=semantic_agi_ram, submit=kernel.submit))

    received: list[AgentResult] = []

    async def consume() -> None:
        async for msg in bus.subscribe(RESULT_TOPIC):
            received.append(msg)
            if len(received) == 3:  # executive + research + semantic
                return

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)

    await kernel.submit(_task({"goal": "test goal"}))
    kernel.start()
    await asyncio.wait_for(consumer, timeout=2.0)
    while kernel.pending_count > 0:
        await asyncio.sleep(0.01)
    await kernel.shutdown()

    by_agent = {r.agent_id for r in received}
    assert {"executive-1", "research-1", "semantic-1"} == by_agent
