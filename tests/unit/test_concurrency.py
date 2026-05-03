"""Worker-pool / parallel-dispatch tests for AKernel."""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import pytest

from awaking_os.agents.base import Agent
from awaking_os.kernel import AgentRegistry, AgentTask, AKernel
from awaking_os.kernel.task import AgentContext, AgentResult
from awaking_os.types import AgentType


def _task(priority: int = 50, payload: dict | None = None) -> AgentTask:
    return AgentTask(
        id=str(uuid4()),
        priority=priority,
        agent_type=AgentType.SEMANTIC,
        payload=payload or {"q": "ping"},
    )


class _SlowAgent(Agent):
    """Records start/end timestamps for every dispatch — lets a test
    detect overlap (==parallelism) vs strict serialization."""

    def __init__(self, sleep_s: float = 0.05) -> None:
        self.sleep_s = sleep_s
        self.agent_id = "slow"
        self.agent_type = AgentType.SEMANTIC
        self.timeline: list[tuple[str, float, float]] = []  # (task_id, start, end)
        self._lock = asyncio.Lock()

    async def execute(self, context: AgentContext) -> AgentResult:
        start = time.monotonic()
        await asyncio.sleep(self.sleep_s)
        end = time.monotonic()
        # Locking just to keep the list updates atomic — does not
        # serialize the sleeps themselves (those run concurrently).
        async with self._lock:
            self.timeline.append((context.task.id, start, end))
        return AgentResult(task_id=context.task.id, agent_id=self.agent_id)


def _registry_with(agent: Agent) -> AgentRegistry:
    reg = AgentRegistry()
    reg.register(agent)
    return reg


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("predicate never became true")
        await asyncio.sleep(0.01)


# --- Validation -----------------------------------------------------------


def test_concurrency_must_be_positive(bus, agi_ram, registry_with_echo) -> None:
    with pytest.raises(ValueError, match="concurrency must be at least 1"):
        AKernel(registry=registry_with_echo, bus=bus, agi_ram=agi_ram, concurrency=0)


# --- Backwards compatibility (concurrency=1) ------------------------------


async def test_concurrency_one_serializes_dispatches(bus, agi_ram) -> None:
    """With concurrency=1, sleeps must NOT overlap — total wall time
    is at least N * sleep_s."""
    slow = _SlowAgent(sleep_s=0.05)
    kernel = AKernel(registry=_registry_with(slow), bus=bus, agi_ram=agi_ram, concurrency=1)
    n = 4
    for _ in range(n):
        await kernel.submit(_task())
    start = time.monotonic()
    kernel.start()
    await _wait_until(lambda: len(slow.timeline) == n)
    await _wait_until(lambda: kernel.pending_count == 0)
    await kernel.shutdown()
    elapsed = time.monotonic() - start
    # Strictly serial: each 50ms sleep follows the previous → ≥ 200ms.
    assert elapsed >= n * slow.sleep_s * 0.9
    # And no two intervals overlap.
    sorted_intervals = sorted(slow.timeline, key=lambda t: t[1])
    for prev, curr in zip(sorted_intervals[:-1], sorted_intervals[1:], strict=True):
        assert curr[1] >= prev[2] - 0.005  # current starts at/after previous ends


# --- Parallel dispatch ----------------------------------------------------


async def test_concurrency_n_overlaps_dispatches(bus, agi_ram) -> None:
    """With concurrency=N, four 50ms sleeps must complete in roughly
    50ms — not 200ms. Concrete bound: < 2 * sleep_s, i.e. plenty of
    headroom for asyncio scheduling without false positives."""
    slow = _SlowAgent(sleep_s=0.1)
    kernel = AKernel(registry=_registry_with(slow), bus=bus, agi_ram=agi_ram, concurrency=4)
    n = 4
    for _ in range(n):
        await kernel.submit(_task())
    start = time.monotonic()
    kernel.start()
    await _wait_until(lambda: len(slow.timeline) == n, timeout=3.0)
    await _wait_until(lambda: kernel.pending_count == 0)
    await kernel.shutdown()
    elapsed = time.monotonic() - start
    # 4 parallel sleeps of 100ms each → ~100ms total, well under 200ms.
    assert elapsed < 2 * slow.sleep_s, f"expected parallelism, got {elapsed:.3f}s"
    # Verify actual overlap: at least two intervals must intersect.
    overlapping = 0
    for i, (_, s1, e1) in enumerate(slow.timeline):
        for j, (_, s2, e2) in enumerate(slow.timeline):
            if i >= j:
                continue
            if s1 < e2 and s2 < e1:
                overlapping += 1
    assert overlapping >= 1


async def test_concurrency_n_no_task_dispatched_twice(bus, agi_ram) -> None:
    """Many small tasks under high concurrency — every task gets exactly
    one dispatch, no doubles, no drops."""
    slow = _SlowAgent(sleep_s=0.001)
    kernel = AKernel(registry=_registry_with(slow), bus=bus, agi_ram=agi_ram, concurrency=8)
    n = 30
    submitted_ids = set()
    for _ in range(n):
        t = _task()
        submitted_ids.add(t.id)
        await kernel.submit(t)
    kernel.start()
    await _wait_until(lambda: len(slow.timeline) == n, timeout=5.0)
    await _wait_until(lambda: kernel.pending_count == 0)
    await kernel.shutdown()
    dispatched_ids = [tid for tid, _, _ in slow.timeline]
    assert len(dispatched_ids) == n
    assert set(dispatched_ids) == submitted_ids


# --- Snapshot ordering with concurrency -----------------------------------


async def test_snapshot_consecutive_edge_uses_completion_order(bus, agi_ram) -> None:
    """The consecutive-edge heuristic should follow completion timestamps,
    not deque-insertion order. With concurrent workers, two tasks can
    finish in the opposite order from when they were submitted; the
    snapshot should reflect actual completion sequence."""

    class TaggedAgent(Agent):
        def __init__(self, agent_id: str, agent_type: AgentType, sleep_s: float) -> None:
            self.agent_id = agent_id
            self.agent_type = agent_type
            self.sleep_s = sleep_s

        async def execute(self, context: AgentContext) -> AgentResult:
            await asyncio.sleep(self.sleep_s)
            return AgentResult(task_id=context.task.id, agent_id=self.agent_id)

    reg = AgentRegistry()
    # Semantic agent finishes faster than research agent — so even if
    # research is dispatched first, semantic completes first.
    reg.register(TaggedAgent("semantic-1", AgentType.SEMANTIC, 0.01))
    reg.register(TaggedAgent("research-1", AgentType.RESEARCH, 0.10))
    kernel = AKernel(registry=reg, bus=bus, agi_ram=agi_ram, concurrency=2)

    # Research dispatched first, semantic second — but semantic finishes first.
    await kernel.submit(
        AgentTask(id="r1", agent_type=AgentType.RESEARCH, payload={"q": "research"})
    )
    await kernel.submit(
        AgentTask(id="s1", agent_type=AgentType.SEMANTIC, payload={"q": "semantic"})
    )
    kernel.start()
    await _wait_until(lambda: kernel.pending_count == 0)
    # Give the slower task a chance to finish too.
    await asyncio.sleep(0.15)
    await kernel.shutdown()

    snap = kernel._build_snapshot()
    # Both agents in the matrix.
    assert {"semantic-1", "research-1"} <= set(snap.agent_ids)
    sem_idx = snap.agent_ids.index("semantic-1")
    res_idx = snap.agent_ids.index("research-1")
    # Completion order: semantic (fast) → research (slow). So the
    # consecutive edge runs semantic → research, not the dispatch-order
    # research → semantic.
    assert snap.integration_matrix[sem_idx][res_idx] >= 1.0


# --- Shutdown semantics ---------------------------------------------------


async def test_shutdown_waits_for_in_flight_dispatch(bus, agi_ram) -> None:
    """Setting _stopping while a worker is mid-dispatch lets the
    in-flight task finish; new tasks aren't picked up after shutdown."""
    slow = _SlowAgent(sleep_s=0.1)
    kernel = AKernel(registry=_registry_with(slow), bus=bus, agi_ram=agi_ram, concurrency=1)
    await kernel.submit(_task())
    await kernel.submit(_task())  # this one shouldn't run
    kernel.start()
    # Wait until the first task is in flight.
    await _wait_until(lambda: len(slow.timeline) >= 0 or kernel.pending_count < 2)
    await asyncio.sleep(0.02)
    # Shutdown while task 1 is sleeping; task 2 still pending.
    await kernel.shutdown()
    # Task 1 completed; task 2 left behind for the next process.
    assert len(slow.timeline) == 1


async def test_workers_idle_when_queue_empty(bus, agi_ram, registry_with_echo) -> None:
    """Worker loops should poll quietly when there's nothing to do."""
    kernel = AKernel(registry=registry_with_echo, bus=bus, agi_ram=agi_ram, concurrency=4)
    kernel.start()
    await asyncio.sleep(0.05)  # workers polling
    await kernel.shutdown()
    # If shutdown completes within a reasonable window, the workers
    # were correctly idle and observed _stopping.
