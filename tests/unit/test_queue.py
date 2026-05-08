"""TaskQueue tests — both InMemory and Persistent."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

from awaking_os.kernel.queue import (
    InMemoryTaskQueue,
    PersistentTaskQueue,
    TaskQueue,
)
from awaking_os.kernel.task import AgentTask
from awaking_os.types import AgentType


def _task(priority: int = 50, payload: dict | None = None) -> AgentTask:
    return AgentTask(
        id=str(uuid4()),
        priority=priority,
        agent_type=AgentType.SEMANTIC,
        payload=payload or {"q": "ping"},
    )


# --- Parametrized over both implementations -------------------------------


@pytest.fixture(
    params=[
        pytest.param("inmem", id="InMemoryTaskQueue"),
        pytest.param("persistent", id="PersistentTaskQueue"),
    ]
)
def queue(request, tmp_path: Path) -> TaskQueue:
    if request.param == "inmem":
        return InMemoryTaskQueue()
    return PersistentTaskQueue(tmp_path / "queue.sqlite")


async def test_empty_queue_get_returns_none_after_timeout(queue: TaskQueue) -> None:
    got = await queue.get(timeout=0.05)
    assert got is None
    assert queue.pending_count == 0


async def test_put_then_get_roundtrips_task(queue: TaskQueue) -> None:
    t = _task(priority=42, payload={"q": "hello"})
    await queue.put(t)
    assert queue.pending_count == 1
    got = await queue.get(timeout=1.0)
    assert got is not None
    assert got.id == t.id
    assert got.priority == 42
    assert got.payload == {"q": "hello"}


async def test_higher_priority_dequeues_first(queue: TaskQueue) -> None:
    low = _task(priority=10)
    mid = _task(priority=50)
    high = _task(priority=90)
    await queue.put(low)
    await queue.put(high)
    await queue.put(mid)

    seen = []
    for _ in range(3):
        got = await queue.get(timeout=1.0)
        assert got is not None
        seen.append(got.priority)
    assert seen == [90, 50, 10]


async def test_same_priority_dequeues_fifo(queue: TaskQueue) -> None:
    first = _task(priority=50, payload={"order": "1"})
    second = _task(priority=50, payload={"order": "2"})
    third = _task(priority=50, payload={"order": "3"})
    for t in (first, second, third):
        await queue.put(t)

    ids = []
    for _ in range(3):
        got = await queue.get(timeout=1.0)
        assert got is not None
        ids.append(got.payload["order"])
    assert ids == ["1", "2", "3"]


async def test_done_is_callable_on_completed_task(queue: TaskQueue) -> None:
    t = _task()
    await queue.put(t)
    got = await queue.get(timeout=1.0)
    assert got is not None
    # Just must not raise — InMemory ignores the metadata, Persistent records it.
    await queue.done(t.id, success=True, elapsed_ms=12)


# --- Persistent-only behavior ---------------------------------------------


async def test_persistent_queue_survives_instance_restart(tmp_path: Path) -> None:
    db = tmp_path / "queue.sqlite"
    q1 = PersistentTaskQueue(db)
    t1 = _task(priority=80, payload={"q": "survives"})
    t2 = _task(priority=20, payload={"q": "also survives"})
    await q1.put(t1)
    await q1.put(t2)

    # Simulate process restart by dropping q1 and creating q2 with the same db.
    del q1
    q2 = PersistentTaskQueue(db)
    assert q2.pending_count == 2

    got = await q2.get(timeout=1.0)
    assert got is not None
    assert got.id == t1.id  # higher priority comes first
    assert got.payload == {"q": "survives"}


async def test_in_progress_tasks_recover_to_pending_on_restart(tmp_path: Path) -> None:
    """If a task was in_progress when the process died, the next instance
    should put it back to pending so it gets re-dispatched."""
    db = tmp_path / "queue.sqlite"
    q1 = PersistentTaskQueue(db)
    t = _task(payload={"q": "crashed mid-dispatch"})
    await q1.put(t)
    claimed = await q1.get(timeout=1.0)
    assert claimed is not None
    assert q1.state_count(PersistentTaskQueue.IN_PROGRESS) == 1
    # Crash: drop q1 without calling done().
    del q1

    q2 = PersistentTaskQueue(db)
    # Recovery moved the task back to pending.
    assert q2.state_count(PersistentTaskQueue.IN_PROGRESS) == 0
    assert q2.pending_count == 1
    recovered = await q2.get(timeout=1.0)
    assert recovered is not None
    assert recovered.id == t.id


async def test_attempt_count_increments_on_recovery(tmp_path: Path) -> None:
    db = tmp_path / "queue.sqlite"
    q1 = PersistentTaskQueue(db)
    t = _task()
    await q1.put(t)
    await q1.get(timeout=1.0)
    del q1

    # First recovery: attempt_count goes 0 → 1.
    q2 = PersistentTaskQueue(db)
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT attempt_count FROM task_queue WHERE task_id = ?", (t.id,)
        ).fetchone()
    assert row[0] == 1

    await q2.get(timeout=1.0)
    del q2

    # Second recovery: attempt_count goes 1 → 2.
    q3 = PersistentTaskQueue(db)
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT attempt_count FROM task_queue WHERE task_id = ?", (t.id,)
        ).fetchone()
    assert row[0] == 2
    del q3


async def test_recovery_marks_failed_after_max_attempts(tmp_path: Path) -> None:
    """A poison-pill task that keeps crashing the process gets marked
    failed instead of being retried forever.

    Trace with max_attempts=2 (recovery increments attempt_count from
    its in_progress value, then re-pends if still under the cap):
      put → pending,    attempt_count=0
      get → in_progress, attempt_count=0
      restart 1 → pending,    attempt_count=1   (0 < 2, re-pended)
      get → in_progress, attempt_count=1
      restart 2 → pending,    attempt_count=2   (1 < 2, re-pended)
      get → in_progress, attempt_count=2
      restart 3 → failed,     attempt_count=2   (2 >= 2, marked failed)
    """
    db = tmp_path / "queue.sqlite"
    q = PersistentTaskQueue(db, max_attempts=2)
    t = _task()
    await q.put(t)

    for _ in range(3):
        await q.get(timeout=1.0)
        del q
        q = PersistentTaskQueue(db, max_attempts=2)

    assert q.state_count(PersistentTaskQueue.FAILED) == 1
    assert q.pending_count == 0


async def test_done_writes_audit_row(tmp_path: Path) -> None:
    db = tmp_path / "queue.sqlite"
    q = PersistentTaskQueue(db)
    t = _task()
    await q.put(t)
    await q.get(timeout=1.0)
    await q.done(t.id, success=True, elapsed_ms=42)

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT state, elapsed_ms, error FROM task_queue WHERE task_id = ?",
            (t.id,),
        ).fetchone()
    state, elapsed_ms, error = row
    assert state == PersistentTaskQueue.COMPLETED
    assert elapsed_ms == 42
    assert error is None


async def test_done_with_error_marks_failed(tmp_path: Path) -> None:
    db = tmp_path / "queue.sqlite"
    q = PersistentTaskQueue(db)
    t = _task()
    await q.put(t)
    await q.get(timeout=1.0)
    await q.done(t.id, success=False, elapsed_ms=7, error="agent crashed")

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT state, error FROM task_queue WHERE task_id = ?", (t.id,)
        ).fetchone()
    assert row[0] == PersistentTaskQueue.FAILED
    assert row[1] == "agent crashed"


async def test_concurrent_get_does_not_double_claim(tmp_path: Path) -> None:
    """Within a single instance, the asyncio.Lock serializes get(), but the
    conditional UPDATE in ``_claim_sync`` is the real correctness guarantee.
    This exercises the in-process path."""
    db = tmp_path / "queue.sqlite"
    q = PersistentTaskQueue(db)
    for _ in range(4):
        await q.put(_task())

    # Four concurrent consumers — each should get exactly one distinct task.
    results = await asyncio.gather(
        q.get(timeout=2.0),
        q.get(timeout=2.0),
        q.get(timeout=2.0),
        q.get(timeout=2.0),
    )
    ids = {r.id for r in results if r is not None}
    assert len(ids) == 4
    assert q.pending_count == 0
    assert q.state_count(PersistentTaskQueue.IN_PROGRESS) == 4


async def test_concurrent_get_across_instances_does_not_double_claim(
    tmp_path: Path,
) -> None:
    """Multiple PersistentTaskQueue instances on the same db must not
    double-claim a task. Each instance has its own asyncio.Lock — so the
    only thing preventing duplicate claims is the conditional UPDATE
    in ``_claim_sync`` (``WHERE state = 'pending'``). This test would
    fail if the UPDATE were unconditional."""
    db = tmp_path / "queue.sqlite"
    q1 = PersistentTaskQueue(db)
    q2 = PersistentTaskQueue(db)
    q3 = PersistentTaskQueue(db)
    q4 = PersistentTaskQueue(db)
    for _ in range(4):
        await q1.put(_task())

    results = await asyncio.gather(
        q1.get(timeout=2.0),
        q2.get(timeout=2.0),
        q3.get(timeout=2.0),
        q4.get(timeout=2.0),
    )
    ids = {r.id for r in results if r is not None}
    assert len(ids) == 4
    # Every instance sees the same shared state.
    assert q1.pending_count == 0
    assert q4.state_count(PersistentTaskQueue.IN_PROGRESS) == 4


def test_persistent_queue_max_attempts_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PersistentTaskQueue(tmp_path / "q.sqlite", max_attempts=0)


# --- Kernel integration ---------------------------------------------------


async def test_kernel_with_persistent_queue_records_audit(
    tmp_path: Path, bus, agi_ram, registry_with_echo
) -> None:
    """End-to-end: kernel + PersistentTaskQueue → audit row marked completed."""
    from awaking_os.kernel import AKernel

    pq = PersistentTaskQueue(tmp_path / "queue.sqlite")
    kernel = AKernel(registry=registry_with_echo, bus=bus, agi_ram=agi_ram, task_queue=pq)
    t = _task(payload={"q": "audited"})
    await kernel.submit(t)
    kernel.start()

    deadline = asyncio.get_running_loop().time() + 2.0
    while pq.state_count(PersistentTaskQueue.COMPLETED) == 0:
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("task did not complete in time")
        await asyncio.sleep(0.01)
    await kernel.shutdown()

    with sqlite3.connect(tmp_path / "queue.sqlite") as conn:
        row = conn.execute(
            "SELECT state, elapsed_ms FROM task_queue WHERE task_id = ?", (t.id,)
        ).fetchone()
    assert row[0] == PersistentTaskQueue.COMPLETED
    assert row[1] >= 0


async def test_kernel_with_persistent_queue_resumes_pending_after_restart(
    tmp_path: Path, bus, agi_ram, registry_with_echo
) -> None:
    """Submit tasks via kernel A, never start the loop, then kernel B
    with the same queue picks them up and runs them."""
    from awaking_os.kernel import AKernel

    db = tmp_path / "queue.sqlite"
    pq_a = PersistentTaskQueue(db)
    kernel_a = AKernel(registry=registry_with_echo, bus=bus, agi_ram=agi_ram, task_queue=pq_a)
    t1 = _task(priority=80)
    t2 = _task(priority=20)
    await kernel_a.submit(t1)
    await kernel_a.submit(t2)
    # No kernel_a.start() — simulates a process that took submissions
    # then died before the dispatch loop ran.

    pq_b = PersistentTaskQueue(db)
    assert pq_b.pending_count == 2
    kernel_b = AKernel(registry=registry_with_echo, bus=bus, agi_ram=agi_ram, task_queue=pq_b)
    kernel_b.start()
    deadline = asyncio.get_running_loop().time() + 2.0
    while pq_b.state_count(PersistentTaskQueue.COMPLETED) < 2:
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("queue did not drain in time")
        await asyncio.sleep(0.01)
    await kernel_b.shutdown()
    assert pq_b.pending_count == 0
