"""Robustness tests surfaced by the deep review of C.3 + C.4 + C.5.

These cover failure modes that don't show up under happy-path
fixtures: a queue method that raises, a trace sink that throws, and a
retry path whose `put()` fails. The kernel must keep running and
record correct audit state in each case.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from awaking_os.agents.base import Agent
from awaking_os.kernel import (
    AgentRegistry,
    AgentTask,
    AKernel,
    InMemoryTaskQueue,
    PersistentTaskQueue,
    RetryPolicy,
)
from awaking_os.kernel.task import AgentContext, AgentResult
from awaking_os.observability.trace import NullTraceSink, TaskTrace, TraceSink
from awaking_os.types import AgentType


def _task(payload: dict | None = None) -> AgentTask:
    return AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload=payload or {"q": "ping"},
    )


async def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("predicate never became true")
        await asyncio.sleep(0.01)


# --- Worker resilience ----------------------------------------------------


class _FlakyDoneQueue(InMemoryTaskQueue):
    """Wraps the in-memory queue but raises on the first done() call.

    A real-world analog is a transient sqlite write failure during the
    audit step. The worker must NOT die — it should log and keep
    polling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.done_failures = 0
        self.done_successes = 0

    async def done(self, task_id, *, success=True, elapsed_ms=0, error=None):
        if self.done_failures == 0:
            self.done_failures = 1
            raise RuntimeError("transient queue write error")
        self.done_successes += 1
        await super().done(task_id, success=success, elapsed_ms=elapsed_ms, error=error)


async def test_worker_survives_unexpected_queue_done_failure(
    bus, agi_ram, registry_with_echo
) -> None:
    """A queue.done() that raises must NOT kill the worker pool.
    Subsequent tasks should still be dispatched."""
    flaky = _FlakyDoneQueue()
    kernel = AKernel(
        registry=registry_with_echo,
        bus=bus,
        agi_ram=agi_ram,
        task_queue=flaky,
        concurrency=1,
    )
    # First task triggers the done() failure; second proves the worker
    # kept running.
    t1 = _task()
    t2 = _task()
    await kernel.submit(t1)
    await kernel.submit(t2)
    kernel.start()
    await _wait_until(lambda: flaky.done_successes >= 1)
    await _wait_until(lambda: kernel.pending_count == 0)
    await kernel.shutdown()
    # The first done() raised but the worker recovered; the second
    # task's done() succeeded.
    assert flaky.done_failures == 1
    assert flaky.done_successes >= 1


# --- Trace sink robustness ------------------------------------------------


class _BrokenTraceSink(TraceSink):
    """Always raises on write — exercises the dispatch-finally guards."""

    def __init__(self) -> None:
        self.calls = 0

    async def write(self, trace: TaskTrace) -> None:
        self.calls += 1
        raise RuntimeError("sink down")


async def test_dispatch_returns_result_when_trace_sink_raises(
    bus, agi_ram, registry_with_echo
) -> None:
    """A failing TraceSink must not turn a successful dispatch into a
    failed AgentResult — observability errors are non-fatal."""
    sink = _BrokenTraceSink()
    kernel = AKernel(
        registry=registry_with_echo,
        bus=bus,
        agi_ram=agi_ram,
        trace_sink=sink,
    )
    task = _task(payload={"q": "trace-sink-broken"})
    result = await kernel.dispatch(task)
    # Result has the agent's normal output, not an error.
    assert "echo" in result.output
    assert result.task_id == task.id
    # Sink was attempted exactly once.
    assert sink.calls == 1


async def test_failing_sink_does_not_trigger_retry(bus, agi_ram, registry_with_echo) -> None:
    """A task with a retry policy + failing trace sink must NOT retry —
    the agent succeeded; only the trace sink failed."""

    class _CountingSink(TraceSink):
        def __init__(self) -> None:
            self.calls = 0

        async def write(self, trace: TaskTrace) -> None:
            self.calls += 1
            raise RuntimeError("sink down")

    sink = _CountingSink()
    kernel = AKernel(
        registry=registry_with_echo,
        bus=bus,
        agi_ram=agi_ram,
        trace_sink=sink,
    )
    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "no-spurious-retry"},
        retry_policy=RetryPolicy(max_attempts=5, initial_backoff_s=0.0, multiplier=1.0),
    )
    await kernel.submit(task)
    kernel.start()
    await _wait_until(lambda: kernel.pending_count == 0)
    await asyncio.sleep(0.05)  # give a chance for any spurious retry
    await kernel.shutdown()
    # Exactly one dispatch happened — the sink failure didn't masquerade
    # as a task failure.
    assert sink.calls == 1


# --- Retry path put() failure ---------------------------------------------


class _PutFailsAfterFirst(InMemoryTaskQueue):
    """First put() succeeds (the initial submit). Subsequent put()s — i.e.,
    those triggered by the kernel's retry path — raise. Lets the test
    verify that _delayed_resubmit closes the audit row instead of
    leaking the task as in_progress."""

    def __init__(self) -> None:
        super().__init__()
        self.put_calls = 0
        self.done_records: list[tuple[str, bool, str | None]] = []

    async def put(self, task):
        self.put_calls += 1
        if self.put_calls > 1:
            raise RuntimeError("queue write error during retry")
        await super().put(task)

    async def done(self, task_id, *, success=True, elapsed_ms=0, error=None):
        self.done_records.append((task_id, success, error))
        await super().done(task_id, success=success, elapsed_ms=elapsed_ms, error=error)


async def test_retry_marks_task_failed_when_resubmit_put_raises(bus, agi_ram) -> None:
    """If the resubmit put() fails, _delayed_resubmit must close the
    audit row as failed instead of leaving it in_progress until the
    next process restart."""

    class _AlwaysFailAgent(Agent):
        agent_id = "always-fail"
        agent_type = AgentType.SEMANTIC

        async def execute(self, context: AgentContext) -> AgentResult:
            return AgentResult(
                task_id=context.task.id,
                agent_id=self.agent_id,
                output={"error": "first attempt failed"},
            )

    reg = AgentRegistry()
    reg.register(_AlwaysFailAgent())
    queue = _PutFailsAfterFirst()
    kernel = AKernel(registry=reg, bus=bus, agi_ram=agi_ram, task_queue=queue)
    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "retry-put-fails"},
        retry_policy=RetryPolicy(max_attempts=3, initial_backoff_s=0.0, multiplier=1.0),
    )
    await kernel.submit(task)
    kernel.start()
    # Wait until the queue has recorded the failed-audit row.
    await _wait_until(lambda: any(r[0] == task.id for r in queue.done_records))
    await kernel.shutdown()
    # Exactly one done record exists for this task, and it's failed.
    matching = [r for r in queue.done_records if r[0] == task.id]
    assert len(matching) == 1
    _, success, error = matching[0]
    assert success is False
    assert error is not None
    assert "retry-resubmit-failed" in error


# --- Shutdown doesn't hang on long backoffs -------------------------------


async def test_shutdown_aborts_pending_long_backoff_promptly(bus, agi_ram) -> None:
    """A retry sleeping on a 30s backoff must be cancelled within the
    shutdown window — otherwise the kernel can't shut down cleanly
    when a transient agent failure has just been scheduled for retry."""

    class _OnceFailingAgent(Agent):
        agent_id = "once-fail"
        agent_type = AgentType.SEMANTIC
        calls = 0

        async def execute(self, context: AgentContext) -> AgentResult:
            type(self).calls += 1
            return AgentResult(
                task_id=context.task.id,
                agent_id=self.agent_id,
                output={"error": "still flaky"},
            )

    reg = AgentRegistry()
    reg.register(_OnceFailingAgent())
    kernel = AKernel(registry=reg, bus=bus, agi_ram=agi_ram, trace_sink=NullTraceSink())
    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "long-backoff"},
        retry_policy=RetryPolicy(
            max_attempts=10,
            initial_backoff_s=30.0,  # impossibly long for a unit test
            multiplier=1.0,
            max_backoff_s=30.0,
        ),
    )
    await kernel.submit(task)
    kernel.start()
    await _wait_until(lambda: _OnceFailingAgent.calls >= 1)
    # First attempt has run; a 30s backoff is now sleeping. Shutdown
    # must cancel it promptly.
    start = asyncio.get_running_loop().time()
    await kernel.shutdown()
    elapsed = asyncio.get_running_loop().time() - start
    # Shutdown should take well under a second — far less than the 30s
    # backoff. 2s ceiling absorbs any CI scheduling jitter.
    assert elapsed < 2.0, f"shutdown took {elapsed:.2f}s — backoff cancellation broken"


# --- Persistent queue audit when retry put fails --------------------------


async def test_persistent_queue_marks_task_failed_when_retry_put_fails(
    tmp_path, bus, agi_ram
) -> None:
    """End-to-end with PersistentTaskQueue: a retry whose put fails
    should leave the audit row in FAILED state, not orphan it as
    IN_PROGRESS."""

    class _FailingPersistentQueue(PersistentTaskQueue):
        """Persistent queue that fails put() after the first call."""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._put_count = 0

        async def put(self, task):
            self._put_count += 1
            if self._put_count > 1:
                raise RuntimeError("simulated disk error during retry")
            await super().put(task)

    class _AlwaysFailAgent(Agent):
        agent_id = "always-fail"
        agent_type = AgentType.SEMANTIC

        async def execute(self, context: AgentContext) -> AgentResult:
            return AgentResult(
                task_id=context.task.id,
                agent_id=self.agent_id,
                output={"error": "boom"},
            )

    reg = AgentRegistry()
    reg.register(_AlwaysFailAgent())
    pq = _FailingPersistentQueue(tmp_path / "queue.sqlite")
    kernel = AKernel(registry=reg, bus=bus, agi_ram=agi_ram, task_queue=pq)
    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "persistent-retry-fail"},
        retry_policy=RetryPolicy(max_attempts=3, initial_backoff_s=0.0, multiplier=1.0),
    )
    await kernel.submit(task)
    kernel.start()
    await _wait_until(lambda: pq.state_count(PersistentTaskQueue.FAILED) == 1)
    await kernel.shutdown()
    # Audit reflects failure — not an orphan in_progress.
    assert pq.state_count(PersistentTaskQueue.IN_PROGRESS) == 0
    assert pq.state_count(PersistentTaskQueue.FAILED) == 1
