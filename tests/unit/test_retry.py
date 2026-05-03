"""RetryPolicy + kernel retry integration tests."""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import pytest

from awaking_os.agents.base import Agent
from awaking_os.kernel import (
    AgentRegistry,
    AgentTask,
    AKernel,
    PersistentTaskQueue,
    RetryPolicy,
)
from awaking_os.kernel.task import AgentContext, AgentResult
from awaking_os.types import AgentType

# --- RetryPolicy unit tests ------------------------------------------------


def test_default_policy_retries_three_times_with_exponential_backoff() -> None:
    p = RetryPolicy()
    assert p.max_attempts == 3
    assert p.initial_backoff_s == 1.0
    assert p.multiplier == 2.0


def test_should_retry_returns_false_when_error_is_none() -> None:
    p = RetryPolicy(max_attempts=5)
    assert p.should_retry(attempts=1, error=None) is False


def test_should_retry_returns_false_when_attempts_exhausted() -> None:
    p = RetryPolicy(max_attempts=3)
    assert p.should_retry(attempts=3, error="fail") is False
    assert p.should_retry(attempts=4, error="fail") is False


def test_should_retry_returns_true_below_max_attempts() -> None:
    p = RetryPolicy(max_attempts=3)
    assert p.should_retry(attempts=1, error="fail") is True
    assert p.should_retry(attempts=2, error="fail") is True


def test_retry_on_errors_filter_matches_substring() -> None:
    p = RetryPolicy(max_attempts=5, retry_on_errors=("timeout",))
    assert p.should_retry(attempts=1, error="agent timeout") is True
    assert p.should_retry(attempts=1, error="other failure") is False


def test_retry_on_errors_with_multiple_substrings() -> None:
    p = RetryPolicy(max_attempts=5, retry_on_errors=("timeout", "ConnectionError"))
    assert p.should_retry(attempts=1, error="timeout") is True
    assert p.should_retry(attempts=1, error="ConnectionError: ...") is True
    assert p.should_retry(attempts=1, error="ValueError") is False


def test_backoff_is_zero_for_zero_attempts() -> None:
    p = RetryPolicy(initial_backoff_s=1.0, multiplier=2.0)
    assert p.backoff_s(0) == 0.0


def test_backoff_grows_exponentially() -> None:
    p = RetryPolicy(initial_backoff_s=1.0, multiplier=2.0, max_backoff_s=1000.0)
    assert p.backoff_s(1) == 1.0
    assert p.backoff_s(2) == 2.0
    assert p.backoff_s(3) == 4.0
    assert p.backoff_s(4) == 8.0


def test_backoff_clamped_at_max_backoff_s() -> None:
    p = RetryPolicy(initial_backoff_s=1.0, multiplier=2.0, max_backoff_s=5.0)
    assert p.backoff_s(10) == 5.0


def test_max_attempts_below_one_raises() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=0)


def test_negative_backoff_raises() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RetryPolicy(initial_backoff_s=-1.0)


# --- Kernel integration ----------------------------------------------------


class _FlakyAgent(Agent):
    """Fails the first ``fail_count`` times, then succeeds.

    Records every invocation so tests can assert how many attempts ran
    and what their attempts counter looked like.
    """

    def __init__(
        self,
        fail_count: int,
        *,
        error_msg: str = "transient failure",
        agent_type: AgentType = AgentType.SEMANTIC,
    ) -> None:
        self.fail_count = fail_count
        self.error_msg = error_msg
        self.agent_id = "flaky-1"
        self.agent_type = agent_type
        self.calls: list[int] = []  # attempts value at each call

    async def execute(self, context: AgentContext) -> AgentResult:
        self.calls.append(context.task.attempts)
        if len(self.calls) <= self.fail_count:
            return AgentResult(
                task_id=context.task.id,
                agent_id=self.agent_id,
                output={"error": self.error_msg},
            )
        return AgentResult(
            task_id=context.task.id,
            agent_id=self.agent_id,
            output={"ok": True},
        )


def _registry_with(agent: Agent) -> AgentRegistry:
    reg = AgentRegistry()
    reg.register(agent)
    return reg


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("predicate did not become true in time")
        await asyncio.sleep(0.01)


async def test_task_without_retry_policy_does_not_retry(bus, agi_ram) -> None:
    """Legacy behaviour preserved: tasks with no retry_policy fail once."""
    flaky = _FlakyAgent(fail_count=5)
    kernel = AKernel(registry=_registry_with(flaky), bus=bus, agi_ram=agi_ram)
    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "no-retry"},
    )
    await kernel.submit(task)
    kernel.start()
    await _wait_for(lambda: kernel.pending_count == 0)
    await kernel.shutdown()
    assert len(flaky.calls) == 1


async def test_task_with_retry_succeeds_after_transient_failures(bus, agi_ram) -> None:
    """A task that fails twice then succeeds: kernel retries until the
    third call returns ok. The agent sees attempts=0,1,2."""
    flaky = _FlakyAgent(fail_count=2)
    kernel = AKernel(registry=_registry_with(flaky), bus=bus, agi_ram=agi_ram)
    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "retries"},
        retry_policy=RetryPolicy(
            max_attempts=5, initial_backoff_s=0.0, multiplier=1.0, max_backoff_s=0.0
        ),
    )
    await kernel.submit(task)
    kernel.start()
    await _wait_for(lambda: len(flaky.calls) == 3, timeout=3.0)
    await _wait_for(lambda: kernel.pending_count == 0)
    await kernel.shutdown()
    assert flaky.calls == [0, 1, 2]


async def test_task_with_retry_gives_up_after_max_attempts(bus, agi_ram, tmp_path) -> None:
    """Task fails more times than the policy allows: queue marks it failed
    and the agent is invoked exactly max_attempts times."""
    flaky = _FlakyAgent(fail_count=99)
    pq = PersistentTaskQueue(tmp_path / "queue.sqlite")
    kernel = AKernel(registry=_registry_with(flaky), bus=bus, agi_ram=agi_ram, task_queue=pq)
    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "exhausts"},
        retry_policy=RetryPolicy(
            max_attempts=3, initial_backoff_s=0.0, multiplier=1.0, max_backoff_s=0.0
        ),
    )
    await kernel.submit(task)
    kernel.start()
    await _wait_for(lambda: pq.state_count(PersistentTaskQueue.FAILED) == 1, timeout=3.0)
    await kernel.shutdown()
    assert len(flaky.calls) == 3
    assert pq.state_count(PersistentTaskQueue.COMPLETED) == 0


async def test_retry_on_errors_filter_blocks_non_matching_failures(bus, agi_ram) -> None:
    """A retry policy with retry_on_errors=("timeout",) doesn't retry
    when the error is something else."""
    flaky = _FlakyAgent(fail_count=99, error_msg="permanent ValueError")
    kernel = AKernel(registry=_registry_with(flaky), bus=bus, agi_ram=agi_ram)
    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "non-retryable"},
        retry_policy=RetryPolicy(
            max_attempts=10,
            initial_backoff_s=0.0,
            multiplier=1.0,
            max_backoff_s=0.0,
            retry_on_errors=("timeout",),
        ),
    )
    await kernel.submit(task)
    kernel.start()
    await _wait_for(lambda: kernel.pending_count == 0)
    # Give a tiny window in case the loop spins one more iteration; if
    # we'd wrongly retried, calls would be > 1 by now.
    await asyncio.sleep(0.05)
    await kernel.shutdown()
    assert len(flaky.calls) == 1


async def test_retry_respects_backoff_delay(bus, agi_ram) -> None:
    """The kernel waits backoff_s between attempts. Two fast retries with
    a 100ms backoff should take >= 200ms wall-clock."""
    flaky = _FlakyAgent(fail_count=2)
    kernel = AKernel(registry=_registry_with(flaky), bus=bus, agi_ram=agi_ram)
    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "with-backoff"},
        retry_policy=RetryPolicy(
            max_attempts=5,
            initial_backoff_s=0.1,
            multiplier=1.0,
            max_backoff_s=0.1,
        ),
    )
    start = time.monotonic()
    await kernel.submit(task)
    kernel.start()
    await _wait_for(lambda: len(flaky.calls) == 3, timeout=3.0)
    await _wait_for(lambda: kernel.pending_count == 0)
    elapsed = time.monotonic() - start
    await kernel.shutdown()
    # Two backoff windows (after attempts 1 and 2) at 100ms each.
    assert elapsed >= 0.2


async def test_attempts_field_increments_across_retries(bus, agi_ram) -> None:
    """The agent's view of task.attempts should grow 0, 1, 2, ..."""
    flaky = _FlakyAgent(fail_count=4)
    kernel = AKernel(registry=_registry_with(flaky), bus=bus, agi_ram=agi_ram)
    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "watch-attempts"},
        retry_policy=RetryPolicy(
            max_attempts=10, initial_backoff_s=0.0, multiplier=1.0, max_backoff_s=0.0
        ),
    )
    await kernel.submit(task)
    kernel.start()
    await _wait_for(lambda: len(flaky.calls) == 5, timeout=3.0)
    await _wait_for(lambda: kernel.pending_count == 0)
    await kernel.shutdown()
    assert flaky.calls == [0, 1, 2, 3, 4]


async def test_shutdown_cancels_pending_retry_backoff(bus, agi_ram) -> None:
    """If shutdown fires during a backoff sleep, the retry is dropped
    cleanly — no asyncio warnings about un-awaited coroutines."""
    flaky = _FlakyAgent(fail_count=99)
    kernel = AKernel(registry=_registry_with(flaky), bus=bus, agi_ram=agi_ram)
    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "shutdown-mid-backoff"},
        retry_policy=RetryPolicy(
            max_attempts=10,
            initial_backoff_s=10.0,  # long enough to outlive the test
            multiplier=1.0,
            max_backoff_s=10.0,
        ),
    )
    await kernel.submit(task)
    kernel.start()
    await _wait_for(lambda: len(flaky.calls) >= 1, timeout=2.0)
    # The flaky agent's first call returned an error; the kernel is now
    # sleeping on the backoff. Shut down should cancel the pending retry.
    await kernel.shutdown()
    # Only the original attempt happened — the backoff was cancelled
    # before the resubmit landed.
    assert len(flaky.calls) == 1
