"""Retry policy for failed task dispatches.

A :class:`RetryPolicy` lives on an :class:`AgentTask`. When the kernel
sees a failure (agent self-reports ``output["error"]`` or raises an
exception caught by the run loop), it consults the policy to decide
whether to re-pend the task and how long to wait first. Exponential
backoff with a cap is the default schedule.

The policy is opt-in: tasks without a ``retry_policy`` keep the legacy
"one shot, then audit as failed" behavior. This keeps the foundation
PRs' contracts intact while giving callers a knob to flip per-task.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetryPolicy(BaseModel):
    """Per-task retry policy.

    ``max_attempts`` counts total attempts including the first try, so
    ``max_attempts=1`` (the validated minimum) means "no retries". The
    backoff is exponential — ``initial_backoff_s * multiplier ** (n-1)``
    after the n-th attempt — and capped at ``max_backoff_s``.

    ``retry_on_errors`` is a tuple of substrings; when non-empty, only
    failures whose error message contains one of them retry. The
    default empty tuple means "retry on any error" (most permissive).
    """

    max_attempts: int = Field(default=3, ge=1, le=100)
    initial_backoff_s: float = Field(default=1.0, ge=0.0)
    multiplier: float = Field(default=2.0, gt=0.0)
    max_backoff_s: float = Field(default=60.0, ge=0.0)
    retry_on_errors: tuple[str, ...] = ()

    def should_retry(self, attempts: int, error: str | None) -> bool:
        """Return True iff this is a retryable failure with budget left.

        ``attempts`` is the count of attempts made so far *including*
        the one that just failed. So after the first failure, callers
        pass ``attempts=1``; the policy retries iff ``1 < max_attempts``.
        """
        if error is None:
            return False
        if attempts >= self.max_attempts:
            return False
        if not self.retry_on_errors:
            return True
        return any(needle in error for needle in self.retry_on_errors)

    def backoff_s(self, attempts: int) -> float:
        """Seconds to wait before the next attempt.

        ``attempts`` is the count of attempts already made; ``attempts=1``
        gives ``initial_backoff_s``, ``attempts=2`` gives that times
        ``multiplier``, and so on, all clamped to ``max_backoff_s``.
        """
        if attempts < 1:
            return 0.0
        delay = self.initial_backoff_s * (self.multiplier ** (attempts - 1))
        return min(delay, self.max_backoff_s)
