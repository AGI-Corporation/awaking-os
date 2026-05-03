"""AKernel — priority-queued task dispatcher."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from awaking_os.kernel.bus import IACBus
from awaking_os.kernel.queue import InMemoryTaskQueue, TaskQueue
from awaking_os.kernel.registry import AgentRegistry
from awaking_os.kernel.task import AgentContext, AgentResult, AgentTask
from awaking_os.observability.trace import (
    TRACE_TOPIC,
    NullTraceSink,
    Tracer,
    TraceSink,
)

if TYPE_CHECKING:
    from awaking_os.consciousness.mc_layer import MCLayer
    from awaking_os.consciousness.snapshot import SystemSnapshot
    from awaking_os.memory.agi_ram import AGIRam

logger = logging.getLogger(__name__)

RESULT_TOPIC = "kernel.result"
__all__ = ["AKernel", "RESULT_TOPIC", "TRACE_TOPIC"]


class AKernel:
    """Priority-queued dispatcher.

    The default :class:`InMemoryTaskQueue` keeps tasks in process memory;
    pass a :class:`PersistentTaskQueue` (or any custom :class:`TaskQueue`
    impl) to get crash-recovery and audit history. When ``mc_layer`` is
    set, after every dispatch the kernel builds a :class:`SystemSnapshot`
    from a sliding window of recent results and publishes the resulting
    :class:`MetaCognitionReport` on the ``mc.report`` topic.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        bus: IACBus,
        agi_ram: AGIRam,
        dispatch_timeout_s: float = 30.0,
        mc_layer: MCLayer | None = None,
        snapshot_window: int = 10,
        task_queue: TaskQueue | None = None,
        trace_sink: TraceSink | None = None,
        concurrency: int = 1,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.registry = registry
        self.bus = bus
        self.agi_ram = agi_ram
        self.dispatch_timeout_s = dispatch_timeout_s
        self.mc_layer = mc_layer
        self.concurrency = concurrency
        self.task_queue: TaskQueue = task_queue if task_queue is not None else InMemoryTaskQueue()
        self.trace_sink: TraceSink = trace_sink if trace_sink is not None else NullTraceSink()
        self._stopping = asyncio.Event()
        self._run_task: asyncio.Task[None] | None = None
        # In-flight retry backoffs. Tracked so shutdown can cancel them
        # promptly and asyncio doesn't warn about un-awaited coroutines.
        self._retry_tasks: set[asyncio.Task[None]] = set()
        self._recent_results: deque[AgentResult] = deque(maxlen=snapshot_window)
        # task_id → (agent_id, parent_task_id, completion_ts_monotonic).
        # The timestamp is what _build_snapshot uses to order tasks for
        # the consecutive-edge heuristic — with concurrency > 1, the
        # deque insertion order can race when workers finish at the
        # same time, but completion timestamps preserve causality.
        self._task_meta: deque[tuple[str, str, str | None, float]] = deque(
            maxlen=snapshot_window * 4
        )
        self.bus.attach_memory(agi_ram)

    async def submit(self, task: AgentTask) -> str:
        await self.task_queue.put(task)
        return task.id

    async def build_context(self, task: AgentTask) -> AgentContext:
        # Memory retrieval keys off the task's content, not its UUID — querying
        # by id would never match anything in AGI-RAM. Fall through several
        # common payload keys and fall back to the id as a last resort.
        query = self._memory_query_for(task)
        memory = await self.bus.query_memory(query)
        return AgentContext(
            task=task,
            memory=memory,
            ethical_boundary=task.ethical_constraints,
        )

    @staticmethod
    def _memory_query_for(task: AgentTask) -> str:
        for key in ("q", "query", "question", "topic", "goal", "content"):
            value = task.payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return task.id

    async def dispatch(self, task: AgentTask) -> AgentResult:
        tracer = Tracer(task_id=task.id, sink=self.trace_sink)
        try:
            agent = self.registry.get(task.agent_type)
            agent_id = getattr(agent, "agent_id", "unknown")
            async with tracer.span("dispatch", agent_type=task.agent_type.value, agent_id=agent_id):
                async with tracer.span("build_context"):
                    context = await self.build_context(task)
                start = time.monotonic()
                async with tracer.span("agent.execute", agent_id=agent_id) as exec_span:
                    try:
                        result = await asyncio.wait_for(
                            agent.execute(context), timeout=self.dispatch_timeout_s
                        )
                    except TimeoutError:
                        logger.warning(
                            "Task %s timed out after %.1fs", task.id, self.dispatch_timeout_s
                        )
                        # Mark the span as a timeout so the trace surfaces
                        # it without us having to let the exception out
                        # (callers expect a result, not a raise).
                        exec_span.error = "timeout"
                        result = AgentResult(
                            task_id=task.id,
                            agent_id=agent_id,
                            output={"error": "timeout"},
                            elapsed_ms=int((time.monotonic() - start) * 1000),
                        )
                if result.elapsed_ms == 0:
                    result.elapsed_ms = int((time.monotonic() - start) * 1000)
                # Track task → (agent, parent, completion_ts) for the
                # snapshot's integration matrix. Done before the result
                # publish so concurrent subscribers see consistent state.
                parent_id = task.payload.get("parent_task_id")
                if not isinstance(parent_id, str):
                    parent_id = None
                self._task_meta.append((task.id, result.agent_id, parent_id, time.monotonic()))
                async with tracer.span("bus.publish", topic=RESULT_TOPIC):
                    await self.bus.publish(RESULT_TOPIC, result)
                self._recent_results.append(result)
                if self.mc_layer is not None:
                    async with tracer.span("mc.monitor"):
                        await self._emit_mc_report()
            return result
        finally:
            # Trace persistence + publication is observability — it must
            # NEVER shadow the actual task outcome. Otherwise a sink that
            # raised would look like a task failure to the run loop and
            # trigger spurious retries.
            try:
                await tracer.finalize()
            except Exception:
                logger.exception("Failed to finalize trace for task %s", task.id)
            try:
                await self.bus.publish(TRACE_TOPIC, tracer.trace)
            except Exception:
                logger.exception("Failed to publish trace for task %s", task.id)

    async def _emit_mc_report(self) -> None:
        from awaking_os.consciousness.mc_layer import MC_REPORT_TOPIC

        snapshot = self._build_snapshot()
        report = await self.mc_layer.monitor(snapshot)  # type: ignore[union-attr]
        await self.bus.publish(MC_REPORT_TOPIC, report)

    # Edge weights for the integration matrix. Parent chains are a real
    # causal signal (ExecutiveAgent submitted this sub-task), so they're
    # weighted higher than mere temporal adjacency.
    _PARENT_EDGE_WEIGHT = 2.0
    _CONSECUTIVE_EDGE_WEIGHT = 1.0

    def _build_snapshot(self) -> SystemSnapshot:
        from awaking_os.consciousness.snapshot import SystemSnapshot

        results = list(self._recent_results)
        agent_ids: list[str] = []
        for r in results:
            if r.agent_id not in agent_ids:
                agent_ids.append(r.agent_id)

        n = len(agent_ids)
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)] if n >= 2 else []

        if n >= 2:
            index = {aid: i for i, aid in enumerate(agent_ids)}

            # Order task meta by completion timestamp. With concurrency=1
            # this matches the deque insertion order; with concurrency>1
            # it tracks actual completion order, which is what the
            # consecutive-edge signal claims to capture.
            ordered_meta = sorted(self._task_meta, key=lambda m: m[3])

            # Strong signal: parent_task_id chains. If task B was submitted
            # as a sub-task of task A, the agent that handled A causally
            # influenced the agent that handled B.
            task_to_agent = {task_id: aid for task_id, aid, _, _ in ordered_meta}
            for _task_id, child_agent, parent_id, _ts in ordered_meta:
                if parent_id is None or parent_id not in task_to_agent:
                    continue
                parent_agent = task_to_agent[parent_id]
                if parent_agent not in index or child_agent not in index:
                    continue
                i, j = index[parent_agent], index[child_agent]
                if i != j:
                    matrix[i][j] += self._PARENT_EDGE_WEIGHT

            # Weaker signal: completion-time adjacency. Useful when no
            # parent chain exists (e.g., independent user-submitted tasks).
            for prev_meta, curr_meta in zip(ordered_meta[:-1], ordered_meta[1:], strict=True):
                prev_agent = prev_meta[1]
                curr_agent = curr_meta[1]
                if prev_agent not in index or curr_agent not in index:
                    continue
                i, j = index[prev_agent], index[curr_agent]
                if i != j:
                    matrix[i][j] += self._CONSECUTIVE_EDGE_WEIGHT

        return SystemSnapshot(
            timestamp=datetime.now(UTC),
            agent_outputs=results,
            integration_matrix=matrix,
            agent_ids=agent_ids,
        )

    async def run(self) -> None:
        """Main dispatch loop. Spawns ``concurrency`` worker coroutines
        that all pull from the same task queue. Returns when every
        worker has observed ``_stopping`` and exited."""
        workers = [
            asyncio.create_task(self._worker_loop(), name=f"awaking-worker-{i}")
            for i in range(self.concurrency)
        ]
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise

    async def _worker_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                task = await self.task_queue.get(timeout=0.1)
                if task is None:
                    continue
                await self._process_one(task)
            except asyncio.CancelledError:
                # Hard shutdown — propagate so the worker exits.
                raise
            except Exception:
                # A single bad iteration (sqlite hiccup, unexpected raise
                # inside _process_one's finally) must NOT take the whole
                # pool down. Log and keep polling so the kernel stays
                # responsive for subsequent tasks.
                logger.exception("Worker loop iteration failed; continuing")

    async def _process_one(self, task: AgentTask) -> None:
        success = True
        error: str | None = None
        elapsed_ms = 0
        start = time.monotonic()
        try:
            result = await self.dispatch(task)
            elapsed_ms = result.elapsed_ms
            # An agent can mark itself as failed via output["error"];
            # mirror that into the queue's audit row.
            if isinstance(result.output, dict) and "error" in result.output:
                success = False
                error = str(result.output["error"])
        except Exception as e:
            success = False
            error = repr(e)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.exception("Dispatch failed for task %s", task.id)
        finally:
            # Decide retry vs final-audit. If the task has a retry
            # policy with budget left and a retryable error, the
            # kernel re-pends it after a backoff (without recording
            # a final done() yet). Otherwise the queue audit closes
            # this attempt.
            attempts_so_far = task.attempts + 1
            if (
                not success
                and task.retry_policy is not None
                and task.retry_policy.should_retry(attempts_so_far, error)
            ):
                delay_s = task.retry_policy.backoff_s(attempts_so_far)
                retried = task.model_copy(update={"attempts": attempts_so_far})
                self._track_retry(retried, delay_s)
            else:
                await self.task_queue.done(
                    task.id, success=success, elapsed_ms=elapsed_ms, error=error
                )

    def _track_retry(self, task: AgentTask, delay_s: float) -> None:
        """Schedule a retry without blocking the dispatch loop.

        The resubmit task is stashed on ``self._retry_tasks`` so
        :meth:`shutdown` can wait for in-flight backoffs to complete or
        be cancelled — otherwise asyncio would warn about an
        un-awaited coroutine on shutdown.
        """
        coro = self._delayed_resubmit(task, delay_s)
        retry_task = asyncio.create_task(coro)
        self._retry_tasks.add(retry_task)
        retry_task.add_done_callback(self._retry_tasks.discard)

    async def _delayed_resubmit(self, task: AgentTask, delay_s: float) -> None:
        # Early exit: if shutdown raced ahead of our schedule, don't even
        # start the backoff sleep. The queue row stays in_progress and
        # PersistentTaskQueue._recover_in_progress will re-pend it on
        # the next process startup.
        if self._stopping.is_set():
            return
        try:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            if self._stopping.is_set():
                # Shutdown happened during the sleep; same recovery path.
                return
            await self.task_queue.put(task)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # put() itself failed (e.g., sqlite write error). The task
            # would otherwise be orphaned with state=in_progress until
            # the next restart. Close the audit row as failed so callers
            # see a definitive outcome.
            logger.exception("Failed to resubmit task %s for retry", task.id)
            try:
                await self.task_queue.done(
                    task.id,
                    success=False,
                    elapsed_ms=0,
                    error=f"retry-resubmit-failed: {e!r}",
                )
            except Exception:
                logger.exception("Also failed to close audit row for task %s", task.id)

    def start(self) -> asyncio.Task[None]:
        if self._run_task is not None and not self._run_task.done():
            return self._run_task
        self._stopping.clear()
        self._run_task = asyncio.create_task(self.run())
        return self._run_task

    async def shutdown(self) -> None:
        self._stopping.set()
        # Cancel currently-pending retries so their backoff sleeps abort
        # immediately. Done before waiting on workers because a
        # multi-second backoff would otherwise stretch shutdown latency.
        for retry_task in list(self._retry_tasks):
            retry_task.cancel()
        # Wait for the worker pool to drain in-flight dispatches.
        # Workers see _stopping at their next loop iteration and exit.
        if self._run_task is not None:
            await self._run_task
            self._run_task = None
        # A worker may have scheduled a new retry between our cancel
        # snapshot and the run-task drain. Those tasks see _stopping
        # set on entry and return immediately, but we still gather them
        # so asyncio doesn't warn about un-awaited coroutines.
        if self._retry_tasks:
            await asyncio.gather(*self._retry_tasks, return_exceptions=True)

    @property
    def pending_count(self) -> int:
        return self.task_queue.pending_count
