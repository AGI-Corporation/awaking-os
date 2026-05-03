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

if TYPE_CHECKING:
    from awaking_os.consciousness.mc_layer import MCLayer
    from awaking_os.consciousness.snapshot import SystemSnapshot
    from awaking_os.memory.agi_ram import AGIRam

logger = logging.getLogger(__name__)

RESULT_TOPIC = "kernel.result"


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
    ) -> None:
        self.registry = registry
        self.bus = bus
        self.agi_ram = agi_ram
        self.dispatch_timeout_s = dispatch_timeout_s
        self.mc_layer = mc_layer
        self.task_queue: TaskQueue = task_queue if task_queue is not None else InMemoryTaskQueue()
        self._stopping = asyncio.Event()
        self._run_task: asyncio.Task[None] | None = None
        self._recent_results: deque[AgentResult] = deque(maxlen=snapshot_window)
        # task_id → (producing agent_id, parent_task_id from payload). Sized to
        # the snapshot window so it doesn't grow unbounded.
        self._task_meta: deque[tuple[str, str, str | None]] = deque(maxlen=snapshot_window * 4)
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
        agent = self.registry.get(task.agent_type)
        context = await self.build_context(task)
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(agent.execute(context), timeout=self.dispatch_timeout_s)
        except TimeoutError:
            logger.warning("Task %s timed out after %.1fs", task.id, self.dispatch_timeout_s)
            result = AgentResult(
                task_id=task.id,
                agent_id=getattr(agent, "agent_id", "unknown"),
                output={"error": "timeout"},
                elapsed_ms=int((time.monotonic() - start) * 1000),
            )
        if result.elapsed_ms == 0:
            result.elapsed_ms = int((time.monotonic() - start) * 1000)
        # Track task → (agent, parent) so the snapshot can build a richer
        # integration matrix from real parent_task_id chains.
        parent_id = task.payload.get("parent_task_id")
        if not isinstance(parent_id, str):
            parent_id = None
        self._task_meta.append((task.id, result.agent_id, parent_id))
        await self.bus.publish(RESULT_TOPIC, result)
        self._recent_results.append(result)
        if self.mc_layer is not None:
            await self._emit_mc_report()
        return result

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

            # Strong signal: parent_task_id chains. If task B was submitted
            # as a sub-task of task A, the agent that handled A causally
            # influenced the agent that handled B.
            task_to_agent = {task_id: aid for task_id, aid, _ in self._task_meta}
            for _task_id, child_agent, parent_id in self._task_meta:
                if parent_id is None or parent_id not in task_to_agent:
                    continue
                parent_agent = task_to_agent[parent_id]
                if parent_agent not in index or child_agent not in index:
                    continue
                i, j = index[parent_agent], index[child_agent]
                if i != j:
                    matrix[i][j] += self._PARENT_EDGE_WEIGHT

            # Weaker signal: consecutive dispatch order. Useful when no
            # parent chain exists (e.g., independent user-submitted tasks).
            for prev, curr in zip(results[:-1], results[1:], strict=True):
                i, j = index[prev.agent_id], index[curr.agent_id]
                if i != j:
                    matrix[i][j] += self._CONSECUTIVE_EDGE_WEIGHT

        return SystemSnapshot(
            timestamp=datetime.now(UTC),
            agent_outputs=results,
            integration_matrix=matrix,
            agent_ids=agent_ids,
        )

    async def run(self) -> None:
        """Main dispatch loop. Stops when ``shutdown()`` is called."""
        while not self._stopping.is_set():
            task = await self.task_queue.get(timeout=0.1)
            if task is None:
                continue
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
                await self.task_queue.done(
                    task.id, success=success, elapsed_ms=elapsed_ms, error=error
                )

    def start(self) -> asyncio.Task[None]:
        if self._run_task is not None and not self._run_task.done():
            return self._run_task
        self._stopping.clear()
        self._run_task = asyncio.create_task(self.run())
        return self._run_task

    async def shutdown(self) -> None:
        self._stopping.set()
        if self._run_task is not None:
            await self._run_task
            self._run_task = None

    @property
    def pending_count(self) -> int:
        return self.task_queue.pending_count
