"""AKernel — priority-queued task dispatcher."""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from awaking_os.kernel.bus import IACBus
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

    The queue stores ``(-priority, seq, task)`` tuples so the highest
    priority runs first and ties break FIFO via a monotonic counter.
    When ``mc_layer`` is set, after every dispatch the kernel builds a
    :class:`SystemSnapshot` from a sliding window of recent results and
    publishes the resulting :class:`MetaCognitionReport` on the
    ``mc.report`` topic.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        bus: IACBus,
        agi_ram: AGIRam,
        dispatch_timeout_s: float = 30.0,
        mc_layer: MCLayer | None = None,
        snapshot_window: int = 10,
    ) -> None:
        self.registry = registry
        self.bus = bus
        self.agi_ram = agi_ram
        self.dispatch_timeout_s = dispatch_timeout_s
        self.mc_layer = mc_layer
        self._queue: asyncio.PriorityQueue[tuple[int, int, AgentTask]] = asyncio.PriorityQueue()
        self._seq = itertools.count()
        self._stopping = asyncio.Event()
        self._run_task: asyncio.Task[None] | None = None
        self._recent_results: deque[AgentResult] = deque(maxlen=snapshot_window)
        # task_id → (producing agent_id, parent_task_id from payload). Sized to
        # the snapshot window so it doesn't grow unbounded.
        self._task_meta: deque[tuple[str, str, str | None]] = deque(maxlen=snapshot_window * 4)
        self.bus.attach_memory(agi_ram)

    async def submit(self, task: AgentTask) -> str:
        await self._queue.put((-task.priority, next(self._seq), task))
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
            try:
                _, _, task = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            try:
                await self.dispatch(task)
            except Exception:
                logger.exception("Dispatch failed for task %s", task.id)
            finally:
                self._queue.task_done()

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
        return self._queue.qsize()
