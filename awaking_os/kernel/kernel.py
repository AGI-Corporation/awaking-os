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
        self.bus.attach_memory(agi_ram)

    async def submit(self, task: AgentTask) -> str:
        await self._queue.put((-task.priority, next(self._seq), task))
        return task.id

    async def build_context(self, task: AgentTask) -> AgentContext:
        memory = await self.bus.query_memory(task.id)
        return AgentContext(
            task=task,
            memory=memory,
            ethical_boundary=task.ethical_constraints,
        )

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

    def _build_snapshot(self) -> SystemSnapshot:
        from awaking_os.consciousness.snapshot import SystemSnapshot

        results = list(self._recent_results)
        agent_ids: list[str] = []
        for r in results:
            if r.agent_id not in agent_ids:
                agent_ids.append(r.agent_id)

        n = len(agent_ids)
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)] if n >= 2 else []
        # Heuristic: consecutive results imply causal influence (i → j),
        # since the kernel runs them in dispatch order on a single loop.
        if n >= 2:
            index = {aid: i for i, aid in enumerate(agent_ids)}
            for prev, curr in zip(results[:-1], results[1:], strict=True):
                i, j = index[prev.agent_id], index[curr.agent_id]
                if i != j:
                    matrix[i][j] += 1.0

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
