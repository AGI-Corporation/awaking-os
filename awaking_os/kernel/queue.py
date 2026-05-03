"""Task-queue ABC + in-memory and sqlite-backed implementations.

The kernel uses a :class:`TaskQueue` to enqueue and dequeue
:class:`AgentTask` instances. Two implementations ship:

- :class:`InMemoryTaskQueue` — wraps :class:`asyncio.PriorityQueue`. Same
  semantics as before this module existed: process restart drops every
  pending task on the floor.
- :class:`PersistentTaskQueue` — sqlite-backed durable queue. On
  startup, any task that was ``in_progress`` when the previous process
  exited is recovered to ``pending`` (with ``attempt_count`` bumped), so
  a crashed dispatch resumes after restart. Completed tasks are kept in
  an audit table with elapsed_ms + final state.

Both implement the same async API; the kernel doesn't care which it
got. A future Redis-backed queue would slot in identically.
"""

from __future__ import annotations

import asyncio
import itertools
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path

from awaking_os.kernel.task import AgentTask


class TaskQueue(ABC):
    @abstractmethod
    async def put(self, task: AgentTask) -> None: ...

    @abstractmethod
    async def get(self, timeout: float = 0.1) -> AgentTask | None:
        """Pop the highest-priority pending task. Returns ``None`` on timeout."""

    @abstractmethod
    async def done(
        self,
        task_id: str,
        *,
        success: bool = True,
        elapsed_ms: int = 0,
        error: str | None = None,
    ) -> None:
        """Mark a previously-``get()``-ed task as finished."""

    @property
    @abstractmethod
    def pending_count(self) -> int: ...


# --- In-memory ----------------------------------------------------------------


class InMemoryTaskQueue(TaskQueue):
    """asyncio.PriorityQueue under the hood. Loses pending tasks on restart."""

    def __init__(self) -> None:
        self._q: asyncio.PriorityQueue[tuple[int, int, AgentTask]] = asyncio.PriorityQueue()
        self._seq = itertools.count()

    async def put(self, task: AgentTask) -> None:
        # Negate priority so larger (more urgent) values come out first.
        await self._q.put((-task.priority, next(self._seq), task))

    async def get(self, timeout: float = 0.1) -> AgentTask | None:
        try:
            _, _, task = await asyncio.wait_for(self._q.get(), timeout=timeout)
        except TimeoutError:
            return None
        return task

    async def done(
        self,
        task_id: str,
        *,
        success: bool = True,
        elapsed_ms: int = 0,
        error: str | None = None,
    ) -> None:
        # The asyncio.PriorityQueue tracks task_done as a counter, not
        # per-id. Audit metadata is dropped here — InMemoryTaskQueue
        # doesn't keep history. Use PersistentTaskQueue if you want it.
        del task_id, success, elapsed_ms, error
        self._q.task_done()

    @property
    def pending_count(self) -> int:
        return self._q.qsize()


# --- Persistent (sqlite) ------------------------------------------------------


class PersistentTaskQueue(TaskQueue):
    """sqlite-backed durable queue with crash-recovery + audit history.

    Schema:

    .. code-block:: sql

        CREATE TABLE task_queue (
            task_id TEXT PRIMARY KEY,
            priority INTEGER NOT NULL,        -- higher = more urgent
            seq INTEGER NOT NULL,             -- FIFO tiebreak within priority
            payload_json TEXT NOT NULL,       -- AgentTask.model_dump_json()
            state TEXT NOT NULL,              -- pending|in_progress|completed|failed
            attempt_count INTEGER NOT NULL DEFAULT 0,
            enqueued_at REAL NOT NULL,
            started_at REAL,
            finished_at REAL,
            elapsed_ms INTEGER,
            error TEXT
        )

    The ``(state, priority DESC, seq ASC)`` index makes ``get()`` O(log n).

    On construction, any rows still ``in_progress`` from a previous
    process are recovered to ``pending`` and their ``attempt_count`` is
    incremented. Tasks are at-least-once: agents must be idempotent (or
    the caller can use the audit table to dedupe).
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

    def __init__(self, db_path: Path, *, max_attempts: int = 3) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_attempts = max_attempts
        self._lock = asyncio.Lock()
        self._init_db()
        self._recover_in_progress()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS task_queue (
                    task_id TEXT PRIMARY KEY,
                    priority INTEGER NOT NULL,
                    seq INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    enqueued_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    elapsed_ms INTEGER,
                    error TEXT
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_state_priority "
                "ON task_queue(state, priority DESC, seq ASC)"
            )

    def _recover_in_progress(self) -> int:
        """Move any in_progress rows back to pending. Returns the count moved."""
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "UPDATE task_queue SET state = ?, attempt_count = attempt_count + 1, "
                "started_at = NULL WHERE state = ? AND attempt_count < ?",
                (self.PENDING, self.IN_PROGRESS, self._max_attempts),
            )
            recovered = cur.rowcount
            # Tasks that have already exhausted their attempts when seen
            # in_progress on startup get marked failed rather than retried
            # forever.
            conn.execute(
                "UPDATE task_queue SET state = ?, finished_at = ?, "
                "error = COALESCE(error, ?) WHERE state = ? AND attempt_count >= ?",
                (
                    self.FAILED,
                    time.time(),
                    "exceeded max_attempts during recovery",
                    self.IN_PROGRESS,
                    self._max_attempts,
                ),
            )
            return int(recovered)

    @staticmethod
    def _next_seq(conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM task_queue").fetchone()
        return int(row[0])

    async def put(self, task: AgentTask) -> None:
        async with self._lock:
            await asyncio.to_thread(self._put_sync, task)

    def _put_sync(self, task: AgentTask) -> None:
        with sqlite3.connect(self._db_path) as conn:
            seq = self._next_seq(conn)
            conn.execute(
                "INSERT OR REPLACE INTO task_queue "
                "(task_id, priority, seq, payload_json, state, enqueued_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    task.id,
                    task.priority,
                    seq,
                    task.model_dump_json(),
                    self.PENDING,
                    time.time(),
                ),
            )

    async def get(self, timeout: float = 0.1) -> AgentTask | None:
        # Poll up to ``timeout``, in 10 ms increments. A future Redis
        # backend would use a blocking pop; sqlite needs polling.
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            async with self._lock:
                task = await asyncio.to_thread(self._claim_sync)
            if task is not None:
                return task
            if asyncio.get_event_loop().time() >= deadline:
                return None
            await asyncio.sleep(0.01)

    def _claim_sync(self) -> AgentTask | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT task_id, payload_json FROM task_queue "
                "WHERE state = ? ORDER BY priority DESC, seq ASC LIMIT 1",
                (self.PENDING,),
            ).fetchone()
            if row is None:
                return None
            task_id, payload_json = row
            now = time.time()
            cur = conn.execute(
                "UPDATE task_queue SET state = ?, started_at = ? WHERE task_id = ? AND state = ?",
                (self.IN_PROGRESS, now, task_id, self.PENDING),
            )
            if cur.rowcount == 0:
                # Lost a race with another consumer. Caller will re-try.
                return None
            return AgentTask.model_validate_json(payload_json)

    async def done(
        self,
        task_id: str,
        *,
        success: bool = True,
        elapsed_ms: int = 0,
        error: str | None = None,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(self._done_sync, task_id, success, elapsed_ms, error)

    def _done_sync(self, task_id: str, success: bool, elapsed_ms: int, error: str | None) -> None:
        state = self.COMPLETED if success else self.FAILED
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE task_queue SET state = ?, finished_at = ?, "
                "elapsed_ms = ?, error = ? WHERE task_id = ?",
                (state, time.time(), int(elapsed_ms), error, task_id),
            )

    @property
    def pending_count(self) -> int:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM task_queue WHERE state = ?",
                (self.PENDING,),
            ).fetchone()
        return int(row[0]) if row else 0

    def state_count(self, state: str) -> int:
        """Count of tasks in a given state. Useful for tests + dashboards."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM task_queue WHERE state = ?", (state,)
            ).fetchone()
        return int(row[0]) if row else 0
