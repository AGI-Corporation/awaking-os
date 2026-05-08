"""Per-task structured tracing.

A :class:`Tracer` produces a :class:`TaskTrace` — a flat list of
:class:`Span` nodes that form a tree via ``parent_span_id``. Each span
captures a name, start/end times, attributes, and an optional error.
The tracer is per-task: one trace per ``AgentTask`` dispatch.

Spans are opened with the async context manager :meth:`Tracer.span`. On
context entry the span is pushed onto the tracer's stack so nested
spans get the correct ``parent_span_id``. On exit (normal or via
exception) the span is closed with elapsed time computed.

Sinks persist the trace at finalization. The default
:class:`NullTraceSink` is a no-op; :class:`JSONLTraceSink` appends one
trace per line to a file. Sinks are pluggable so an OpenTelemetry
exporter could be wired in later without touching the kernel.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

TRACE_TOPIC = "kernel.trace"


def _now_ms() -> float:
    """Epoch milliseconds. Float so sub-millisecond resolution survives."""
    return time.time() * 1000.0


class Span(BaseModel):
    """A single timed operation within a trace.

    ``parent_span_id`` is ``None`` for the root span of a trace and set
    to the enclosing span's id otherwise. ``start_ms`` and ``end_ms``
    are float epoch milliseconds; ``elapsed_ms`` is ``end_ms - start_ms``
    once the span is closed.
    """

    span_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_span_id: str | None = None
    name: str
    start_ms: float
    end_ms: float | None = None
    elapsed_ms: float | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @property
    def is_open(self) -> bool:
        return self.end_ms is None

    def close(self, *, error: str | None = None) -> None:
        if self.end_ms is not None:
            return  # already closed; idempotent
        self.end_ms = _now_ms()
        self.elapsed_ms = self.end_ms - self.start_ms
        if error is not None:
            self.error = error


class TaskTrace(BaseModel):
    """All spans for a single task dispatch, plus task-level timing."""

    task_id: str
    started_at_ms: float
    ended_at_ms: float | None = None
    elapsed_ms: float | None = None
    spans: list[Span] = Field(default_factory=list)

    def root_spans(self) -> list[Span]:
        return [s for s in self.spans if s.parent_span_id is None]

    def children_of(self, span_id: str) -> list[Span]:
        return [s for s in self.spans if s.parent_span_id == span_id]

    def total_elapsed_ms(self) -> float | None:
        return self.elapsed_ms


class TraceSink(ABC):
    """Where finalized :class:`TaskTrace` instances go."""

    @abstractmethod
    async def write(self, trace: TaskTrace) -> None: ...


class NullTraceSink(TraceSink):
    """Default sink — drops the trace on the floor."""

    async def write(self, trace: TaskTrace) -> None:
        del trace


class JSONLTraceSink(TraceSink):
    """Append-only JSONL file. One trace per line.

    Concurrent writes are serialized through an asyncio lock so two
    trace writes from the same kernel can't interleave bytes. The
    actual file write happens in a thread to avoid blocking the loop
    on disk I/O.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def write(self, trace: TaskTrace) -> None:
        line = trace.model_dump_json() + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append_sync, line)

    def _append_sync(self, line: str) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)


class Tracer:
    """Per-task tracer. Use :meth:`span` to time operations.

    Typical use::

        tracer = Tracer(task_id=task.id, sink=sink)
        async with tracer.span("dispatch", agent=type_):
            async with tracer.span("build_context"):
                ...
            async with tracer.span("agent.execute"):
                ...
        await tracer.finalize()  # writes to sink

    On context exit (normal or via exception) the span is closed with
    its elapsed time. If the inner block raised, the exception is
    captured in :attr:`Span.error` and re-raised so callers can still
    handle it. The error is **not** swallowed.
    """

    def __init__(self, task_id: str, sink: TraceSink | None = None) -> None:
        self._trace = TaskTrace(task_id=task_id, started_at_ms=_now_ms())
        self._sink = sink if sink is not None else NullTraceSink()
        self._stack: list[Span] = []
        self._finalized = False

    @property
    def trace(self) -> TaskTrace:
        return self._trace

    @property
    def sink(self) -> TraceSink:
        return self._sink

    @asynccontextmanager
    async def span(self, name: str, **attrs: Any) -> AsyncIterator[Span]:
        parent_id = self._stack[-1].span_id if self._stack else None
        span = Span(
            parent_span_id=parent_id,
            name=name,
            start_ms=_now_ms(),
            attrs=dict(attrs),
        )
        self._trace.spans.append(span)
        self._stack.append(span)
        try:
            yield span
        except BaseException as e:
            # Capture both regular exceptions and CancelledError so
            # cancelled tasks still leave a useful trace. The exception
            # is re-raised; we don't swallow it.
            span.close(error=repr(e))
            self._stack.pop()
            raise
        else:
            span.close()
            self._stack.pop()

    async def finalize(self) -> None:
        """Seal the trace and hand it to the sink. Idempotent."""
        if self._finalized:
            return
        self._finalized = True
        self._trace.ended_at_ms = _now_ms()
        self._trace.elapsed_ms = self._trace.ended_at_ms - self._trace.started_at_ms
        await self._sink.write(self._trace)
