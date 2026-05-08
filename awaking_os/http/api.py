"""HTTP API surface for the kernel.

A FastAPI app that wraps an :class:`AKernel` so callers outside the
local process can submit tasks, fetch results, and stream live events.
The kernel runs in the same process as the API; the bus, agi_ram, and
queue are all shared. A future distributed deployment (Phase E.3)
would put the kernel behind Redis and have multiple API instances
share the same backing store.

Endpoints (all JSON unless noted):

- ``POST /submit`` — submit an ``AgentTask``; returns ``{task_id}``
- ``GET  /result/{task_id}`` — latest cached result for a task, 404 if missing
- ``GET  /stream/results`` — SSE feed of every dispatched ``AgentResult``
- ``GET  /stream/traces`` — SSE feed of every ``TaskTrace``
- ``GET  /stream/mc`` — SSE feed of every ``MetaCognitionReport``
- ``GET  /health`` — kernel pending count + concurrency

Auth: when ``AWAKING_API_TOKEN`` is set in the environment, every
endpoint requires ``Authorization: Bearer <token>``. Unset = no auth.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from awaking_os.kernel import (
    RESULT_TOPIC,
    TRACE_TOPIC,
    AgentTask,
    AKernel,
    RetryPolicy,
)
from awaking_os.kernel.task import AgentResult
from awaking_os.types import AgentType

logger = logging.getLogger(__name__)


class SubmitRequest(BaseModel):
    """Body of ``POST /submit``. Mirrors the kernel-facing :class:`AgentTask`
    but accepts an optional id (server-generated when missing) and is
    intentionally permissive about extra payload keys."""

    id: str | None = None
    agent_type: AgentType
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=50, ge=0, le=100)
    ethical_constraints: list[str] = Field(default_factory=list)
    retry_policy: RetryPolicy | None = None


class SubmitResponse(BaseModel):
    task_id: str


class HealthResponse(BaseModel):
    status: str
    pending_count: int
    concurrency: int


# --- Result cache ---------------------------------------------------------


class _ResultCache:
    """FIFO-bounded cache of the most recent ``AgentResult`` per task_id.

    The kernel doesn't index results by task_id — its sliding window is
    ordered by completion timestamp. The HTTP API needs random access by
    task_id (for ``GET /result/{task_id}``), so we maintain our own
    cache populated by a background subscriber on ``RESULT_TOPIC``.

    Bounded so a long-running server doesn't leak memory: at ``maxlen``
    entries, the oldest result is evicted. Callers that miss should
    accept the data is gone — this is a result *cache*, not durable
    storage.
    """

    def __init__(self, maxlen: int = 4096) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be at least 1")
        self._results: OrderedDict[str, AgentResult] = OrderedDict()
        self._maxlen = maxlen

    def set(self, result: AgentResult) -> None:
        # Move-to-end on re-set so a freshly-overwritten task is not
        # treated as oldest.
        if result.task_id in self._results:
            self._results.move_to_end(result.task_id)
        self._results[result.task_id] = result
        while len(self._results) > self._maxlen:
            self._results.popitem(last=False)

    def get(self, task_id: str) -> AgentResult | None:
        return self._results.get(task_id)

    def __len__(self) -> int:
        return len(self._results)


# --- Auth -----------------------------------------------------------------


def _verify_token(authorization: str | None = Header(default=None)) -> None:
    """Bearer-token auth. Off when ``AWAKING_API_TOKEN`` is unset.

    Comparison uses :func:`secrets.compare_digest` to avoid timing
    leaks if the token is short.
    """
    expected = os.environ.get("AWAKING_API_TOKEN")
    if not expected:
        return  # auth disabled
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    presented = authorization.removeprefix("Bearer ").strip()
    import secrets

    if not secrets.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="invalid bearer token")


# --- App factory ----------------------------------------------------------


def create_app(
    kernel: AKernel,
    *,
    result_cache_size: int = 4096,
    manage_kernel_lifecycle: bool = False,
) -> FastAPI:
    """Build a FastAPI app bound to ``kernel``.

    The app's lifespan starts a background subscriber on
    ``RESULT_TOPIC`` to populate the result cache. When
    ``manage_kernel_lifecycle=True`` (the default for the ``serve``
    CLI), the lifespan also calls ``kernel.start()`` on entry and
    ``kernel.shutdown()`` on exit so the API owns the kernel's
    dispatch loop. Tests that already manage the kernel themselves
    leave it ``False``.
    """
    cache = _ResultCache(maxlen=result_cache_size)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        background_tasks: list[asyncio.Task[None]] = []

        async def listen() -> None:
            try:
                async for msg in kernel.bus.subscribe(RESULT_TOPIC):
                    if isinstance(msg, AgentResult):
                        cache.set(msg)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Result-cache subscriber crashed")

        background_tasks.append(asyncio.create_task(listen()))
        if manage_kernel_lifecycle:
            kernel.start()
        try:
            yield
        finally:
            if manage_kernel_lifecycle:
                await kernel.shutdown()
            for t in background_tasks:
                t.cancel()
            for t in background_tasks:
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Background task raised on shutdown")

    app = FastAPI(
        title="Awaking OS",
        description="HTTP surface over the A-Kernel.",
        version="0.1.0",
        lifespan=lifespan,
    )

    auth = Depends(_verify_token)

    @app.post("/submit", response_model=SubmitResponse, dependencies=[auth])
    async def submit(req: SubmitRequest) -> SubmitResponse:
        from uuid import uuid4

        task_id = req.id or str(uuid4())
        task = AgentTask(
            id=task_id,
            priority=req.priority,
            agent_type=req.agent_type,
            payload=req.payload,
            ethical_constraints=req.ethical_constraints,
            retry_policy=req.retry_policy,
        )
        await kernel.submit(task)
        return SubmitResponse(task_id=task.id)

    @app.get("/result/{task_id}", response_model=AgentResult, dependencies=[auth])
    async def get_result(task_id: str) -> AgentResult:
        result = cache.get(task_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"no result cached for task {task_id!r}",
            )
        return result

    @app.get("/health", response_model=HealthResponse, dependencies=[auth])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            pending_count=kernel.pending_count,
            concurrency=kernel.concurrency,
        )

    @app.get("/stream/results", dependencies=[auth])
    async def stream_results() -> EventSourceResponse:
        return EventSourceResponse(_subscribe_sse(kernel, RESULT_TOPIC))

    @app.get("/stream/traces", dependencies=[auth])
    async def stream_traces() -> EventSourceResponse:
        return EventSourceResponse(_subscribe_sse(kernel, TRACE_TOPIC))

    @app.get("/stream/mc", dependencies=[auth])
    async def stream_mc() -> EventSourceResponse:
        from awaking_os.consciousness.mc_layer import MC_REPORT_TOPIC

        return EventSourceResponse(_subscribe_sse(kernel, MC_REPORT_TOPIC))

    # Expose the cache for tests; not part of the public API contract.
    app.state.result_cache = cache  # type: ignore[attr-defined]
    return app


async def _subscribe_sse(kernel: AKernel, topic: str) -> AsyncIterator[dict[str, str]]:
    """Generator that yields SSE events from a bus subscription.

    Each event's ``data`` is the message's ``model_dump_json()``. The
    SSE generator runs until the client disconnects or the kernel's
    bus is torn down — both surface as ``CancelledError`` propagating
    out of the subscribe iterator.
    """
    try:
        async for msg in kernel.bus.subscribe(topic):
            yield {"event": topic, "data": msg.model_dump_json()}
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("SSE subscriber for %s crashed", topic)
