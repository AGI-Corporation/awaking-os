"""Tracer / Span / TaskTrace tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from awaking_os.observability.trace import (
    JSONLTraceSink,
    NullTraceSink,
    Span,
    TaskTrace,
    Tracer,
)

# --- Span model -----------------------------------------------------------


def test_span_starts_open() -> None:
    s = Span(name="t", start_ms=1.0)
    assert s.is_open is True
    assert s.end_ms is None
    assert s.elapsed_ms is None


def test_span_close_sets_end_and_elapsed() -> None:
    s = Span(name="t", start_ms=10.0)
    s.close()
    assert s.is_open is False
    assert s.end_ms is not None
    assert s.elapsed_ms is not None
    # start_ms came from the literal 10.0; close sets end via _now_ms()
    assert s.elapsed_ms >= 0


def test_span_close_is_idempotent() -> None:
    s = Span(name="t", start_ms=10.0)
    s.close()
    end_first = s.end_ms
    s.close()
    assert s.end_ms == end_first


def test_span_close_records_error() -> None:
    s = Span(name="t", start_ms=0.0)
    s.close(error="boom")
    assert s.error == "boom"


# --- Tracer span semantics ------------------------------------------------


async def test_tracer_records_a_single_span() -> None:
    tracer = Tracer(task_id="task-1")
    async with tracer.span("hello") as s:
        assert s.name == "hello"
        assert s.is_open is True
    assert s.is_open is False
    assert len(tracer.trace.spans) == 1
    assert tracer.trace.spans[0].parent_span_id is None


async def test_nested_spans_track_parent_relationships() -> None:
    tracer = Tracer(task_id="task-1")
    async with tracer.span("outer") as outer:
        async with tracer.span("inner") as inner:
            async with tracer.span("leaf") as leaf:
                assert leaf.parent_span_id == inner.span_id
            assert inner.parent_span_id == outer.span_id
        assert outer.parent_span_id is None
    # All three are recorded; trace.children_of resolves the tree.
    assert len(tracer.trace.spans) == 3
    roots = tracer.trace.root_spans()
    assert len(roots) == 1 and roots[0].name == "outer"
    assert [s.name for s in tracer.trace.children_of(outer.span_id)] == ["inner"]
    assert [s.name for s in tracer.trace.children_of(inner.span_id)] == ["leaf"]


async def test_sibling_spans_share_a_parent() -> None:
    tracer = Tracer(task_id="task-1")
    async with tracer.span("parent") as parent:
        async with tracer.span("a"):
            pass
        async with tracer.span("b"):
            pass
    children = tracer.trace.children_of(parent.span_id)
    assert {s.name for s in children} == {"a", "b"}


async def test_span_attrs_are_preserved() -> None:
    tracer = Tracer(task_id="task-1")
    async with tracer.span("op", agent="semantic", retries=3):
        pass
    s = tracer.trace.spans[0]
    assert s.attrs == {"agent": "semantic", "retries": 3}


async def test_span_records_error_on_exception() -> None:
    tracer = Tracer(task_id="task-1")
    with pytest.raises(ValueError, match="boom"):
        async with tracer.span("op"):
            raise ValueError("boom")
    s = tracer.trace.spans[0]
    assert s.error is not None
    assert "boom" in s.error
    assert s.is_open is False  # closed even on exception


async def test_exception_in_inner_span_pops_stack_correctly() -> None:
    """If an inner span raises, the outer span should still see its own
    parent_id correctly when subsequent siblings are added."""
    tracer = Tracer(task_id="task-1")
    async with tracer.span("outer") as outer:
        with pytest.raises(RuntimeError):
            async with tracer.span("inner-fails"):
                raise RuntimeError("oops")
        # After the exception, the stack should have only `outer` on it.
        async with tracer.span("inner-after") as after:
            assert after.parent_span_id == outer.span_id
    assert {s.name for s in tracer.trace.spans} == {"outer", "inner-fails", "inner-after"}


async def test_finalize_seals_trace() -> None:
    tracer = Tracer(task_id="task-1")
    async with tracer.span("op"):
        pass
    await tracer.finalize()
    assert tracer.trace.ended_at_ms is not None
    assert tracer.trace.elapsed_ms is not None
    assert tracer.trace.elapsed_ms >= 0


async def test_finalize_is_idempotent() -> None:
    tracer = Tracer(task_id="task-1")
    await tracer.finalize()
    first_end = tracer.trace.ended_at_ms
    await tracer.finalize()
    assert tracer.trace.ended_at_ms == first_end


# --- Sinks ----------------------------------------------------------------


async def test_null_sink_drops_trace() -> None:
    tracer = Tracer(task_id="task-1", sink=NullTraceSink())
    async with tracer.span("op"):
        pass
    await tracer.finalize()  # must not raise


async def test_jsonl_sink_appends_one_line_per_trace(tmp_path: Path) -> None:
    sink = JSONLTraceSink(tmp_path / "traces.jsonl")
    for i in range(3):
        tracer = Tracer(task_id=f"task-{i}", sink=sink)
        async with tracer.span("op", i=i):
            pass
        await tracer.finalize()

    lines = (tmp_path / "traces.jsonl").read_text().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [p["task_id"] for p in parsed] == ["task-0", "task-1", "task-2"]
    # Each trace's spans round-trip.
    assert all(len(p["spans"]) == 1 for p in parsed)


async def test_jsonl_sink_concurrent_writes_dont_interleave(tmp_path: Path) -> None:
    """Two concurrent finalize() calls must produce two complete lines —
    not one line with bytes from both writes interleaved."""
    sink = JSONLTraceSink(tmp_path / "traces.jsonl")

    async def trace_one(tid: str) -> None:
        tracer = Tracer(task_id=tid, sink=sink)
        async with tracer.span("op"):
            pass
        await tracer.finalize()

    await asyncio.gather(*(trace_one(f"task-{i}") for i in range(20)))
    lines = (tmp_path / "traces.jsonl").read_text().splitlines()
    assert len(lines) == 20
    # Every line is well-formed JSON (would fail to parse if interleaved).
    parsed = [json.loads(line) for line in lines]
    assert {p["task_id"] for p in parsed} == {f"task-{i}" for i in range(20)}


# --- TaskTrace pydantic round-trip ---------------------------------------


def test_task_trace_round_trips_through_json() -> None:
    t = TaskTrace(task_id="t", started_at_ms=0.0)
    t.spans.append(Span(name="a", start_ms=1.0, end_ms=2.0, elapsed_ms=1.0))
    encoded = t.model_dump_json()
    decoded = TaskTrace.model_validate_json(encoded)
    assert decoded.task_id == "t"
    assert len(decoded.spans) == 1
    assert decoded.spans[0].name == "a"


# --- Kernel integration ---------------------------------------------------


async def test_kernel_publishes_trace_on_dispatch(bus, agi_ram, registry_with_echo) -> None:
    """The kernel publishes a TaskTrace on TRACE_TOPIC after every dispatch.
    The trace should contain build_context and agent.execute spans nested
    under a top-level dispatch span."""
    from uuid import uuid4

    from awaking_os.kernel import TRACE_TOPIC, AKernel
    from awaking_os.kernel.task import AgentTask
    from awaking_os.types import AgentType

    kernel = AKernel(registry=registry_with_echo, bus=bus, agi_ram=agi_ram)
    received: list[TaskTrace] = []

    async def listen() -> None:
        async for msg in bus.subscribe(TRACE_TOPIC):
            received.append(msg)
            return

    listener = asyncio.create_task(listen())
    await asyncio.sleep(0)  # give the subscriber a chance to register

    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "ping"},
    )
    await kernel.dispatch(task)
    await asyncio.wait_for(listener, timeout=2.0)

    assert len(received) == 1
    trace = received[0]
    assert trace.task_id == task.id
    span_names = {s.name for s in trace.spans}
    assert {"dispatch", "build_context", "agent.execute", "bus.publish"} <= span_names


async def test_kernel_writes_trace_to_jsonl_sink(
    tmp_path: Path, bus, agi_ram, registry_with_echo
) -> None:
    """If a JSONLTraceSink is configured, dispatching writes one trace line."""
    from uuid import uuid4

    from awaking_os.kernel import AKernel
    from awaking_os.kernel.task import AgentTask
    from awaking_os.types import AgentType

    sink = JSONLTraceSink(tmp_path / "traces.jsonl")
    kernel = AKernel(registry=registry_with_echo, bus=bus, agi_ram=agi_ram, trace_sink=sink)
    task = AgentTask(
        id=str(uuid4()),
        priority=50,
        agent_type=AgentType.SEMANTIC,
        payload={"q": "ping"},
    )
    await kernel.dispatch(task)

    lines = (tmp_path / "traces.jsonl").read_text().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["task_id"] == task.id
    assert any(s["name"] == "agent.execute" for s in payload["spans"])
