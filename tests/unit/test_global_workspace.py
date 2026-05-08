"""GlobalWorkspace tests."""

from __future__ import annotations

import pytest

from awaking_os.consciousness.global_workspace import GlobalWorkspace
from awaking_os.kernel.task import AgentResult


def _result(
    task_id: str = "t",
    agent_id: str = "a",
    nodes: int = 0,
    phi: float = 0.0,
) -> AgentResult:
    return AgentResult(
        task_id=task_id,
        agent_id=agent_id,
        knowledge_nodes_created=[f"n{i}" for i in range(nodes)],
        phi_contribution=phi,
    )


def test_broadcast_appends_to_history() -> None:
    ws = GlobalWorkspace()
    ws.broadcast(_result())
    assert len(ws) == 1


def test_capacity_evicts_oldest() -> None:
    ws = GlobalWorkspace(capacity=3)
    for i in range(5):
        ws.broadcast(_result(task_id=f"t{i}"))
    history = ws.history()
    assert len(history) == 3
    assert [r.task_id for r in history] == ["t2", "t3", "t4"]


def test_salient_returns_top_k_by_score() -> None:
    ws = GlobalWorkspace()
    ws.broadcast(_result(task_id="boring", nodes=0, phi=0.0))
    ws.broadcast(_result(task_id="middling", nodes=2, phi=0.0))
    ws.broadcast(_result(task_id="exciting", nodes=1, phi=5.0))
    top = ws.salient(2)
    assert [r.task_id for r in top] == ["exciting", "middling"]


def test_salient_breaks_ties_by_recency() -> None:
    ws = GlobalWorkspace()
    ws.broadcast(_result(task_id="old", nodes=1, phi=0.0))
    ws.broadcast(_result(task_id="new", nodes=1, phi=0.0))
    top = ws.salient(1)
    assert [r.task_id for r in top] == ["new"]


def test_salient_with_zero_k_returns_empty() -> None:
    ws = GlobalWorkspace()
    ws.broadcast(_result())
    assert ws.salient(0) == []


def test_salient_on_empty_workspace_returns_empty() -> None:
    assert GlobalWorkspace().salient(5) == []


def test_broadcast_many_appends_all() -> None:
    ws = GlobalWorkspace()
    ws.broadcast_many([_result(task_id="a"), _result(task_id="b"), _result(task_id="c")])
    assert [r.task_id for r in ws.history()] == ["a", "b", "c"]


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        GlobalWorkspace(capacity=0)


def test_capacity_property_exposed() -> None:
    ws = GlobalWorkspace(capacity=8)
    assert ws.capacity == 8
