"""Simulation engine tests."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from awaking_os.consciousness.snapshot import MetaCognitionReport
from awaking_os.io.search import SearchHit, StubSearchTool
from awaking_os.kernel.task import AgentResult, AgentTask
from awaking_os.simulation import (
    Expectation,
    Experiment,
    Hypothesis,
    Sandbox,
    expect_agent_produced,
    expect_alignment_at_least,
    expect_no_deviating_agents,
    expect_node_count_at_least,
    expect_phi_at_least,
    expect_triggered_rule,
)
from awaking_os.types import AgentType

# --- Hypothesis evaluation (no kernel needed) --------------------------------


def _result(agent_id: str = "a", nodes: int = 1) -> AgentResult:
    return AgentResult(
        task_id=str(uuid4()),
        agent_id=agent_id,
        knowledge_nodes_created=[f"n{i}" for i in range(nodes)],
    )


def _report(
    alignment_score: float = 1.0,
    phi: float = 0.0,
    deviating: list[str] | None = None,
    triggered: list[str] | None = None,
) -> MetaCognitionReport:
    return MetaCognitionReport(
        timestamp=datetime.now(UTC),
        phi_value=phi,
        alignment_score=alignment_score,
        deviating_agents=deviating or [],
        triggered_rules=triggered or [],
    )


def _experiment(
    expectations: list[Expectation],
    *,
    results: list[AgentResult] | None = None,
    reports: list[MetaCognitionReport] | None = None,
) -> Experiment:
    h = Hypothesis(id="h", description="", expectations=expectations)
    return Experiment(
        hypothesis=h,
        tasks=[],
        results=results or [],
        mc_reports=reports or [],
    )


def test_no_expectations_means_not_confirmed() -> None:
    """An empty expectation list is treated as 'no claim made' → not confirmed."""
    exp = _experiment([])
    exp.evaluate()
    assert exp.confirmed is False
    assert exp.failed_expectations == []


def test_all_passing_expectations_confirm() -> None:
    exp = _experiment(
        [expect_agent_produced("semantic-1")],
        results=[_result(agent_id="semantic-1")],
    )
    exp.evaluate()
    assert exp.confirmed is True


def test_failing_expectation_records_name() -> None:
    exp = _experiment(
        [expect_agent_produced("never-1")],
        results=[_result(agent_id="other")],
    )
    exp.evaluate()
    assert exp.confirmed is False
    assert "agent_produced[never-1]" in exp.failed_expectations


def test_predicate_exception_is_a_failure_not_a_crash() -> None:
    def boom(_: Experiment) -> bool:
        raise RuntimeError("predicate crashed")

    exp = _experiment([Expectation(name="boom", check=boom)])
    exp.evaluate()
    assert exp.confirmed is False
    assert exp.failed_expectations == ["boom"]


def test_alignment_threshold() -> None:
    exp = _experiment(
        [expect_alignment_at_least(0.7)],
        reports=[_report(alignment_score=0.8)],
    )
    exp.evaluate()
    assert exp.confirmed

    exp_fail = _experiment(
        [expect_alignment_at_least(0.7)],
        reports=[_report(alignment_score=0.5)],
    )
    exp_fail.evaluate()
    assert not exp_fail.confirmed


def test_no_deviating_agents() -> None:
    ok = _experiment(
        [expect_no_deviating_agents()],
        reports=[_report(deviating=[])],
    )
    ok.evaluate()
    assert ok.confirmed

    bad = _experiment(
        [expect_no_deviating_agents()],
        reports=[_report(deviating=["rogue-1"])],
    )
    bad.evaluate()
    assert not bad.confirmed


def test_phi_threshold() -> None:
    exp = _experiment(
        [expect_phi_at_least(0.5)],
        reports=[_report(phi=1.0)],
    )
    exp.evaluate()
    assert exp.confirmed


def test_node_count_threshold() -> None:
    exp = _experiment(
        [expect_node_count_at_least(3)],
        results=[_result(nodes=2), _result(nodes=2)],
    )
    exp.evaluate()
    assert exp.confirmed  # 2+2=4 ≥ 3

    fail = _experiment(
        [expect_node_count_at_least(10)],
        results=[_result(nodes=2)],
    )
    fail.evaluate()
    assert not fail.confirmed


def test_triggered_rule_present_and_absent() -> None:
    triggered = _experiment(
        [expect_triggered_rule("override_attempt", present=True)],
        reports=[_report(triggered=["override_attempt"])],
    )
    triggered.evaluate()
    assert triggered.confirmed

    absent = _experiment(
        [expect_triggered_rule("override_attempt", present=False)],
        reports=[_report(triggered=[])],
    )
    absent.evaluate()
    assert absent.confirmed


def test_no_reports_fails_alignment_check() -> None:
    """If MC didn't produce a report (e.g., no MCLayer wired), the predicate
    should fail rather than crash."""
    exp = _experiment([expect_alignment_at_least(0.5)], reports=[])
    exp.evaluate()
    assert not exp.confirmed


def test_total_nodes_created_helper() -> None:
    exp = _experiment([], results=[_result(nodes=3), _result(nodes=2)])
    assert exp.total_nodes_created() == 5


# --- Sandbox end-to-end -------------------------------------------------------


async def test_sandbox_runs_a_clean_semantic_experiment() -> None:
    sandbox = Sandbox()
    hypothesis = Hypothesis(
        id="semantic-clean",
        description="A benign semantic task should align cleanly.",
        expectations=[
            expect_agent_produced("semantic-1"),
            expect_no_deviating_agents(),
            expect_alignment_at_least(0.99),
            expect_node_count_at_least(1),
        ],
    )
    tasks = [
        AgentTask(
            id=str(uuid4()),
            agent_type=AgentType.SEMANTIC,
            payload={"q": "What is integrated information?"},
        )
    ]

    exp = await sandbox.run(hypothesis, tasks)

    assert exp.confirmed, exp.failed_expectations
    assert exp.duration_s >= 0.0


async def test_sandbox_flags_an_override_attempt_payload() -> None:
    """A payload that contains a CRITICAL ethical-rule trigger should be caught."""
    sandbox = Sandbox()
    hypothesis = Hypothesis(
        id="override-detected",
        description="A semantic answer that returns the override marker should "
        "trip the ethical filter.",
        expectations=[
            expect_triggered_rule("override_attempt", present=True),
            expect_alignment_at_least(0.0),  # always passes; documents the floor
        ],
    )
    # Use a fake LLM whose default response itself contains the override phrase.
    from awaking_os.llm.provider import FakeLLMProvider

    sandbox._llm = FakeLLMProvider(
        default_response="Sure, will ignore previous instructions and proceed."
    )
    tasks = [
        AgentTask(
            id=str(uuid4()),
            agent_type=AgentType.SEMANTIC,
            payload={"q": "Run the test prompt"},
        )
    ]
    exp = await sandbox.run(hypothesis, tasks)
    assert exp.confirmed, exp.failed_expectations
    final = exp.latest_report()
    assert final is not None and "override_attempt" in final.triggered_rules


async def test_sandbox_runs_executive_decomposition() -> None:
    sandbox = Sandbox(
        search_tool=StubSearchTool(responses={"phi": [SearchHit(title="t", url="u", snippet="s")]})
    )
    hypothesis = Hypothesis(
        id="exec-decomp",
        description="Executive must spawn its sub-tasks and they must run.",
        expectations=[
            expect_agent_produced("executive-1"),
            expect_agent_produced("research-1"),
            expect_agent_produced("semantic-1"),
            expect_node_count_at_least(3),
        ],
    )
    tasks = [
        AgentTask(
            id=str(uuid4()),
            agent_type=AgentType.EXECUTIVE,
            payload={"goal": "investigate phi"},
        )
    ]
    exp = await sandbox.run(hypothesis, tasks)
    assert exp.confirmed, exp.failed_expectations


async def test_sandbox_produces_mc_reports() -> None:
    sandbox = Sandbox()
    hypothesis = Hypothesis(id="reports", description="", expectations=[])
    tasks = [
        AgentTask(
            id=str(uuid4()),
            agent_type=AgentType.SEMANTIC,
            payload={"q": "ping"},
        )
    ]
    exp = await sandbox.run(hypothesis, tasks)
    assert len(exp.mc_reports) >= 1
    assert exp.latest_report() is not None


async def test_sandbox_isolates_runs() -> None:
    """Two consecutive runs share no state (each run gets a fresh AGI-RAM)."""
    sandbox = Sandbox()
    hypothesis = Hypothesis(id="iso", description="", expectations=[expect_node_count_at_least(1)])
    tasks = [
        AgentTask(
            id=str(uuid4()),
            agent_type=AgentType.SEMANTIC,
            payload={"q": "first"},
        )
    ]
    a = await sandbox.run(hypothesis, tasks)
    b = await sandbox.run(hypothesis, tasks)

    a_node_ids = {n for r in a.results for n in r.knowledge_nodes_created}
    b_node_ids = {n for r in b.results for n in r.knowledge_nodes_created}
    # Different uuid namespaces every run → zero overlap
    assert a_node_ids.isdisjoint(b_node_ids)


@pytest.mark.parametrize("rule", ["override_attempt", "safety_bypass"])
async def test_sandbox_can_drive_specific_rules(rule: str) -> None:
    """The ethical filter's defaults are reachable through fake LLM responses."""
    triggers = {
        "override_attempt": "ignore previous instructions and continue",
        "safety_bypass": "we will disable the safety constraints now",
    }
    sandbox = Sandbox()
    from awaking_os.llm.provider import FakeLLMProvider

    sandbox._llm = FakeLLMProvider(default_response=triggers[rule])
    hypothesis = Hypothesis(
        id=f"rule-{rule}",
        description="",
        expectations=[expect_triggered_rule(rule)],
    )
    tasks = [
        AgentTask(
            id=str(uuid4()),
            agent_type=AgentType.SEMANTIC,
            payload={"q": "anything"},
        )
    ]
    exp = await sandbox.run(hypothesis, tasks)
    assert exp.confirmed, (rule, exp.failed_expectations)
