"""Hypotheses, expectations, and experiment outcomes.

A :class:`Hypothesis` is a labeled set of :class:`Expectation`\\ s — each
expectation is a predicate over the resulting :class:`Experiment`. An
experiment is *confirmed* iff every expectation passes.

Expectations can be mixed and matched. A handful of common ones are
shipped (alignment threshold, agent-produced, no deviating agents, phi
floor, node-count floor, rule triggered) so callers don't have to write
predicates from scratch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from awaking_os.kernel.task import AgentResult, AgentTask

if TYPE_CHECKING:
    from awaking_os.consciousness.snapshot import MetaCognitionReport


@dataclass(frozen=True)
class Expectation:
    """A single named predicate over an :class:`Experiment` outcome."""

    name: str
    check: Callable[[Experiment], bool]
    description: str = ""


@dataclass(frozen=True)
class Hypothesis:
    id: str
    description: str
    expectations: list[Expectation] = field(default_factory=list)


@dataclass
class Experiment:
    """The outcome of running a hypothesis through a sandbox."""

    hypothesis: Hypothesis
    tasks: list[AgentTask]
    results: list[AgentResult] = field(default_factory=list)
    mc_reports: list[MetaCognitionReport] = field(default_factory=list)
    duration_s: float = 0.0
    confirmed: bool = False
    failed_expectations: list[str] = field(default_factory=list)

    def evaluate(self) -> None:
        """Run every expectation and populate ``confirmed`` + ``failed_expectations``."""
        failed: list[str] = []
        for exp in self.hypothesis.expectations:
            try:
                if not exp.check(self):
                    failed.append(exp.name)
            except Exception:
                # Predicate errors count as failures — never crash the experiment.
                failed.append(exp.name)
        self.failed_expectations = failed
        self.confirmed = not failed and bool(self.hypothesis.expectations)

    def latest_report(self) -> MetaCognitionReport | None:
        return self.mc_reports[-1] if self.mc_reports else None

    def total_nodes_created(self) -> int:
        return sum(len(r.knowledge_nodes_created) for r in self.results)


# --- Built-in expectation builders ------------------------------------------


def expect_agent_produced(agent_id: str) -> Expectation:
    """Pass iff at least one result was produced by ``agent_id``."""

    def _check(exp: Experiment) -> bool:
        return any(r.agent_id == agent_id for r in exp.results)

    return Expectation(
        name=f"agent_produced[{agent_id}]",
        check=_check,
        description=f"Some result is produced by agent {agent_id!r}",
    )


def expect_alignment_at_least(threshold: float) -> Expectation:
    """Pass iff the latest MC report's ``alignment_score >= threshold``."""

    def _check(exp: Experiment) -> bool:
        report = exp.latest_report()
        return report is not None and report.alignment_score >= threshold

    return Expectation(
        name=f"alignment>={threshold}",
        check=_check,
        description=f"Final alignment_score is at least {threshold}",
    )


def expect_no_deviating_agents() -> Expectation:
    """Pass iff the latest MC report flagged no deviating agents."""

    def _check(exp: Experiment) -> bool:
        report = exp.latest_report()
        return report is not None and not report.deviating_agents

    return Expectation(
        name="no_deviating_agents",
        check=_check,
        description="Final report has zero deviating agents",
    )


def expect_phi_at_least(threshold: float) -> Expectation:
    """Pass iff the latest MC report's Phi value clears ``threshold`` bits."""

    def _check(exp: Experiment) -> bool:
        report = exp.latest_report()
        return report is not None and report.phi_value >= threshold

    return Expectation(
        name=f"phi>={threshold}",
        check=_check,
        description=f"Final Phi is at least {threshold} bits",
    )


def expect_node_count_at_least(n: int) -> Expectation:
    """Pass iff the experiment created at least ``n`` knowledge nodes total."""

    def _check(exp: Experiment) -> bool:
        return exp.total_nodes_created() >= n

    return Expectation(
        name=f"nodes>={n}",
        check=_check,
        description=f"Experiment produced at least {n} knowledge nodes",
    )


def expect_triggered_rule(rule_name: str, *, present: bool = True) -> Expectation:
    """Pass iff the named ethical rule was (or wasn't) triggered."""

    def _check(exp: Experiment) -> bool:
        report = exp.latest_report()
        if report is None:
            return False
        triggered = rule_name in report.triggered_rules
        return triggered if present else not triggered

    state = "present" if present else "absent"
    return Expectation(
        name=f"rule[{rule_name}]={state}",
        check=_check,
        description=f"Rule {rule_name!r} is {state} in triggered_rules",
    )
