"""Simulation Engine: isolated sandboxes for hypothesis-testing experiments."""

from awaking_os.simulation.hypothesis import (
    Expectation,
    Experiment,
    Hypothesis,
    expect_agent_produced,
    expect_alignment_at_least,
    expect_no_deviating_agents,
    expect_node_count_at_least,
    expect_phi_at_least,
    expect_triggered_rule,
)
from awaking_os.simulation.sandbox import Sandbox

__all__ = [
    "Expectation",
    "Experiment",
    "Hypothesis",
    "Sandbox",
    "expect_agent_produced",
    "expect_alignment_at_least",
    "expect_no_deviating_agents",
    "expect_node_count_at_least",
    "expect_phi_at_least",
    "expect_triggered_rule",
]
