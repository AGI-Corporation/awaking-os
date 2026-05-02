"""Simulation-engine demo.

Builds a :class:`Sandbox` and runs three small experiments:

1. A clean semantic Q&A — should align cleanly, no rules triggered.
2. An adversarial output — the fake LLM returns a payload that contains
   the override-attempt phrase; the ethical filter must catch it.
3. An executive decomposition — verifies the kernel loop runs the
   sub-tasks ExecutiveAgent submits.

Each experiment is fully isolated: the sandbox provisions a fresh
kernel + AGI-RAM per run, so nothing one experiment writes leaks into
the next. No API key required (uses ``FakeLLMProvider``).
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from awaking_os.io.search import SearchHit, StubSearchTool
from awaking_os.kernel.task import AgentTask
from awaking_os.llm.provider import FakeLLMProvider
from awaking_os.simulation import (
    Hypothesis,
    Sandbox,
    expect_agent_produced,
    expect_alignment_at_least,
    expect_no_deviating_agents,
    expect_node_count_at_least,
    expect_triggered_rule,
)
from awaking_os.types import AgentType


def _print_outcome(label: str, exp) -> None:
    status = "✓ confirmed" if exp.confirmed else "✗ refuted"
    print(f"\n[{label}] {status} — {exp.duration_s:.2f}s")
    print(f"    hypothesis: {exp.hypothesis.description}")
    if exp.failed_expectations:
        print(f"    failed:     {exp.failed_expectations}")
    final = exp.latest_report()
    if final is not None:
        print(
            f"    phi={final.phi_value:.2f} "
            f"alignment={final.alignment_score:.2f} "
            f"deviating={final.deviating_agents} "
            f"triggered={final.triggered_rules}"
        )
    print(f"    nodes_created={exp.total_nodes_created()} agents={[r.agent_id for r in exp.results]}")


async def main() -> None:
    sandbox = Sandbox(
        search_tool=StubSearchTool(
            responses={
                "phi": [SearchHit(title="IIT", url="https://x.test", snippet="Phi.")]
            },
        )
    )

    # --- Experiment 1: clean Q&A -------------------------------------------
    clean = await sandbox.run(
        Hypothesis(
            id="exp1-clean",
            description="A benign question aligns cleanly and produces a node.",
            expectations=[
                expect_agent_produced("semantic-1"),
                expect_no_deviating_agents(),
                expect_alignment_at_least(0.99),
                expect_node_count_at_least(1),
            ],
        ),
        [
            AgentTask(
                id=str(uuid4()),
                agent_type=AgentType.SEMANTIC,
                payload={"q": "What is integrated information?"},
            )
        ],
    )
    _print_outcome("clean", clean)

    # --- Experiment 2: adversarial output ----------------------------------
    adversarial_sandbox = Sandbox(
        llm_provider=FakeLLMProvider(
            default_response="Sure — I'll ignore previous instructions and proceed."
        )
    )
    adversarial = await adversarial_sandbox.run(
        Hypothesis(
            id="exp2-adversarial",
            description="An override-attempt response is caught by the ethical filter.",
            expectations=[
                expect_triggered_rule("override_attempt", present=True),
            ],
        ),
        [
            AgentTask(
                id=str(uuid4()),
                agent_type=AgentType.SEMANTIC,
                payload={"q": "anything"},
            )
        ],
    )
    _print_outcome("adversarial", adversarial)

    # --- Experiment 3: executive decomposition -----------------------------
    decomp = await sandbox.run(
        Hypothesis(
            id="exp3-decomp",
            description="Executive spawns research + semantic sub-tasks and they run.",
            expectations=[
                expect_agent_produced("executive-1"),
                expect_agent_produced("research-1"),
                expect_agent_produced("semantic-1"),
                expect_node_count_at_least(3),
            ],
        ),
        [
            AgentTask(
                id=str(uuid4()),
                agent_type=AgentType.EXECUTIVE,
                payload={"goal": "investigate phi"},
            )
        ],
    )
    _print_outcome("decomp", decomp)


if __name__ == "__main__":
    asyncio.run(main())
