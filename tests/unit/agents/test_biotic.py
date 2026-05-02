"""BioticAgent tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from awaking_os.agents.biotic import BioticAgent
from awaking_os.kernel.task import AgentContext, AgentTask
from awaking_os.memory.agi_ram import AGIRam
from awaking_os.types import AgentType


def _task(payload: dict | None = None) -> AgentTask:
    return AgentTask(id=str(uuid4()), agent_type=AgentType.BIOTIC, payload=payload or {})


def _ctx(task: AgentTask) -> AgentContext:
    return AgentContext(task=task, memory=[], ethical_boundary=[])


async def test_default_signal_is_cetacean(biotic_agent: BioticAgent) -> None:
    result = await biotic_agent.execute(_ctx(_task()))
    assert result.output["signal_type"] == "cetacean"
    assert result.output["samples"] == BioticAgent.DEFAULT_SAMPLES


async def test_cetacean_summary_has_stats(biotic_agent: BioticAgent) -> None:
    result = await biotic_agent.execute(_ctx(_task({"signal_type": "cetacean", "samples": 64})))
    summary = result.output["summary"]
    for key in ("mean", "std", "min", "max"):
        assert key in summary
    assert summary["min"] <= summary["max"]


async def test_eeg_summary_has_stats(biotic_agent: BioticAgent) -> None:
    result = await biotic_agent.execute(_ctx(_task({"signal_type": "eeg", "samples": 32})))
    summary = result.output["summary"]
    assert summary["mean"] is not None
    assert summary["std"] >= 0


async def test_genomic_summary_has_gc_and_counts(biotic_agent: BioticAgent) -> None:
    result = await biotic_agent.execute(_ctx(_task({"signal_type": "genomic", "samples": 200})))
    summary = result.output["summary"]
    assert 0.0 <= summary["gc_content"] <= 1.0
    assert set(summary["counts"]).issubset({"A", "C", "G", "T"})
    assert sum(summary["counts"].values()) == 200


async def test_persists_knowledge_node(biotic_agent: BioticAgent, semantic_agi_ram: AGIRam) -> None:
    result = await biotic_agent.execute(_ctx(_task({"signal_type": "cetacean", "samples": 16})))
    assert len(result.knowledge_nodes_created) == 1
    node = await semantic_agi_ram.get(result.knowledge_nodes_created[0])
    assert node is not None
    assert node.type == "event"
    assert node.metadata["signal_type"] == "cetacean"
    assert node.metadata["samples"] == 16


async def test_caps_samples_at_max(biotic_agent: BioticAgent) -> None:
    result = await biotic_agent.execute(_ctx(_task({"samples": 999_999})))
    assert result.output["samples"] == BioticAgent.MAX_SAMPLES


async def test_unknown_signal_type_raises(biotic_agent: BioticAgent) -> None:
    with pytest.raises(ValueError):
        await biotic_agent.execute(_ctx(_task({"signal_type": "telepathy"})))


async def test_seeded_runs_are_deterministic(
    biotic_agent: BioticAgent,
) -> None:
    a = await biotic_agent.execute(
        _ctx(_task({"signal_type": "genomic", "samples": 50, "seed": 42}))
    )
    b = await biotic_agent.execute(
        _ctx(_task({"signal_type": "genomic", "samples": 50, "seed": 42}))
    )
    assert a.output["summary"]["counts"] == b.output["summary"]["counts"]
