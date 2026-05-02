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


# --- spectral / k-mer feature surfaces ---------------------------------------


async def test_cetacean_summary_surfaces_dominant_frequency(
    biotic_agent: BioticAgent,
) -> None:
    # MockBioSignalStream cetacean is a 20 Hz sine at 1 kHz → dominant ≈ 20 Hz.
    result = await biotic_agent.execute(_ctx(_task({"signal_type": "cetacean", "samples": 1024})))
    summary = result.output["summary"]
    assert "dominant_freq_hz" in summary
    assert abs(summary["dominant_freq_hz"] - 20.0) < 2.0
    assert summary["spectral_entropy_bits"] >= 0.0


async def test_eeg_summary_includes_band_powers(biotic_agent: BioticAgent) -> None:
    # EEG mock is 10 Hz alpha + 20 Hz beta at 256 Hz. Both bands should hold mass.
    result = await biotic_agent.execute(_ctx(_task({"signal_type": "eeg", "samples": 512})))
    summary = result.output["summary"]
    assert "band_powers" in summary
    assert summary["band_powers"]["alpha"] > 0.05
    # And the EEG-specific bands are surfaced (not the cetacean ones).
    assert "delta" in summary["band_powers"]
    assert "low" not in summary["band_powers"]


async def test_genomic_summary_surfaces_dimer_features(biotic_agent: BioticAgent) -> None:
    result = await biotic_agent.execute(_ctx(_task({"signal_type": "genomic", "samples": 200})))
    summary = result.output["summary"]
    assert "dimer_counts" in summary
    assert "dimer_entropy_bits" in summary
    # Dimer counts should sum to N-1 (one fewer transition than samples).
    assert sum(summary["dimer_counts"].values()) == 199
    # Entropy bounded by log2(16) = 4 bits for a 4-letter alphabet.
    assert 0.0 <= summary["dimer_entropy_bits"] <= 4.0 + 1e-9


async def test_cetacean_band_powers_use_cetacean_bands(biotic_agent: BioticAgent) -> None:
    result = await biotic_agent.execute(_ctx(_task({"signal_type": "cetacean", "samples": 256})))
    summary = result.output["summary"]
    # 20 Hz dominant → mass in the "infrasonic" or "low" cetacean band.
    assert "low" in summary["band_powers"] or "infrasonic" in summary["band_powers"]
    assert "alpha" not in summary["band_powers"]  # EEG bands shouldn't appear here
