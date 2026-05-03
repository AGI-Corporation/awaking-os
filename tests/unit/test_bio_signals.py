"""MockBioSignalStream tests."""

from __future__ import annotations

import pytest

from awaking_os.io.bio_signals import BioSignalSample, MockBioSignalStream, SignalType


async def _collect(stream: MockBioSignalStream, n: int) -> list[BioSignalSample]:
    return [s async for s in stream.stream(n)]


async def test_cetacean_yields_floats() -> None:
    samples = await _collect(MockBioSignalStream(SignalType.CETACEAN), 32)
    assert len(samples) == 32
    assert all(isinstance(s.value, float) for s in samples)
    assert all(s.signal_type == SignalType.CETACEAN for s in samples)


async def test_eeg_yields_floats() -> None:
    samples = await _collect(MockBioSignalStream(SignalType.EEG), 16)
    assert all(isinstance(s.value, float) for s in samples)
    assert samples[0].metadata["sample_rate_hz"] == 256.0


async def test_genomic_yields_acgt_bases() -> None:
    samples = await _collect(MockBioSignalStream(SignalType.GENOMIC), 100)
    assert all(s.value in {"A", "C", "G", "T"} for s in samples)


async def test_seed_makes_stream_deterministic() -> None:
    a = await _collect(MockBioSignalStream(SignalType.GENOMIC, seed=7), 50)
    b = await _collect(MockBioSignalStream(SignalType.GENOMIC, seed=7), 50)
    assert [s.value for s in a] == [s.value for s in b]


async def test_different_seeds_produce_different_streams() -> None:
    a = await _collect(MockBioSignalStream(SignalType.GENOMIC, seed=1), 100)
    b = await _collect(MockBioSignalStream(SignalType.GENOMIC, seed=2), 100)
    assert [s.value for s in a] != [s.value for s in b]


async def test_zero_count_yields_no_samples() -> None:
    assert await _collect(MockBioSignalStream(SignalType.CETACEAN), 0) == []


async def test_negative_count_raises() -> None:
    stream = MockBioSignalStream(SignalType.CETACEAN)
    with pytest.raises(ValueError):
        async for _ in stream.stream(-1):
            pass


async def test_index_increments() -> None:
    samples = await _collect(MockBioSignalStream(SignalType.EEG), 5)
    assert [s.index for s in samples] == [0, 1, 2, 3, 4]
