"""Mock bio-signal streams for the BioticAgent.

Synthetic, deterministic streams keyed by signal type. Cetacean and EEG
emit floats; genomic emits ACGT bases. Real hardware integration is out
of scope.
"""

from __future__ import annotations

import math
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum


class SignalType(StrEnum):
    CETACEAN = "cetacean"
    GENOMIC = "genomic"
    EEG = "eeg"


@dataclass(frozen=True)
class BioSignalSample:
    signal_type: SignalType
    index: int
    value: float | str
    metadata: dict[str, str | float | int]


class MockBioSignalStream:
    """Deterministic synthetic bio-signal stream.

    - ``cetacean``: sine wave at ~20 Hz with low-amplitude noise
    - ``eeg``: sum of alpha (10 Hz) and beta (20 Hz) components plus noise
    - ``genomic``: ACGT bases drawn with a fixed-seed RNG
    """

    def __init__(self, signal_type: SignalType, seed: int = 42) -> None:
        self.signal_type = signal_type
        self._rng = random.Random(seed)

    async def stream(self, count: int) -> AsyncIterator[BioSignalSample]:
        if count < 0:
            raise ValueError("count must be non-negative")
        for i in range(count):
            yield self._sample(i)

    def _sample(self, i: int) -> BioSignalSample:
        if self.signal_type == SignalType.CETACEAN:
            # 20 Hz sine sampled at 1 kHz, plus 0.05 noise
            t = i / 1000.0
            value = math.sin(2 * math.pi * 20.0 * t) + self._rng.uniform(-0.05, 0.05)
            return BioSignalSample(
                signal_type=self.signal_type,
                index=i,
                value=value,
                metadata={"unit": "normalized", "sample_rate_hz": 1000.0},
            )
        if self.signal_type == SignalType.EEG:
            # Sum of alpha (10 Hz) and beta (20 Hz)
            t = i / 256.0
            alpha = math.sin(2 * math.pi * 10.0 * t)
            beta = 0.5 * math.sin(2 * math.pi * 20.0 * t)
            noise = self._rng.uniform(-0.1, 0.1)
            return BioSignalSample(
                signal_type=self.signal_type,
                index=i,
                value=alpha + beta + noise,
                metadata={"unit": "uV", "sample_rate_hz": 256.0},
            )
        if self.signal_type == SignalType.GENOMIC:
            base = self._rng.choice(["A", "C", "G", "T"])
            return BioSignalSample(
                signal_type=self.signal_type,
                index=i,
                value=base,
                metadata={"alphabet": "ACGT"},
            )
        raise ValueError(f"Unknown signal type: {self.signal_type}")
