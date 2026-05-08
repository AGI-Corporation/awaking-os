"""Bio-signal feature extraction.

Time-series features (cetacean / EEG):
- ``dominant_freq_hz`` — peak frequency from the real-valued FFT
- ``spectral_entropy_bits`` — Shannon entropy of the normalized power
  spectrum (in bits); flat spectrum → high entropy, single-tone → ~0
- ``band_powers`` — fraction of total power in named frequency bands

Sequence features (genomic):
- ``dimer_counts`` — counts of every observed 2-mer
- ``dimer_entropy_bits`` — Shannon entropy of the 2-mer distribution;
  uniform mixing → 4 bits, all one dimer → 0
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

# Standard EEG frequency bands (Hz)
EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

# Cetacean acoustic bands (Hz). Rough, not biologically authoritative; useful
# for showing the feature-extraction shape end-to-end.
CETACEAN_BANDS: dict[str, tuple[float, float]] = {
    "infrasonic": (1.0, 20.0),
    "low": (20.0, 200.0),
    "mid": (200.0, 2000.0),
    "high": (2000.0, 20000.0),
}


def time_series_features(
    values: list[float],
    sample_rate_hz: float,
    bands: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float | dict[str, float]]:
    """Spectral features of a 1-D real-valued time series.

    Returns ``{}`` when there's not enough signal to compute anything
    meaningful (n < 2 or zero total power).
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        return {}

    n = arr.size
    spectrum = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    power = spectrum**2
    total = float(power.sum())
    if total == 0.0:
        return {
            "dominant_freq_hz": 0.0,
            "spectral_entropy_bits": 0.0,
            "band_powers": {name: 0.0 for name in (bands or {})},
        }

    peak_idx = int(np.argmax(power))
    dominant_hz = float(freqs[peak_idx])

    power_norm = power / total
    nonzero = power_norm[power_norm > 0]
    spectral_entropy = float(-(nonzero * np.log2(nonzero)).sum())

    band_powers: dict[str, float] = {}
    for name, (low, high) in (bands or {}).items():
        mask = (freqs >= low) & (freqs < high)
        band_powers[name] = float(power_norm[mask].sum())

    return {
        "dominant_freq_hz": dominant_hz,
        "spectral_entropy_bits": spectral_entropy,
        "band_powers": band_powers,
    }


def sequence_features(bases: list[str]) -> dict[str, float | dict[str, int]]:
    """k-mer features for a DNA-like alphabet (A/C/G/T)."""
    if len(bases) < 2:
        return {}

    dimer_counts: dict[str, int] = {}
    for i in range(len(bases) - 1):
        dimer = f"{bases[i]}{bases[i + 1]}"
        dimer_counts[dimer] = dimer_counts.get(dimer, 0) + 1

    total = sum(dimer_counts.values())
    if total == 0:
        return {"dimer_counts": {}, "dimer_entropy_bits": 0.0}

    entropy = 0.0
    for count in dimer_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return {
        "dimer_counts": dict(sorted(dimer_counts.items())),
        "dimer_entropy_bits": entropy,
    }
