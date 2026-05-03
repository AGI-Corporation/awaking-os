"""Bio-signal feature-extraction tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from awaking_os.io.bio_features import (
    CETACEAN_BANDS,
    EEG_BANDS,
    sequence_features,
    time_series_features,
)


def _sine(freq_hz: float, sample_rate_hz: float, n: int, amplitude: float = 1.0) -> list[float]:
    t = np.arange(n) / sample_rate_hz
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).tolist()


# --- time_series_features ----------------------------------------------------


def test_pure_sine_dominant_frequency_matches_input() -> None:
    """A 50 Hz sine sampled at 1 kHz must come back with dominant ~ 50 Hz."""
    signal = _sine(50.0, 1000.0, 1024)
    f = time_series_features(signal, sample_rate_hz=1000.0)
    # FFT bin width is 1000/1024 ≈ 0.98 Hz; allow ±2 bins.
    assert abs(f["dominant_freq_hz"] - 50.0) < 2.0


def test_low_entropy_for_pure_tone() -> None:
    """A single-frequency signal should have a peaky spectrum → low entropy."""
    signal = _sine(20.0, 1000.0, 1024)
    f = time_series_features(signal, sample_rate_hz=1000.0)
    # Numerical leakage from the rectangular window broadens the peak slightly,
    # but it stays well below the broadband-noise reference (~7 bits below).
    assert f["spectral_entropy_bits"] < 5.0


def test_higher_entropy_for_white_noise() -> None:
    rng = np.random.default_rng(42)
    noise = rng.normal(size=1024).tolist()
    f = time_series_features(noise, sample_rate_hz=1000.0)
    # White noise spreads power across the band → high entropy.
    assert f["spectral_entropy_bits"] > 7.0


def test_band_powers_localize_correctly() -> None:
    """A 10 Hz signal sampled at 256 Hz should put its mass in the 'alpha' EEG band."""
    signal = _sine(10.0, 256.0, 512)
    f = time_series_features(signal, sample_rate_hz=256.0, bands=EEG_BANDS)
    band_powers = f["band_powers"]
    assert band_powers["alpha"] > 0.9
    assert band_powers["delta"] < 0.05


def test_band_powers_sum_to_at_most_one() -> None:
    rng = np.random.default_rng(7)
    signal = rng.normal(size=512).tolist()
    f = time_series_features(signal, sample_rate_hz=256.0, bands=EEG_BANDS)
    total = sum(f["band_powers"].values())
    # Disjoint bands; total ≤ 1 with the rest above the gamma cutoff.
    assert 0.0 <= total <= 1.0 + 1e-9


def test_short_input_returns_empty() -> None:
    assert time_series_features([], 1000.0) == {}
    assert time_series_features([1.0], 1000.0) == {}


def test_zero_input_returns_zero_features() -> None:
    f = time_series_features([0.0] * 16, sample_rate_hz=1000.0)
    assert f["dominant_freq_hz"] == 0.0
    assert f["spectral_entropy_bits"] == 0.0


def test_invalid_sample_rate_raises() -> None:
    with pytest.raises(ValueError):
        time_series_features([1.0, 2.0, 3.0], sample_rate_hz=0.0)


def test_cetacean_bands_present() -> None:
    f = time_series_features(_sine(100.0, 1000.0, 512), 1000.0, bands=CETACEAN_BANDS)
    assert "low" in f["band_powers"]


# --- sequence_features --------------------------------------------------------


def test_dimer_counts_for_short_sequence() -> None:
    # ACGT has 3 dimers: AC, CG, GT
    f = sequence_features(["A", "C", "G", "T"])
    assert f["dimer_counts"] == {"AC": 1, "CG": 1, "GT": 1}


def test_repeating_sequence_has_zero_entropy() -> None:
    bases = ["A", "A"] * 50  # only "AA" dimers
    f = sequence_features(bases)
    assert f["dimer_counts"] == {"AA": 99}
    assert f["dimer_entropy_bits"] == pytest.approx(0.0, abs=1e-9)


def test_balanced_dimer_distribution_has_high_entropy() -> None:
    # Two distinct dimers in equal counts → entropy = log2(2) = 1 bit.
    bases = list("ACAC" * 25)  # dimers: AC, CA alternating
    f = sequence_features(bases)
    # Counts: AC ≈ 50, CA ≈ 49 (one fewer at the tail) → entropy ≈ 1.0 bit.
    assert f["dimer_entropy_bits"] == pytest.approx(1.0, abs=0.05)


def test_short_sequence_returns_empty() -> None:
    assert sequence_features([]) == {}
    assert sequence_features(["A"]) == {}


def test_dimer_entropy_in_valid_range() -> None:
    bases = list("ACGTACGTACGTACGT")
    f = sequence_features(bases)
    # 4 distinct dimers (AC, CG, GT, TA) with counts (4,4,4,3) → ~2 bits.
    assert 0.0 <= f["dimer_entropy_bits"] <= math.log2(16) + 1e-9
