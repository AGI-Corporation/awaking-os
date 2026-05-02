"""Phi (Φ) calculator — spectral-entropy proxy for IIT.

Caveat: this is *not* the full IIT Phi (which is exponential in the
number of nodes via PyPhi). We compute the Shannon entropy of the
spectral distribution of the symmetric part of the integration matrix,
in bits, as a tractable proxy for "how integrated is the system".

- An empty or single-node matrix → 0.0 (no integration possible).
- A fully decoupled matrix (only diagonal) → low entropy.
- A matrix with many nonzero entries spread evenly → high entropy.

The number is a real-valued metric, not a probability, so it has no
upper bound from this calculator's perspective. Callers should treat
it as ordinal: bigger means more integrated.
"""

from __future__ import annotations

import numpy as np


class PhiCalculator:
    def calculate(self, matrix: list[list[float]]) -> float:
        if not matrix or len(matrix) < 2:
            return 0.0

        m = np.asarray(matrix, dtype=float)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("integration_matrix must be square 2D")
        if m.shape[0] < 2:
            return 0.0

        # Symmetric part — the bidirectional integration
        sym = (m + m.T) / 2.0
        eigvals = np.linalg.eigvalsh(sym)
        absvals = np.abs(eigvals)
        total = absvals.sum()
        if total == 0.0:
            return 0.0

        probs = absvals / total
        nonzero = probs[probs > 0.0]
        # Shannon entropy in bits
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        return entropy
