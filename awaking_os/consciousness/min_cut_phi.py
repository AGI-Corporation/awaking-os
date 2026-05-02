"""Exact min-cut Phi for small networks.

PyPhi (the canonical IIT library) is unavailable on Python 3.10+ —
its only pinned PyPI release does ``from collections import Iterable``,
which was removed from the ``collections`` module in 3.10. So instead
of a PyPhi shim, this module ships a tractable, exact, IIT-inspired
calculation that we control end-to-end:

For an n×n influence matrix, brute-force every non-trivial bipartition
of the n nodes and compute the **minimum cross-cut weight, normalized
by the size of the smaller part**. This is a real graph-theoretic
quantity that captures "how integrated is this system": high when the
network can't be cleanly partitioned, low when it can.

It is *not* full IIT 3.0 Phi (that needs a TPM and the MIP over
concept distributions, which is exponential in N). But it is a
defensible, deterministic, parameter-free Phi proxy on the same
input shape (an integration matrix) as :class:`PhiCalculator`.

The brute-force is feasible up to ``max_n``; beyond that, the
calculator falls back to the spectral entropy in :class:`PhiCalculator`
so callers don't have to switch implementations as the network grows.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from awaking_os.consciousness.phi_calculator import PhiCalculator


class MinCutPhiCalculator:
    """Exact min-cut Phi for n ≤ ``max_n``; spectral fallback above.

    The ``calculate`` interface matches :class:`PhiCalculator` so this
    can be passed to :class:`MCLayer` interchangeably.
    """

    def __init__(self, max_n: int = 6) -> None:
        if max_n < 2:
            raise ValueError("max_n must be at least 2")
        self.max_n = max_n
        self._fallback = PhiCalculator()

    def calculate(self, matrix: list[list[float]]) -> float:
        if not matrix:
            return 0.0
        m = np.asarray(matrix, dtype=float)
        if m.size == 0:
            return 0.0
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("integration_matrix must be square 2D")
        n = m.shape[0]
        if n < 2:
            return 0.0
        if n > self.max_n:
            return self._fallback.calculate(matrix)

        # Symmetric weight (bidirectional integration). Negative entries are
        # discarded — Phi is a "how much do they affect each other" measure,
        # not a signed correlation.
        sym = np.abs(m + m.T) / 2.0
        if sym.sum() == 0.0:
            return 0.0

        # Try every non-trivial bipartition (k ranges 1..n//2 to avoid
        # double-counting since {A, complement(A)} is the same cut).
        nodes = list(range(n))
        best = float("inf")
        for k in range(1, n // 2 + 1):
            for combo in combinations(nodes, k):
                a = list(combo)
                b = [i for i in nodes if i not in combo]
                # Cross-cut weight: edges from A to B + B to A.
                # In the symmetrized matrix this is just one term.
                cross = float(sym[np.ix_(a, b)].sum())
                # Normalize by the size of the smaller part — a cut that
                # peels off a single node is "easy" and shouldn't dominate
                # if there's a more balanced partition with similar weight.
                normalized = cross / min(len(a), len(b))
                if normalized < best:
                    best = normalized
                if best == 0.0:
                    return 0.0  # disconnected — no need to continue
        return best
