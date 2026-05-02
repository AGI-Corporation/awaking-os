"""MinCutPhiCalculator tests."""

from __future__ import annotations

import pytest

from awaking_os.consciousness.min_cut_phi import MinCutPhiCalculator
from awaking_os.consciousness.phi_calculator import PhiCalculator


@pytest.fixture
def calc() -> MinCutPhiCalculator:
    return MinCutPhiCalculator()


def test_empty_matrix_returns_zero(calc: MinCutPhiCalculator) -> None:
    assert calc.calculate([]) == 0.0


def test_single_node_returns_zero(calc: MinCutPhiCalculator) -> None:
    assert calc.calculate([[1.0]]) == 0.0


def test_zero_matrix_returns_zero(calc: MinCutPhiCalculator) -> None:
    assert calc.calculate([[0.0, 0.0], [0.0, 0.0]]) == 0.0


def test_disconnected_components_phi_is_zero(calc: MinCutPhiCalculator) -> None:
    """Two clusters with zero cross-edges → trivially partitionable → Phi=0."""
    matrix = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
    assert calc.calculate(matrix) == 0.0


def test_fully_connected_phi_is_positive(calc: MinCutPhiCalculator) -> None:
    """An all-ones (off-diagonal) network has high min-cut weight."""
    n = 4
    matrix = [[0.0 if i == j else 1.0 for j in range(n)] for i in range(n)]
    phi = calc.calculate(matrix)
    assert phi > 0.0


def test_more_connections_yield_higher_phi(calc: MinCutPhiCalculator) -> None:
    """A denser network should have a stronger min-cut than a sparser one."""
    sparse = [
        [0.0, 0.1, 0.0, 0.0],
        [0.1, 0.0, 0.1, 0.0],
        [0.0, 0.1, 0.0, 0.1],
        [0.0, 0.0, 0.1, 0.0],
    ]
    dense = [
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
    ]
    assert calc.calculate(dense) > calc.calculate(sparse)


def test_two_node_phi_equals_normalized_cross_weight(calc: MinCutPhiCalculator) -> None:
    """With n=2 there's only one bipartition. Symmetric weight 0.5 → Phi=0.5."""
    matrix = [[0.0, 0.5], [0.5, 0.0]]
    phi = calc.calculate(matrix)
    assert phi == pytest.approx(0.5, abs=1e-9)


def test_balanced_cuts_preferred_over_peeling_one_node() -> None:
    """A network where peeling one node has the same total cross-weight as
    a balanced cut should prefer the balanced cut after normalization."""
    calc = MinCutPhiCalculator()
    # Path graph: 0—1—2—3 with weight 1.0 on each edge.
    # Cut {0} | {1,2,3}: cross=1.0, min_size=1, normalized=1.0
    # Cut {0,1} | {2,3}: cross=1.0, min_size=2, normalized=0.5
    # Cut {0,3} | {1,2}: cross=2.0, min_size=2, normalized=1.0
    matrix = [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    phi = calc.calculate(matrix)
    assert phi == pytest.approx(0.5, abs=1e-9)


def test_non_square_raises(calc: MinCutPhiCalculator) -> None:
    with pytest.raises(ValueError):
        calc.calculate([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_falls_back_to_spectral_above_max_n() -> None:
    """For n > max_n, the calculator should defer to the spectral approximation."""
    calc = MinCutPhiCalculator(max_n=3)
    # n=4 — exceeds the cap; should match the spectral output exactly.
    matrix = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    expected = PhiCalculator().calculate(matrix)
    assert calc.calculate(matrix) == pytest.approx(expected, abs=1e-9)


def test_max_n_must_be_at_least_two() -> None:
    with pytest.raises(ValueError):
        MinCutPhiCalculator(max_n=1)


def test_directed_asymmetric_is_handled(calc: MinCutPhiCalculator) -> None:
    """Directed asymmetric weights are symmetrized for the bipartition score."""
    # 3-node ring: 0→1, 1→2, 2→0 with weight 1.0
    matrix = [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
    phi = calc.calculate(matrix)
    # Symmetrized matrix has weight 0.5 on each edge in both directions.
    # Best cut for n=3: {0} vs {1,2} → cross weight (sym) = 0.5+0.5 = 1.0,
    # min_size=1 → normalized = 1.0. Other partitions are similar.
    assert phi > 0.0


def test_returns_float(calc: MinCutPhiCalculator) -> None:
    phi = calc.calculate([[0.0, 1.0], [1.0, 0.0]])
    assert isinstance(phi, float)


def test_can_swap_into_mc_layer() -> None:
    """The calculator must be drop-in compatible with MCLayer."""
    from awaking_os.consciousness.ethical_filter import EthicalFilter
    from awaking_os.consciousness.global_workspace import GlobalWorkspace
    from awaking_os.consciousness.mc_layer import MCLayer

    layer = MCLayer(
        phi_calculator=MinCutPhiCalculator(),
        ethical_filter=EthicalFilter(),
        global_workspace=GlobalWorkspace(),
    )
    # Just exercise the wiring; behavior is covered by the calculator's own tests.
    assert layer.phi.__class__.__name__ == "MinCutPhiCalculator"
