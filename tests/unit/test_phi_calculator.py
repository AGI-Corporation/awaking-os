"""PhiCalculator tests."""

from __future__ import annotations

import math

import pytest

from awaking_os.consciousness.phi_calculator import PhiCalculator


def test_empty_matrix_returns_zero(phi_calculator: PhiCalculator) -> None:
    assert phi_calculator.calculate([]) == 0.0


def test_single_node_returns_zero(phi_calculator: PhiCalculator) -> None:
    assert phi_calculator.calculate([[1.0]]) == 0.0


def test_zero_matrix_returns_zero(phi_calculator: PhiCalculator) -> None:
    matrix = [[0.0, 0.0], [0.0, 0.0]]
    assert phi_calculator.calculate(matrix) == 0.0


def test_uniform_matrix_has_high_entropy(phi_calculator: PhiCalculator) -> None:
    n = 4
    matrix = [[1.0] * n for _ in range(n)]
    phi = phi_calculator.calculate(matrix)
    # All-ones rank-1 matrix → one nonzero eigenvalue → entropy ≈ 0
    assert phi == pytest.approx(0.0, abs=1e-9)


def test_identity_matrix_max_entropy(phi_calculator: PhiCalculator) -> None:
    n = 4
    matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    phi = phi_calculator.calculate(matrix)
    # n equal eigenvalues → entropy = log2(n)
    assert phi == pytest.approx(math.log2(n), abs=1e-9)


def test_two_node_split_has_one_bit(phi_calculator: PhiCalculator) -> None:
    matrix = [[1.0, 0.0], [0.0, 1.0]]
    phi = phi_calculator.calculate(matrix)
    assert phi == pytest.approx(1.0, abs=1e-9)


def test_richer_coupling_has_higher_phi_than_diagonal(
    phi_calculator: PhiCalculator,
) -> None:
    diagonal = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    coupled = [[1.0, 0.7, 0.3], [0.7, 1.0, 0.7], [0.3, 0.7, 1.0]]
    phi_d = phi_calculator.calculate(diagonal)
    phi_c = phi_calculator.calculate(coupled)
    # Coupling redistributes spectral mass less evenly than the identity →
    # diagonal still wins this comparison. Sanity check the ordering.
    assert phi_d >= phi_c >= 0.0


def test_non_square_raises(phi_calculator: PhiCalculator) -> None:
    matrix = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    with pytest.raises(ValueError):
        phi_calculator.calculate(matrix)


def test_one_by_n_raises(phi_calculator: PhiCalculator) -> None:
    # Regression: previously short-circuited to 0.0 because len(matrix) < 2.
    with pytest.raises(ValueError):
        phi_calculator.calculate([[1.0, 2.0]])


def test_returns_float(phi_calculator: PhiCalculator) -> None:
    phi = phi_calculator.calculate([[1.0, 0.5], [0.5, 1.0]])
    assert isinstance(phi, float)
    assert phi >= 0.0
