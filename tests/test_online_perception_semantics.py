"""Signed radar-affordance regression for online primitive routing."""

from __future__ import annotations

import numpy as np

from manifold_motion.online_perception import _bilateral_lateral_free_semi_from_radar


def _surface(x: float, y: float, z: float = 0.75, count: int = 5) -> np.ndarray:
    return np.column_stack([
        np.linspace(x - 0.12, x + 0.12, count),
        np.full(count, y),
        np.full(count, z),
    ])


def test_unilateral_obstacle_does_not_become_side_gate() -> None:
    route = np.column_stack([np.linspace(0.0, 1.0, 32), np.zeros(32), np.full(32, 0.78)])
    unilateral = _surface(0.50, 0.31)
    value = _bilateral_lateral_free_semi_from_radar(route, unilateral, 0.78)
    assert value == 0.70


def test_bilateral_surfaces_contract_side_aperture() -> None:
    route = np.column_stack([np.linspace(0.0, 1.0, 32), np.zeros(32), np.full(32, 0.78)])
    bilateral = np.vstack([_surface(0.50, 0.31), _surface(0.50, -0.33)])
    value = _bilateral_lateral_free_semi_from_radar(route, bilateral, 0.78)
    assert 0.12 < value < 0.15


def test_longitudinally_separated_surfaces_are_not_a_gate() -> None:
    route = np.column_stack([np.linspace(0.0, 2.0, 64), np.zeros(64), np.full(64, 0.78)])
    alternating = np.vstack([_surface(0.25, 0.31), _surface(1.30, -0.33)])
    value = _bilateral_lateral_free_semi_from_radar(route, alternating, 0.78)
    assert value == 0.70


if __name__ == "__main__":
    test_unilateral_obstacle_does_not_become_side_gate()
    test_bilateral_surfaces_contract_side_aperture()
    test_longitudinally_separated_surfaces_are_not_a_gate()
    print("online perception semantic tests passed")
