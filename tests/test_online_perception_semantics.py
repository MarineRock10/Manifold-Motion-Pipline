"""Signed radar-affordance regression for online primitive routing."""

from __future__ import annotations

import numpy as np

from manifold_motion.deploy_perception import (
    ProbabilisticSlidingVoxelGrid, SlidingGridConfig, _vertical_free_half_extent,
)
from manifold_motion.online_perception import (
    OnlinePerceptionNavigator, _bilateral_lateral_free_semi_from_radar,
)


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


def test_vertical_clearance_uses_body_footprint_neighborhood() -> None:
    config = SlidingGridConfig(size_xyz=(48, 48, 24), size_xy=(48, 48), resolution_m=0.08)
    grid = ProbabilisticSlidingVoxelGrid(config, np.array([0.0, 0.0, 0.78]))
    grid.log_odds[:] = -8.0
    z = int(np.floor(1.20 / config.resolution_m)) - int(grid.origin_cell_xyz[2])
    x = int(np.floor(0.20 / config.resolution_m)) - int(grid.origin_cell_xyz[0])
    y = int(np.floor(0.08 / config.resolution_m)) - int(grid.origin_cell_xyz[1])
    grid.log_odds[z:z + 1, y:y + 1, x:x + 2] = 8.0
    free = _vertical_free_half_extent(
        grid, np.array([0.0, 0.0]), 0.78, 1.20, 0.08)
    assert free < 0.50


def test_posture_release_requires_more_evidence_than_contraction() -> None:
    navigator = OnlinePerceptionNavigator.__new__(OnlinePerceptionNavigator)
    navigator.crouch_semi_z_m = 1.12
    navigator.side_semi_y_m = 0.40
    navigator.decision_confirm_updates = 2
    navigator.decision_release_confirm_updates = 4
    navigator.stable_primitive_id = 2
    navigator.pending_primitive_id = None
    navigator.pending_primitive_updates = 0
    open_corridor = np.zeros((8, 6), dtype=np.float64)
    open_corridor[:, 4] = 0.70
    open_corridor[:, 5] = 1.20
    for _ in range(3):
        decision = navigator._primitive_decision(open_corridor, 0.70)
        assert decision["primitive_id"] == 2
        assert decision["confirm_updates"] == 4
    decision = navigator._primitive_decision(open_corridor, 0.70)
    assert decision["primitive_id"] == 5
    assert decision["primitive_changed"]


if __name__ == "__main__":
    test_unilateral_obstacle_does_not_become_side_gate()
    test_bilateral_surfaces_contract_side_aperture()
    test_longitudinally_separated_surfaces_are_not_a_gate()
    test_vertical_clearance_uses_body_footprint_neighborhood()
    test_posture_release_requires_more_evidence_than_contraction()
    print("online perception semantic tests passed")
