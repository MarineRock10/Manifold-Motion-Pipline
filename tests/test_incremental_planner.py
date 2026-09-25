"""Fast regression checks for the incremental ESDF/D* Lite route contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from manifold_motion.deploy_perception import ProbabilisticSlidingVoxelGrid, SlidingGridConfig
from manifold_motion.incremental_planner import (
    DStarLitePlanner,
    IncrementalPlannerConfig,
    IncrementalTruncatedESDF,
    _euclidean_distance_cells,
)


def test_esdf_dirty_update_matches_full() -> None:
    rng = np.random.default_rng(20260923)
    first = rng.random((32, 36)) < 0.07
    second = first.copy()
    second[12:15, 17:21] = True
    second[4:6, 7:9] = False
    incremental = IncrementalTruncatedESDF(0.08, 1.2)
    report0 = incremental.update(first, np.array([100, -20]))
    report1 = incremental.update(second, np.array([101, -20]))
    expected = np.minimum(_euclidean_distance_cells(second) * 0.08, 1.2)
    assert np.allclose(incremental.distance_m, expected, atol=1.0e-5)
    assert report0["full_rebuild"] and not report1["full_rebuild"]
    assert report1["changed_cells"] > 0


def _grid_with_block(block_x: int) -> ProbabilisticSlidingVoxelGrid:
    config = SlidingGridConfig(size_xyz=(64, 64, 16), size_xy=(64, 64), resolution_m=0.08)
    grid = ProbabilisticSlidingVoxelGrid(config, np.array([0.0, 0.0, 0.78]))
    # A thin wall in the robot-height band; all other voxels remain unknown/free for this test.
    z0 = int(np.floor(0.25 / config.resolution_m)) - int(grid.origin_cell_xyz[2])
    z1 = int(np.ceil(1.30 / config.resolution_m)) - int(grid.origin_cell_xyz[2])
    x = block_x - int(grid.origin_cell_xyz[0])
    y0 = int(np.floor(-0.65 / config.resolution_m)) - int(grid.origin_cell_xyz[1])
    y1 = int(np.ceil(0.65 / config.resolution_m)) - int(grid.origin_cell_xyz[1])
    grid.log_odds[:, :, :] = -8.0
    grid.log_odds[max(0, z0):min(grid.shape_zyx[0], z1), max(0, y0):min(grid.shape_zyx[1], y1), x:x + 1] = 8.0
    return grid


def test_dstar_reuses_state_after_map_change(tmp_path: Path) -> None:
    cfg = IncrementalPlannerConfig(body_radius_m=0.16, clearance_m=0.08,
                                   body_half_height_m=0.60, route_preference_weight=0.0)
    planner = DStarLitePlanner(0.08, cfg)
    first = _grid_with_block(12)
    route0, report0 = planner.plan(first, np.array([-1.8, 0.0, 0.78]), np.array([1.8, 0.0, 0.78]))
    assert len(route0) > 2 and report0["reinitialized"]
    second = _grid_with_block(12)
    # Removing one opening changes the route costs without resetting D* Lite.
    x = 12 - int(second.origin_cell_xyz[0])
    y = int(np.floor(0.0 / second.config.resolution_m)) - int(second.origin_cell_xyz[1])
    second.log_odds[:, max(0, y - 2):y + 3, x:x + 1] = -8.0
    route1, report1 = planner.plan(second, np.array([-1.7, 0.0, 0.78]), np.array([1.8, 0.0, 0.78]))
    assert len(route1) > 2
    assert not report1["reinitialized"]
    assert report1["changed_cost_cells"] > 0
    payload = {"initial": report0, "after_map_change": report1,
               "route0_points": int(len(route0)), "route1_points": int(len(route1))}
    (tmp_path / "incremental_planner_report.json").write_text(json.dumps(payload, indent=2))


def test_overhead_ceiling_does_not_block_ground_route() -> None:
    config = SlidingGridConfig(size_xyz=(64, 64, 24), size_xy=(64, 64), resolution_m=0.08)
    grid = ProbabilisticSlidingVoxelGrid(config, np.array([0.0, 0.0, 0.78]))
    grid.log_odds[:] = -8.0
    z = int(np.floor(1.20 / config.resolution_m)) - int(grid.origin_cell_xyz[2])
    x = int(np.floor(0.0 / config.resolution_m)) - int(grid.origin_cell_xyz[0])
    y0 = int(np.floor(-0.70 / config.resolution_m)) - int(grid.origin_cell_xyz[1])
    y1 = int(np.ceil(0.70 / config.resolution_m)) - int(grid.origin_cell_xyz[1])
    grid.log_odds[z:z + 1, y0:y1 + 1, x:x + 1] = 8.0
    planner = DStarLitePlanner(0.08, IncrementalPlannerConfig(
        body_radius_m=0.16, clearance_m=0.08, body_half_height_m=0.78,
        ground_projection_height_m=0.45, route_preference_weight=0.0))
    route, _ = planner.plan(grid, np.array([-1.5, 0.0, 0.78]), np.array([1.5, 0.0, 0.78]))
    assert float(np.max(np.abs(route[:, 1]))) < 0.20


if __name__ == "__main__":
    test_esdf_dirty_update_matches_full()
    test_dstar_reuses_state_after_map_change(Path("/tmp"))
    test_overhead_ceiling_does_not_block_ground_route()
    print("incremental planner tests passed")
