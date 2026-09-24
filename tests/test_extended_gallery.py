"""Fast geometry/routing regression for the extended long-sequence fixtures.

The full acceptance runner still performs Flow generation and continuous SONIC/MuJoCo
execution.  This test protects the preceding causal contract: each scene must remain
globally routable beyond the historical 3.65 m planner horizon, and its measured corridor
must request the intended primitive families without consulting a segment schedule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manifold_motion.stage2_long_horizon_avoidance import PlannerConfig, _astar, _simplify
from manifold_motion.stage2_manifold_adaptive import (
    _adaptive_segment_condition,
    _bilateral_lateral_free_semi,
    _calibrate_envelope,
    _ground_obstacles,
    _physical_boxes,
    _route_decision,
    _segments,
    _subdivide,
)


ROOT = Path(__file__).resolve().parents[1]


def _decisions(name: str, goal_x: float) -> tuple[np.ndarray, list[dict]]:
    scene = ROOT / "data" / "g1_flat" / f"scene_manifold_{name}.xml"
    ground = _ground_obstacles(scene)
    planner = PlannerConfig(
        x_bounds=(-0.05, max(3.65, goal_x + 0.05)),
        body_radius_m=0.40,
        clearance_m=0.10,
    )
    route = _simplify(
        _astar(np.array([0.0, 0.0]), np.array([goal_x, 0.0]), ground, planner),
        ground,
        planner,
    )
    keyframes = _subdivide(route, 0.45)
    segments, _ = _segments(keyframes)
    boxes = _physical_boxes(scene)
    envelope = _calibrate_envelope()
    previous_heading = 0.0
    decisions = []
    for segment in segments:
        corridor, _, _ = _adaptive_segment_condition(segment, boxes, envelope)
        heading = float(np.arctan2(
            segment[-1, 1] - segment[0, 1],
            segment[-1, 0] - segment[0, 0],
        ))
        decision = _route_decision(
            corridor,
            heading,
            previous_heading,
            crouch_semi_z_m=1.12,
            side_semi_y_m=0.42,
            turn_threshold_rad=0.28,
            bilateral_lateral_semi_m=_bilateral_lateral_free_semi(segment, boxes),
        )
        decisions.append(decision)
        previous_heading = heading
    return keyframes, decisions


def test_extended_fixture_routing_contract() -> None:
    chicane_k, chicane = _decisions("chicane", 6.0)
    compound_k, compound = _decisions("low_side_turn", 5.6)
    cycle_k, cycle = _decisions("gate_cycle", 6.4)
    slalom_k, slalom = _decisions("slalom", 6.0)

    assert len(chicane_k) - 1 == 17
    assert len(compound_k) - 1 == 18
    assert len(cycle_k) - 1 == 15
    assert len(slalom_k) - 1 == 18
    assert min(chicane_k[-1, 0], compound_k[-1, 0], cycle_k[-1, 0], slalom_k[-1, 0]) > 3.65

    assert {row["primitive"] for row in compound} >= {
        "crouch", "walk_lateral_reverse", "walk_nominal"
    }
    assert {row["primitive"] for row in cycle} >= {
        "crouch", "walk_lateral_reverse", "walk_nominal"
    }
    assert {row["primitive"] for row in chicane} == {"walk_nominal"}
    assert sum(row["primitive"] == "walk_lateral_reverse" for row in slalom) <= 3
    assert sum(row["primitive"] == "walk_nominal" for row in slalom) >= 12
    assert sum(row["requires_turn"] for row in chicane) == 9
    assert sum(row["requires_turn"] for row in compound) == 3
    assert sum(row["requires_turn"] for row in slalom) == 9


if __name__ == "__main__":
    test_extended_fixture_routing_contract()
    print("extended gallery routing tests passed")
