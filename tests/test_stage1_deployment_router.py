from __future__ import annotations

import numpy as np
import torch

from manifold_motion.planning.primitive_router import Router
from manifold_motion.stage2.manifold_adaptive import (
    _route_decision, _stage1_router_prediction,
)


def _corridor(x: float = 0.95, y: float = 0.70, z: float = 1.34) -> np.ndarray:
    result = np.zeros((48, 7), dtype=np.float32)
    result[:, 3:6] = (x, y, z)
    return result


def test_route_decision_uses_bilateral_me_contraction() -> None:
    decision = _route_decision(
        _corridor(y=0.08), heading=0.8, previous_heading=0.8,
        crouch_semi_z_m=1.12, side_semi_y_m=0.42, turn_threshold_rad=0.28,
        bilateral_lateral_semi_m=0.08,
    )
    assert decision["primitive_id"] == 4
    assert decision["reason"] == "bilateral_lateral_free_semi_below_side_threshold"


def test_route_decision_ignores_mild_one_sided_obstacle() -> None:
    decision = _route_decision(
        _corridor(y=0.35), heading=0.0, previous_heading=0.0,
        crouch_semi_z_m=1.12, side_semi_y_m=0.42, turn_threshold_rad=0.28,
        bilateral_lateral_semi_m=0.70,
    )
    assert decision["primitive_id"] == 5


def test_stage1_prediction_uses_checkpoint_names_for_non_executor_family() -> None:
    model = Router(input_dim=17, output_dim=2, hidden=4)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        model.net[-1].bias[0] = 2.0
    checkpoint = {
        "feature_mean": np.zeros(17, dtype=np.float32),
        "feature_std": np.ones(17, dtype=np.float32),
        "active_primitive_ids": np.asarray([0, 5], dtype=np.int64),
        "active_primitive_names": ["all_fours", "walk_nominal"],
    }
    result = _stage1_router_prediction(
        model, checkpoint, _corridor(), np.zeros((10, 10, 8), dtype=np.float32),
    )
    assert result["primitive_id"] == 0
    assert result["primitive"] == "all_fours"
