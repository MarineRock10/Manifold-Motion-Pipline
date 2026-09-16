"""Thin wrapper around planner_sonic.onnx (kinematic motion generator)."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from . import constants as C


class SonicPlanner:
    MIN_TOKENS = 6
    MAX_TOKENS = 16

    def __init__(self, path=C.PLANNER_ONNX, num_threads: int = C.ONNX_THREADS):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1        # see above: keep parallel workers thread-light
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.allowed_tokens = np.ones((1, self.MAX_TOKENS - self.MIN_TOKENS + 1), dtype=np.int64)

    def plan(self, context_qpos: np.ndarray, mode: int = 2, movement=(1.0, 0.0, 0.0),
             facing=(1.0, 0.0, 0.0), target_vel: float = -1.0, height: float = -1.0,
             seed: int = 0) -> np.ndarray:
        """context_qpos: [4, 36] recent qpos frames (root xyz + quat wxyz + 29 MuJoCo joints)."""
        inputs = {
            "context_mujoco_qpos": np.asarray(context_qpos, dtype=np.float32)[None, ...],
            "target_vel": np.array([target_vel], dtype=np.float32),
            "mode": np.array([mode], dtype=np.int64),
            "movement_direction": np.array([movement], dtype=np.float32),
            "facing_direction": np.array([facing], dtype=np.float32),
            "random_seed": np.array([seed], dtype=np.int64),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": self.allowed_tokens,
            "height": np.array([height], dtype=np.float32),
        }
        qpos, num_frames = self.session.run(None, inputs)
        return qpos[0, : int(num_frames[0])]
