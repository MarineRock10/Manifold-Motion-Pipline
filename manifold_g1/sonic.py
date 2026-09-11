"""Frozen SONIC encoder/decoder inference (ONNX Runtime) with the deploy's observation layout.

The observation assembly follows the (now removed) GEAR-SONIC C++ deployment:
  - encoder, g1 mode: mode id + 10-frame step-5 joint positions/velocities + anchor orientation
  - decoder: 64-D tokens + 10-frame step-1 history (oldest first, zero padded)
  - action: q_target = default_angles + action[isaaclab_to_mujoco] * action_scale
"""

from __future__ import annotations

from collections import deque

import numpy as np
import onnxruntime as ort

from . import constants as C


class SonicController:
    def __init__(self, encoder_path=C.ENCODER_ONNX, decoder_path=C.DECODER_ONNX,
                 obs_config=C.OBS_CONFIG_PATH):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 4
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(encoder_path), sess_options=options, providers=providers)
        self.decoder = ort.InferenceSession(str(decoder_path), sess_options=options, providers=providers)

        layouts = C.load_obs_layout(obs_config)
        self.enc_layout = layouts["encoder"]
        self.dec_layout = layouts["policy"]
        g1_mode = layouts["encoder_modes"]["g1"]
        self.mode_id = g1_mode["mode_id"]
        self.required = set(g1_mode["required"])

        self.enc_dim = int(self.encoder.get_inputs()[0].shape[1])
        self.dec_dim = int(self.decoder.get_inputs()[0].shape[1])
        self.enc_in = np.zeros(self.enc_dim, dtype=np.float32)
        self.dec_in = np.zeros(self.dec_dim, dtype=np.float32)

        self.history: deque[dict] = deque(maxlen=10)
        self.last_action = np.zeros(29, dtype=np.float32)
        self.delta_heading: np.ndarray | None = None
        self.reset()

    # -- state logging ------------------------------------------------------
    @staticmethod
    def _zero_entry() -> dict:
        return {
            "ang_vel": np.zeros(3, dtype=np.float32),
            "q_rel": np.zeros(29, dtype=np.float32),
            "dq": np.zeros(29, dtype=np.float32),
            "last_action": np.zeros(29, dtype=np.float32),
            "gravity": np.zeros(3, dtype=np.float32),
        }

    def reset(self) -> None:
        self.history.clear()
        for _ in range(10):
            self.history.append(self._zero_entry())
        self.last_action = np.zeros(29, dtype=np.float32)
        self.delta_heading = None

    def append_state(self, q_hw: np.ndarray, dq_hw: np.ndarray, base_quat: np.ndarray,
                     base_ang_vel: np.ndarray) -> None:
        """Record one 50 Hz control tick; last_action is the action from the previous tick."""
        q_rel = q_hw[C.MUJOCO_TO_ISAACLAB] - C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB]
        gravity = C.quat_rotate(C.quat_conj(base_quat), np.array([0.0, 0.0, -1.0]))
        self.history.append({
            "ang_vel": np.asarray(base_ang_vel, dtype=np.float32),
            "q_rel": q_rel.astype(np.float32),
            "dq": dq_hw[C.MUJOCO_TO_ISAACLAB].astype(np.float32),
            "last_action": self.last_action.copy(),
            "gravity": gravity.astype(np.float32),
        })

    # -- observation assembly ----------------------------------------------
    def _encoder_input(self, reference, base_quat: np.ndarray) -> np.ndarray:
        self.enc_in.fill(0.0)
        for name, (dim, offset) in self.enc_layout.items():
            if name not in self.required:
                continue
            if name == "encoder_mode_4":
                self.enc_in[offset] = self.mode_id
            elif name == "motion_joint_positions_10frame_step5":
                for i in range(10):
                    frame = reference.frame(i, 5)
                    self.enc_in[offset + i * 29: offset + (i + 1) * 29] = reference.joint_pos[frame]
            elif name == "motion_joint_velocities_10frame_step5":
                for i in range(10):
                    frame = reference.frame(i, 5)
                    if reference.play:
                        self.enc_in[offset + i * 29: offset + (i + 1) * 29] = reference.joint_vel[frame]
            elif name == "motion_anchor_orientation_10frame_step5":
                for i in range(10):
                    frame = reference.frame(i, 5)
                    self.enc_in[offset + i * 6: offset + (i + 1) * 6] = C.anchor_orientation_6d(
                        base_quat, reference.root_quat[frame], self.delta_heading)
            else:
                raise NotImplementedError(f"encoder observation '{name}' is not handled")
        return self.enc_in

    def _decoder_input(self, tokens: np.ndarray) -> np.ndarray:
        history = list(self.history)  # oldest -> newest
        self.dec_in.fill(0.0)
        for name, (dim, offset) in self.dec_layout.items():
            if name == "token_state":
                self.dec_in[offset: offset + dim] = tokens
            elif name.startswith("his_base_angular_velocity"):
                self.dec_in[offset: offset + dim] = np.stack([e["ang_vel"] for e in history]).ravel()
            elif name.startswith("his_body_joint_positions"):
                self.dec_in[offset: offset + dim] = np.stack([e["q_rel"] for e in history]).ravel()
            elif name.startswith("his_body_joint_velocities"):
                self.dec_in[offset: offset + dim] = np.stack([e["dq"] for e in history]).ravel()
            elif name.startswith("his_last_actions"):
                self.dec_in[offset: offset + dim] = np.stack([e["last_action"] for e in history]).ravel()
            elif name.startswith("his_gravity_dir"):
                self.dec_in[offset: offset + dim] = np.stack([e["gravity"] for e in history]).ravel()
            else:
                raise NotImplementedError(f"policy observation '{name}' is not handled")
        return self.dec_in

    # -- control tick -------------------------------------------------------
    def act(self, reference, base_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (raw action in IsaacLab order, q_target in hardware order, tokens)."""
        if self.delta_heading is None:
            ref_quat = reference.root_quat[reference.cursor]
            self.delta_heading = C.quat_mul(C.heading_quat(base_quat), C.quat_conj(C.heading_quat(ref_quat)))

        enc_in = self._encoder_input(reference, base_quat)
        tokens = self.encoder.run(None, {"obs_dict": enc_in[None, :]})[0][0]

        dec_in = self._decoder_input(tokens)
        action = self.decoder.run(None, {"obs_dict": dec_in[None, :]})[0][0].astype(np.float64)

        q_target = C.DEFAULT_ANGLES + action[C.ISAACLAB_TO_MUJOCO] * C.ACTION_SCALE
        self.last_action = action.astype(np.float32)
        return action, q_target, tokens
