"""Frozen SONIC encoder/decoder inference (ONNX Runtime) with the deploy's observation layout.

The observation assembly follows the (now removed) GEAR-SONIC C++ deployment:
  - encoder, g1 mode: mode id + 10-frame step-5 joint positions/velocities + anchor orientation
  - decoder: 64-D tokens + 10-frame step-1 history (oldest first, zero padded)
  - action: q_target = default_angles + action[isaaclab_to_mujoco] * action_scale
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

from manifold_motion.core import constants as C


class SonicController:
    def __init__(self, encoder_path=C.ENCODER_ONNX, decoder_path=C.DECODER_ONNX,
                 obs_config=C.OBS_CONFIG_PATH, adapter_path: Path | None = None,
                 adapter_device: str = "cpu"):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = C.ONNX_THREADS
        options.inter_op_num_threads = 1        # ORT defaults this to the core count, which
                                                # oversubscribes the CPU across parallel workers
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
        # The adapter is deliberately opt-in.  Keeping the ONNX sessions as the frozen base
        # makes a missing/failed fine-tuning artifact fail closed to the proven controller.
        self.adapter = None
        self.adapter_device = str(adapter_device)
        self.adapter_condition: np.ndarray | None = None
        self.adapter_action_mask: np.ndarray | None = None
        self.adapter_path: str | None = None
        if adapter_path is not None:
            self._load_adapter(Path(adapter_path))
        self.reset()

    def _load_adapter(self, path: Path) -> None:
        """Load a trained bounded residual without changing frozen SONIC at zero residual."""
        if not path.is_file():
            raise FileNotFoundError(f"SONIC adapter checkpoint not found: {path}")
        import torch
        from manifold_motion.stage2.sonic_adapter import SonicAdapterConfig, SonicConditionAdapter
        payload = torch.load(path, map_location=self.adapter_device, weights_only=False)
        if not isinstance(payload, dict) or "config" not in payload or "state_dict" not in payload:
            raise ValueError("SONIC adapter checkpoint must contain config and state_dict")
        config = SonicAdapterConfig(**payload["config"])
        model = SonicConditionAdapter(config).to(self.adapter_device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        self.adapter = model
        self.adapter_path = str(path)

    def set_adapter_condition(self, condition: np.ndarray | None,
                              action_mask: np.ndarray | None = None) -> None:
        """Set one flattened Stage-2 condition for the next act() call.

        Conditions are refreshed every control tick by the route executor.  ``None`` disables
        the residual for that tick, which is the safe behavior during warm-up/hold phases.
        """
        self.adapter_condition = (None if condition is None else
                                  np.asarray(condition, dtype=np.float32).reshape(-1))
        self.adapter_action_mask = (None if action_mask is None else
                                    np.asarray(action_mask, dtype=np.float32).reshape(29))

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

        if self.adapter is not None and self.adapter_condition is not None:
            import torch
            base_tensor = torch.as_tensor(action.astype(np.float32)[None, :],
                                          device=self.adapter_device)
            condition_tensor = torch.as_tensor(self.adapter_condition[None, :],
                                               device=self.adapter_device)
            mask_tensor = (None if self.adapter_action_mask is None else
                           torch.as_tensor(self.adapter_action_mask[None, :],
                                           device=self.adapter_device))
            with torch.no_grad():
                action = self.adapter(base_tensor, condition_tensor, mask_tensor)[0].cpu().numpy()
            action = np.asarray(action, dtype=np.float64)

        q_target = C.DEFAULT_ANGLES + action[C.ISAACLAB_TO_MUJOCO] * C.ACTION_SCALE
        self.last_action = action.astype(np.float32)
        return action, q_target, tokens
