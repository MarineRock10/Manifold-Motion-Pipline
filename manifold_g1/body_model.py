"""The body model: landmark positions as a function of a 29-joint pose.

Extracted from `static_fit.py`, which also contained a retired 4-channel geometric training
route (`StaticFitEnv`, `PoseSlew`, and its `train`/`report`/`verify`/`view` CLI). Only
`BodyModel` has consumers today: the in-the-loop fine-tune, both viewers, the verifier and the
demonstration builder all use it to turn a pose into body-surface points, which is the ruler
every manifold in this project is measured with.

Two things it provides:

  * `at_pose` / `at_pose_batch` - the eight extremity landmark positions for a pose, by exact
    forward kinematics in MuJoCo (the calibrated 4-axis grid in `body_envelope.json` cannot
    represent a 29-joint pose like a walking frame, so poses are evaluated directly);
  * `mesh_points_pose` - the sampled body *surface*, which is what containment is measured
    against. `family.BASE_SEMI` and the demonstration manifolds are fitted from this, so it is
    the single definition of "the body's shape at a pose".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import constants as C

ENVELOPE_JSON = C.REPO / "reports" / "manifold_g1" / "body_envelope.json"


class BodyModel:
    """Landmark positions as a function of (crouch, lean, twist), trilinearly interpolated."""

    def __init__(self, path: Path = ENVELOPE_JSON):
        data = json.loads(Path(path).read_text())
        rows = data["rows"]
        self.offsets = {k: np.array(v) for k, v in data.get("landmark_offsets", {}).items()}
        self.crouch = np.array(sorted({r["crouch"] for r in rows}))
        self.lean = np.array(sorted({r["lean"] for r in rows}))
        self.twist = np.array(sorted({r["twist"] for r in rows}))
        self.arms = np.array(sorted({r["arms"] for r in rows}))
        self.names = list(rows[0]["landmarks"].keys())
        shape = (len(self.crouch), len(self.lean), len(self.twist), len(self.arms),
                 len(self.names), 3)
        self.grid = np.zeros(shape)
        for r in rows:
            i = int(np.argmin(np.abs(self.crouch - r["crouch"])))
            j = int(np.argmin(np.abs(self.lean - r["lean"])))
            k = int(np.argmin(np.abs(self.twist - r["twist"])))
            m = int(np.argmin(np.abs(self.arms - r["arms"])))
            self.grid[i, j, k, m] = [r["landmarks"][n] for n in self.names]
        standing = next(r for r in rows if r["crouch"] == 0.0 and r["lean"] == 0.0
                        and r["twist"] == 0.0 and r["arms"] == 0.0)
        self.standing_semi = np.array(standing["semi"])
        self.standing_center = np.array(standing["center"])

    def at_pose(self, pose: np.ndarray) -> np.ndarray:
        """Landmark positions for a 29-joint pose delta (policy order).

        Exact, by forward kinematics in MuJoCo: the calibrated 4-axis grid cannot represent a
        29-joint pose like a walking frame, so poses are evaluated directly.
        """
        return self.at_pose_batch(np.asarray(pose, dtype=np.float64)[None, :])[0]

    def mesh_points_pose(self, pose: np.ndarray, max_vertices: int = 120) -> np.ndarray:
        """Sampled body surface for a 29-joint pose delta, pelvis-anchored (mesh envelope).

        This is the ruler the manifold family is defined on (`family.BASE_SEMI` is a mesh fit),
        so containment for arbitrary poses must be judged the same way.
        """
        import mujoco

        from .body_envelope import body_points

        if not hasattr(self, "_kin"):
            self.at_pose_batch(np.zeros((1, len(C.DEFAULT_ANGLES))))
        env = self._kin
        env.data.qpos[env.body_qadr] = C.DEFAULT_ANGLES + np.asarray(pose)[C.ISAACLAB_TO_MUJOCO]
        mujoco.mj_kinematics(env.model, env.data)
        points = body_points(env.model, env.data, max_vertices=max_vertices)
        pelvis = env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
        return points - pelvis

    def at_pose_batch(self, poses: np.ndarray) -> np.ndarray:
        """Landmark positions for a batch of 29-joint pose deltas: [N,29] -> [N,K,3]."""
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, len(C.DEFAULT_ANGLES))
        if not hasattr(self, "_kin"):
            import mujoco
            from .env import G1FlatEnv
            self._kin = G1FlatEnv()
            self._kin_bid = [mujoco.mj_name2id(self._kin.model, mujoco.mjtObj.mjOBJ_BODY, name)
                             for name in self.names]
        import mujoco
        env = self._kin
        # landmark bodies are few, so read xpos/xmat directly instead of rebuilding dicts
        bid = np.array(self._kin_bid)
        offsets = np.stack([self.offsets[n] for n in self.names])       # [K, 3]
        out = np.zeros((len(poses), len(self.names), 3))
        for i, pose in enumerate(poses):
            env.data.qpos[env.body_qadr] = C.DEFAULT_ANGLES + pose[C.ISAACLAB_TO_MUJOCO]
            mujoco.mj_kinematics(env.model, env.data)      # poses only need body frames
            mats = env.data.xmat[bid].reshape(-1, 3, 3)
            points = env.data.xpos[bid] + np.einsum("kij,kj->ki", mats, offsets)
            out[i] = points - points[self.pelvis_index()]
        return out

    def pelvis_index(self) -> int:
        return self.names.index("pelvis")

    def read(self, env) -> dict[str, np.ndarray]:
        """The same landmark points, measured on the robot in MuJoCo."""
        from .body_envelope import read_landmarks

        return read_landmarks(env.env.model, env.env.data, self.offsets)

    def at(self, *pose) -> np.ndarray:
        """Landmark world positions for a pose (pelvis at x = y = 0)."""
        return self.at_batch(np.asarray(pose, dtype=np.float64)[None, :])[0]

    def axis_weights(self, values):
        """Per-axis bracketing index and interpolation weight inside the grid."""
        index, weight = [], []
        for axis, v in zip((self.crouch, self.lean, self.twist, self.arms), values):
            v = np.clip(v, axis[0], axis[-1])
            i = np.clip(np.searchsorted(axis, v) - 1, 0, len(axis) - 2)
            index.append(i)
            weight.append((v - axis[i]) / (axis[i + 1] - axis[i]))
        return index, weight

    def pose_grid(self, n_crouch: int = 9, n_lean: int = 7, n_twist: int = 9, n_arms: int = 7):
        """Batch of poses and their landmark positions: (poses [N,4], landmarks [N,K,3])."""
        axes = np.meshgrid(np.linspace(0.0, self.crouch[-1], n_crouch),
                           np.linspace(0.0, self.lean[-1], n_lean),
                           np.linspace(-self.twist[-1], self.twist[-1], n_twist),
                           np.linspace(0.0, self.arms[-1], n_arms), indexing="ij")
        poses = np.stack([a.ravel() for a in axes], axis=1)
        return poses, self.at_batch(poses)

    def at_batch(self, poses: np.ndarray) -> np.ndarray:
        """Landmark positions for a batch of poses: [N,4] -> [N,K,3] (vectorized interpolation)."""
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4)
        index, weight = self.axis_weights(poses.T)
        out = np.zeros((len(poses), len(self.names), 3))
        for a in (0, 1):
            for b in (0, 1):
                for d in (0, 1):
                    for e in (0, 1):
                        w = ((weight[0] if a else 1 - weight[0])
                             * (weight[1] if b else 1 - weight[1])
                             * (weight[2] if d else 1 - weight[2])
                             * (weight[3] if e else 1 - weight[3]))
                        out += w[:, None, None] * self.grid[index[0] + a, index[1] + b,
                                                           index[2] + d, index[3] + e]
        return out


