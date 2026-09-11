"""50 Hz motion-reference buffers consumed by the SONIC encoder.

A reference is the task input to the frozen controller: joint position/velocity
tracks in IsaacLab order plus the pelvis pose (root z and orientation are what the
g1-mode encoder observes).
"""

from __future__ import annotations

import numpy as np

from . import constants as C


def slerp(q0: np.ndarray, q1: np.ndarray, w: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + w * (q1 - q0)
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        s = np.sin(theta)
        out = np.sin((1.0 - w) * theta) / s * q0 + np.sin(w * theta) / s * q1
    return out / np.linalg.norm(out)


class ReferenceBuffer:
    """Fixed reference clip; the cursor advances one 50 Hz frame per control tick."""

    def __init__(self, joint_pos, joint_vel, root_pos, root_quat, play: bool = True):
        self.joint_pos = np.asarray(joint_pos, dtype=np.float64)   # [T, 29] IsaacLab order
        self.joint_vel = np.asarray(joint_vel, dtype=np.float64)   # [T, 29] IsaacLab order
        self.root_pos = np.asarray(root_pos, dtype=np.float64)     # [T, 3] world
        self.root_quat = np.asarray(root_quat, dtype=np.float64)   # [T, 4] wxyz world
        self.T = len(self.joint_pos)
        self.cursor = 0
        self.play = play

    def frame(self, idx: int = 0, step: int = 1) -> int:
        if self.play:
            return min(self.cursor + idx * step, self.T - 1)
        return min(self.cursor, self.T - 1)

    def advance(self) -> None:
        if self.play:
            self.cursor = min(self.cursor + 1, self.T - 1)

    @classmethod
    def static_stand(cls, robot_quat) -> "ReferenceBuffer":
        joints = C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB]
        return cls(
            joint_pos=joints[None, :],
            joint_vel=np.zeros((1, 29)),
            root_pos=np.array([[0.0, 0.0, C.DEFAULT_HEIGHT]]),
            root_quat=np.asarray(robot_quat, dtype=np.float64)[None, :],
        )

    def context_qpos(self, num: int = 4, rate: float = 30.0) -> np.ndarray:
        """Resample `num` qpos frames at `rate` Hz starting at the cursor (planner context)."""
        frames = []
        for n in range(num):
            f = min(self.cursor + n * (50.0 / rate), self.T - 1.0)
            f0 = int(np.floor(f))
            f1 = min(f0 + 1, self.T - 1)
            w1 = f - f0
            qpos = np.empty(7 + 29)
            qpos[0:3] = (1.0 - w1) * self.root_pos[f0] + w1 * self.root_pos[f1]
            qpos[3:7] = slerp(self.root_quat[f0], self.root_quat[f1], w1)
            joints_isaac = (1.0 - w1) * self.joint_pos[f0] + w1 * self.joint_pos[f1]
            qpos[7:] = joints_isaac[C.ISAACLAB_TO_MUJOCO]
            frames.append(qpos)
        return np.stack(frames)


class PlannedReference(ReferenceBuffer):
    """Reference generated online by planner_sonic.onnx with 8-frame crossfade splicing."""

    @classmethod
    def from_qpos(cls, qpos: np.ndarray) -> "PlannedReference":
        joints_isaac = qpos[:, 7:36][:, C.MUJOCO_TO_ISAACLAB]
        vel = np.zeros_like(joints_isaac)
        if len(joints_isaac) > 1:
            vel[:-1] = np.diff(joints_isaac, axis=0) * 50.0
            vel[-1] = vel[-2]
        return cls(joints_isaac, vel, qpos[:, 0:3], qpos[:, 3:7])

    @classmethod
    def from_plan(cls, planner, context_qpos: np.ndarray, **plan_kwargs) -> "PlannedReference":
        return cls.from_qpos(planner.plan(context_qpos, **plan_kwargs))

    def maybe_replan(self, planner, reserve: int = 16, context: np.ndarray | None = None,
                     force: bool = False, **plan_kwargs) -> bool:
        """Replan when fewer than `reserve` frames remain (or when forced)."""
        if not force and self.T - self.cursor > reserve:
            return False
        ctx = self.context_qpos() if context is None else np.asarray(context, dtype=np.float64)
        self.splice(PlannedReference.from_plan(planner, ctx, **plan_kwargs))
        return True

    def splice(self, new: "PlannedReference", blend: int = 8) -> None:
        m = min(blend, new.T, self.T - self.cursor)
        joint_pos = new.joint_pos.copy()
        joint_vel = new.joint_vel.copy()
        root_pos = new.root_pos.copy()
        root_quat = new.root_quat.copy()
        for f in range(m):
            w = (f + 1) / (m + 1)
            joint_pos[f] = (1.0 - w) * self.joint_pos[self.cursor + f] + w * joint_pos[f]
            joint_vel[f] = (1.0 - w) * self.joint_vel[self.cursor + f] + w * joint_vel[f]
            root_pos[f] = (1.0 - w) * self.root_pos[self.cursor + f] + w * root_pos[f]
            root_quat[f] = slerp(self.root_quat[self.cursor + f], root_quat[f], w)
        self.joint_pos, self.joint_vel = joint_pos, joint_vel
        self.root_pos, self.root_quat = root_pos, root_quat
        self.T = len(joint_pos)
        self.cursor = 0
        self.play = True
