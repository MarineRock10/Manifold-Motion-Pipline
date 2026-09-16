"""Static keyframe references for the frozen SONIC controller.

A keyframe is a single-frame motion reference (T = 1): the standing pose plus a combination
of *pose directions* (crouch, lean). SONIC holds whatever keyframe it is given, so the static
policy's output maps directly onto the reference - no kinematic planner is involved.

Directions live in IsaacLab joint order (the order the reference and the policy use).
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


# --- pose directions ---------------------------------------------------------
# The reference joint vector is in policy (IsaacLab) order, which interleaves left/right and
# puts the ankle before the shoulder - so directions are written by joint name, never by index.
def policy_index(name: str) -> int:
    """Index of a hardware joint name in the policy (IsaacLab) joint order."""
    if name not in C.MOTOR_NAMES:
        raise KeyError(f"unknown joint '{name}'")
    return int(C.ISAACLAB_TO_MUJOCO[C.MOTOR_NAMES.index(name)])


def direction(**joints: float) -> np.ndarray:
    """A pose direction from hardware joint names -> policy-order offsets."""
    vec = np.zeros(len(C.MOTOR_NAMES))
    for name, value in joints.items():
        vec[policy_index(name)] = value
    return vec


# sagittal crouch: hips flex, knees bend, ankles dorsiflex
CROUCH_DIRECTION = direction(
    left_hip_pitch=-0.6, right_hip_pitch=-0.6,
    left_knee=1.0, right_knee=1.0,
    left_ankle_pitch=-0.4, right_ankle_pitch=-0.4,
)

# forward lean of the torso, with the ankles and hips compensating so the robot stays balanced
LEAN_DIRECTION = direction(
    waist_pitch=1.0,
    left_hip_pitch=-0.25, right_hip_pitch=-0.25,
    left_ankle_pitch=-0.35, right_ankle_pitch=-0.35,
)

# trunk twist: the waist yaws, which swings the arm span out of the y axis and the shoulders
# carry the arms along, so a laterally narrow manifold can still be entered side-on.
TWIST_DIRECTION = direction(
    waist_yaw=1.0,
    left_shoulder_yaw=0.25, right_shoulder_yaw=0.25,
)

# arm tuck: shoulders roll in and the elbows fold, which pulls the hands to the body. The hands
# are the widest and the furthest-forward part of the robot, so this is the pose dimension that
# shrinks the footprint (measured: |y| 0.248 -> 0.194 m, |x| 0.294 -> 0.223 m at full tuck).
ARM_DIRECTION = direction(
    left_shoulder_roll=-0.8, right_shoulder_roll=0.8,
    left_elbow=1.2, right_elbow=1.2,
)


class ReferenceBuffer:
    """Motion reference the SONIC encoder reads; a keyframe is a single-frame buffer."""

    def __init__(self, joint_pos, joint_vel, root_pos, root_quat, play: bool = True):
        self.joint_pos = np.asarray(joint_pos, dtype=np.float64)   # [T, 29] IsaacLab order
        self.joint_vel = np.asarray(joint_vel, dtype=np.float64)
        self.root_pos = np.asarray(root_pos, dtype=np.float64)     # [T, 3]
        self.root_quat = np.asarray(root_quat, dtype=np.float64)   # [T, 4] wxyz
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


class KeyframeReference(ReferenceBuffer):
    """Standing keyframe plus crouch/lean/twist offsets, held until the pose changes."""

    def __init__(self, root_quat, crouch: float = 0.0, lean: float = 0.0, twist: float = 0.0,
                 arms: float = 0.0):
        joints = C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB].copy()
        super().__init__(joints[None, :], np.zeros((1, 29)),
                         np.array([[0.0, 0.0, C.DEFAULT_HEIGHT]]),
                         np.asarray(root_quat, dtype=np.float64)[None, :])
        self.crouch = 0.0
        self.lean = 0.0
        self.twist = 0.0
        self.arms = 0.0
        self.set_pose(crouch, lean, twist, arms)

    def set_joints(self, q_isaac: np.ndarray) -> None:
        """Set the keyframe from an explicit policy-order joint vector.

        The primitive model outputs all 29 joints, not the four pose channels, so validating its
        poses on SONIC means handing the keyframe the exact joint vector - the four-channel path
        cannot express what it produces.
        """
        joints = C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB] + np.asarray(q_isaac, dtype=np.float64)
        self.joint_pos = joints[None, :]
        self.joint_vel = np.zeros((1, len(joints)))

    def set_pose(self, crouch: float, lean: float = 0.0, twist: float = 0.0,
                 arms: float = 0.0) -> None:
        delta = ((float(crouch) - self.crouch) * CROUCH_DIRECTION
                 + (float(lean) - self.lean) * LEAN_DIRECTION
                 + (float(twist) - self.twist) * TWIST_DIRECTION
                 + (float(arms) - self.arms) * ARM_DIRECTION)
        self.joint_pos = self.joint_pos + delta[None, :]
        self.crouch = float(crouch)
        self.lean = float(lean)
        self.twist = float(twist)
        self.arms = float(arms)

    @classmethod
    def standing(cls, root_quat) -> "KeyframeReference":
        return cls(root_quat)


class PlannedReference(ReferenceBuffer):
    """A motion clip generated online by planner_sonic.onnx, spliced with an 8-frame crossfade.

    This is the moving counterpart of `KeyframeReference`: the static stage holds one pose,
    the data pipeline replays whole motions and records what SONIC actually does with them.
    """

    @classmethod
    def from_qpos(cls, qpos: np.ndarray) -> "PlannedReference":
        joints = qpos[:, 7:36][:, C.MUJOCO_TO_ISAACLAB]      # MuJoCo order -> policy order
        vel = np.zeros_like(joints)
        if len(joints) > 1:
            vel[:-1] = np.diff(joints, axis=0) / C.PLANNER_DT
            vel[-1] = vel[-2]
        return cls(joints, vel, qpos[:, 0:3], qpos[:, 3:7])

    @classmethod
    def from_plan(cls, planner, context_qpos: np.ndarray, **plan_kwargs) -> "PlannedReference":
        return cls.from_qpos(planner.plan(context_qpos, **plan_kwargs))

    def context_qpos(self, frames: int = 4) -> np.ndarray:
        """The last `frames` qpos rows of the clip, for the planner's context input."""
        idx = [min(self.cursor + i * 5, self.T - 1) for i in range(frames)]
        qpos = np.concatenate([self.root_pos[idx], self.root_quat[idx],
                               C.DEFAULT_ANGLES[None, :].repeat(len(idx), axis=0)], axis=1)
        joints = self.joint_pos[idx]
        qpos[:, 7:36] = joints[:, C.ISAACLAB_TO_MUJOCO]
        return qpos

    def maybe_replan(self, planner, reserve: int = 16, context: np.ndarray | None = None,
                     force: bool = False, **plan_kwargs) -> bool:
        """Replan when fewer than `reserve` frames remain (or when forced)."""
        if not force and self.T - self.cursor > reserve:
            return False
        ctx = self.context_qpos() if context is None else np.asarray(context, dtype=np.float64)
        self.splice(PlannedReference.from_plan(planner, ctx, **plan_kwargs))
        return True

    def splice(self, new: "PlannedReference", blend: int = 8) -> None:
        m = min(blend, new.T)
        joint_pos, joint_vel = new.joint_pos.copy(), new.joint_vel.copy()
        root_pos, root_quat = new.root_pos.copy(), new.root_quat.copy()
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
