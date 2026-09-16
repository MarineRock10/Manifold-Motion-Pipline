"""Constants mirrored from the GEAR-SONIC deployment stack.

The C++ deployment sources are no longer part of this repository; the values below were
taken from its `policy_parameters.hpp`, and the observation layout is still parsed from
`gear_sonic_deploy/policy/release/observation_config.yaml`, which ships with the ONNX models.

Two joint orders appear everywhere:
  hardware/MuJoCo order - MJCF actuator order, used by default_angles, gains and LowCmd.
  IsaacLab/policy order - SONIC decoder output and motion-reference joint data.
The arrays below are the exact permutations used by the C++ deployment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent

OBS_CONFIG_PATH = REPO / "gear_sonic_deploy" / "policy" / "release" / "observation_config.yaml"
ENCODER_ONNX = REPO / "gear_sonic_deploy" / "policy" / "release" / "model_encoder.onnx"
DECODER_ONNX = REPO / "gear_sonic_deploy" / "policy" / "release" / "model_decoder.onnx"
PLANNER_ONNX = REPO / "gear_sonic_deploy" / "planner" / "target_vel" / "V2" / "planner_sonic.onnx"
FLAT_SCENE = REPO / "data" / "g1_flat" / "scene_flat.xml"

# --- hardware/MuJoCo motor order (MJCF actuator order) -----------------------
MOTOR_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
assert len(MOTOR_NAMES) == 29

DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float64)
assert DEFAULT_ANGLES.shape == (29,)

# --- policy/IsaacLab order permutations -------------------------------------
# A[hw] -> isaaclab index (used for the decoder action and hardware->policy reorder)
ISAACLAB_TO_MUJOCO = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
], dtype=np.int64)
# B[isaaclab] -> hardware index (used to reorder measured state into policy order)
MUJOCO_TO_ISAACLAB = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
], dtype=np.int64)
assert np.array_equal(ISAACLAB_TO_MUJOCO[MUJOCO_TO_ISAACLAB], np.arange(29))

# --- gains and action scaling (policy_parameters.hpp) ------------------------
_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_NATURAL_FREQ = 10.0 * 2.0 * np.pi
_DAMPING_RATIO = 2.0

MOTOR_TYPES = [
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_14", "5020", "5020",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
]

# ankle pitch/roll and waist roll/pitch use doubled gains (see kps/kds in policy_parameters.hpp)
_GAIN_MULT = np.ones(29)
for _i in (4, 5, 10, 11, 13, 14):
    _GAIN_MULT[_i] = 2.0

KP = np.array([
    _GAIN_MULT[i] * _ARMATURE[t] * _NATURAL_FREQ**2 for i, t in enumerate(MOTOR_TYPES)
], dtype=np.float64)
KD = np.array([
    _GAIN_MULT[i] * 2.0 * _DAMPING_RATIO * _ARMATURE[t] * _NATURAL_FREQ for i, t in enumerate(MOTOR_TYPES)
], dtype=np.float64)
ACTION_SCALE = np.array([
    0.25 * _EFFORT[t] / (_ARMATURE[t] * _NATURAL_FREQ**2) for t in MOTOR_TYPES
], dtype=np.float64)

# torque clamps, from motor_effort_limit_list in wbc_configs/g1_29dof_sonic_model12.yaml (first 29)
EFFORT_LIMITS = np.array([
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 50.0, 50.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
], dtype=np.float64)

HAND_MOTOR_NAMES = [
    f"{side}_hand_{finger}_{i}"
    for side in ("left", "right")
    for finger, count in (("thumb", 3), ("middle", 2), ("index", 2))
    for i in range(count)
]
assert len(HAND_MOTOR_NAMES) == 14

DEFAULT_HEIGHT = 0.793  # pelvis height used by the repo simulator at reset
CONTROL_DT = 0.02      # 50 Hz policy/control tick
PLANNER_DT = 1.0 / 30.0  # planner_sonic.onnx emits frames at 30 Hz
# ONNX intra-op threads per session. Collection runs several processes in parallel, so a big
# per-session thread count oversubscribes the CPU: 6 workers x 8 threads on 12 cores pushed the
# load average to 76 and slowed every worker down.
ONNX_THREADS = 2
SIM_DT = 0.005         # 200 Hz physics
DECIMATION = 4






def load_obs_layout(path: Path = OBS_CONFIG_PATH) -> dict:
    """Read observation_config.yaml and return name -> (dim, offset) for both groups."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    layouts: dict[str, dict[str, tuple[int, int]]] = {}
    for group, entries in (("policy", cfg.get("observations", [])),
                           ("encoder", (cfg.get("encoder") or {}).get("encoder_observations", []))):
        offset = 0
        layout: dict[str, tuple[int, int]] = {}
        for entry in entries:
            if not entry.get("enabled", True):
                continue
            name = entry["name"]
            if name == "token_state":
                dim = int((cfg.get("encoder") or {}).get("dimension", 64))
            else:
                dim = _REGISTRY_DIMS[name]
            layout[name] = (int(dim), offset)
            offset += int(dim)
        layouts[group] = layout

    modes = {}
    for mode in (cfg.get("encoder") or {}).get("encoder_modes", []):
        modes[mode["name"]] = dict(mode_id=int(mode["mode_id"]),
                                   required=list(mode.get("required_observations", [])))
    layouts["encoder_modes"] = modes
    return layouts


# exact dimensions of the observations used by the default release config
_REGISTRY_DIMS = {
    # policy observations
    "his_base_angular_velocity_4frame_step1": 12,
    "his_body_joint_positions_4frame_step1": 116,
    "his_body_joint_velocities_4frame_step1": 116,
    "his_last_actions_4frame_step1": 116,
    "his_gravity_dir_4frame_step1": 12,
    "his_base_angular_velocity_10frame_step1": 30,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_gravity_dir_10frame_step1": 30,
    "base_angular_velocity": 3,
    "body_joint_positions": 29,
    "body_joint_velocities": 29,
    "last_actions": 29,
    "gravity_dir": 3,
    # encoder observations
    "encoder_mode": 3,
    "encoder_mode_4": 4,
    "motion_joint_positions": 29,
    "motion_joint_velocities": 29,
    "motion_anchor_orientation": 6,
    "motion_root_z_position": 1,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_root_z_position_10frame_step5": 10,
    "motion_anchor_orientation_10frame_step5": 60,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_10frame_step1": 60,
    "motion_joint_positions_wrists_10frame_step1": 60,
}


# --- small quaternion helpers (w, x, y, z) -----------------------------------
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_matrix(q) @ v


def quat_from_yaw(yaw: np.ndarray | float) -> np.ndarray:
    half = 0.5 * np.asarray(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def heading_quat(q: np.ndarray) -> np.ndarray:
    """Yaw-only quaternion of q, via the forward axis azimuth (matches calc_heading_quat_d)."""
    forward = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    return quat_from_yaw(np.arctan2(forward[1], forward[0]))




def anchor_orientation_6d(base_quat: np.ndarray, ref_quat: np.ndarray,
                          apply_delta_heading: np.ndarray) -> np.ndarray:
    """First two rotation-matrix columns (row-wise) of conj(base) * delta * ref."""
    ref = quat_mul(apply_delta_heading, ref_quat)
    rel = quat_mul(quat_conj(base_quat), ref)
    rot = quat_to_matrix(rel)
    return np.array([rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], rot[2, 0], rot[2, 1]])
