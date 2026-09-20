"""Replay selected BONES-SEED G1 motions through the frozen SONIC controller.

The SEED CSVs are kinematic references at 120 Hz: root translations are centimetres,
all rotations are degrees, and the 29 body joint columns use G1's hardware naming.  A
CSV is *not* executable evidence by itself.  This module converts it into SONIC's
policy order, resamples it at the 50 Hz control rate, and records both sides of a
replay:

``q_ref``
    The reference supplied to the frozen SONIC encoder (policy / IsaacLab order).
``q_exec``
    The position that MuJoCo actually executed, in the same order.

Only clips that pass the stored safety and tracking checks should be used to train the
dynamic stage.  Failed clips are deliberately retained for audit; they must not quietly
become positive examples.

Examples
--------
Run three representative motions as a smoke test::

    python3 -m manifold_motion.seed_replay --motion-id seed2_00000 seed2_00700 seed2_01150

Replay a resumable pilot batch::

    python3 -m manifold_motion.seed_replay --jobs 4 --strata walk_nominal walk_turn
    python3 -m manifold_motion.seed_replay --jobs 4 --per-stratum-limit 25
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from . import constants as C
from .env import G1FlatEnv
from .reference import ReferenceBuffer
from .sonic import SonicController


SEED_SOURCE_HZ = 120.0
ROOT_COLUMNS = (
    "root_translateX", "root_translateY", "root_translateZ",
    "root_rotateX", "root_rotateY", "root_rotateZ",
)
JOINT_COLUMNS = tuple(f"{name}_joint_dof" for name in C.MOTOR_NAMES)


def _interpolate(values: np.ndarray, source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """Column-wise linear interpolation without pulling pandas into the runtime."""
    values = np.asarray(values, dtype=np.float64)
    return np.stack([np.interp(target_t, source_t, values[:, i])
                     for i in range(values.shape[1])], axis=1)


def _euler_degrees_to_quat(euler_deg: np.ndarray, sequence: str) -> np.ndarray:
    """Convert the CSV root Euler rows to normalized MuJoCo ``wxyz`` quaternions.

    BONES-SEED's G1 export stores XYZ Euler channels in degrees.  The default uses the
    matching extrinsic XYZ interpretation; ``--euler-sequence`` is exposed explicitly
    so a future export with a different rig convention is never silently misread.
    """
    euler = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    quat = np.empty((len(euler), 4), dtype=np.float64)
    for i, angles in enumerate(euler):
        mujoco.mju_euler2Quat(quat[i], angles, sequence)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return quat


@dataclass(frozen=True)
class SeedMotion:
    """One sanitized, control-rate SEED motion.

    Root positions retain the source coordinate axes and are metadata/audit signals;
    SONIC consumes the root orientation and the body joint reference.  This is important:
    the controller does not receive a root-position tracking objective, so comparing its
    world translation directly to the source would be a false failure criterion.
    """

    source_path: Path
    source_frames: np.ndarray             # [T], fractional source frame numbers at 50 Hz
    joint_pos_hw: np.ndarray              # [T, 29], radians, MuJoCo hardware order
    joint_vel_hw: np.ndarray              # [T, 29], radians/s, hardware order
    root_pos_seed_m: np.ndarray           # [T, 3], source axes, metres
    root_quat: np.ndarray                 # [T, 4], wxyz
    control_hz: float = 50.0

    @property
    def T(self) -> int:
        return int(len(self.source_frames))

    @property
    def duration_s(self) -> float:
        return float((self.T - 1) / self.control_hz) if self.T > 1 else 0.0

    @property
    def joint_pos_policy(self) -> np.ndarray:
        return self.joint_pos_hw[:, C.MUJOCO_TO_ISAACLAB]

    @property
    def joint_vel_policy(self) -> np.ndarray:
        return self.joint_vel_hw[:, C.MUJOCO_TO_ISAACLAB]

    def reference(self) -> ReferenceBuffer:
        return ReferenceBuffer(self.joint_pos_policy, self.joint_vel_policy,
                               self.root_pos_seed_m, self.root_quat, play=True)

    @classmethod
    def from_csv(cls, path: Path, *, source_hz: float = SEED_SOURCE_HZ,
                 control_hz: float = 50.0, euler_sequence: str = "XYZ") -> "SeedMotion":
        path = Path(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(ROOT_COLUMNS + JOINT_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is not a 29-DoF BONES-SEED G1 CSV; missing {sorted(missing)}")
            rows = list(reader)
        if len(rows) < 2:
            raise ValueError(f"{path} has fewer than two frames")

        try:
            source_frames = np.asarray([float(row["Frame"]) for row in rows], dtype=np.float64)
            root = np.asarray([[float(row[column]) for column in ROOT_COLUMNS] for row in rows],
                              dtype=np.float64)
            joints_deg = np.asarray([[float(row[column]) for column in JOINT_COLUMNS] for row in rows],
                                    dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"could not parse numeric BONES-SEED data from {path}") from error
        if not (np.isfinite(root).all() and np.isfinite(joints_deg).all()):
            raise ValueError(f"{path} contains non-finite values")
        if np.any(np.diff(source_frames) <= 0):
            raise ValueError(f"{path} has non-increasing Frame values")

        source_t = (source_frames - source_frames[0]) / float(source_hz)
        duration_s = float(source_t[-1])
        target_t = np.arange(0.0, duration_s + 1e-9, 1.0 / float(control_hz))
        if target_t[-1] < duration_s - 1e-7:
            target_t = np.append(target_t, duration_s)

        joint_pos_hw = np.deg2rad(_interpolate(joints_deg, source_t, target_t))
        root_pos_seed_m = _interpolate(root[:, :3], source_t, target_t) * 0.01

        # Euler channels can cross +/-180 degrees during a turn; unwrap before interpolation.
        euler_rad = np.unwrap(np.deg2rad(root[:, 3:]), axis=0)
        euler_deg = np.rad2deg(_interpolate(euler_rad, source_t, target_t))
        root_quat = _euler_degrees_to_quat(euler_deg, euler_sequence)

        joint_vel_hw = np.gradient(joint_pos_hw, target_t, axis=0, edge_order=1)
        frame_at_control = np.interp(target_t, source_t, source_frames)
        return cls(path, frame_at_control, joint_pos_hw, joint_vel_hw, root_pos_seed_m,
                   root_quat, float(control_hz))


@dataclass(frozen=True)
class ReplayConfig:
    """Safety gates used to decide whether a SONIC replay becomes training data."""

    warmup_seconds: float = 0.40
    ignore_metrics_seconds: float = 0.30
    fall_height: float = 0.30
    max_roll_deg: float = 45.0
    max_track_err_rad: float = 0.35
    max_leg_track_err_rad: float = 0.35
    min_foot_contact_ratio: float = 0.03
    max_nonfoot_floor_ratio: float = 0.03
    min_reference_planar_path_m: float = 0.25
    min_progress_ratio: float = 0.20
    # A source label containing "jump" is not proof that frozen SONIC actually executed a
    # jump.  These gates require a measurable take-off and landing in MuJoCo before the label
    # may enter the dynamic primitive library.
    jump_min_lift_m: float = 0.06
    jump_min_airborne_ticks: int = 3


class _ContactMonitor:
    """Classify support and named-scene-obstacle contacts without geom ordering."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.floor < 0:
            raise KeyError("flat scene has no geom named 'floor'")
        self.feet = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name): i
                     for i, name in enumerate(("left_ankle_roll_link", "right_ankle_roll_link"))}
        self.hand_sides: dict[int, int] = {}
        for body_id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if name.startswith("left_wrist") or name.startswith("left_hand"):
                self.hand_sides[body_id] = 0
            elif name.startswith("right_wrist") or name.startswith("right_hand"):
                self.hand_sides[body_id] = 1
        self.obstacles = {
            geom_id for geom_id in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith("obstacle_")
        }

    def flags(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        feet = np.zeros(2, dtype=bool)
        hands = np.zeros(2, dtype=bool)
        nonfoot = False
        obstacle = False
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if contact.geom1 in self.obstacles or contact.geom2 in self.obstacles:
                other = contact.geom2 if contact.geom1 in self.obstacles else contact.geom1
                # A static obstacle can only be in contact with the robot here; retaining the
                # body test avoids treating accidental obstacle-obstacle scene overlap as a
                # robot safety event.
                if int(self.model.geom_bodyid[other]) != 0:
                    obstacle = True
            if contact.geom1 == self.floor:
                other = contact.geom2
            elif contact.geom2 == self.floor:
                other = contact.geom1
            else:
                continue
            body = int(self.model.geom_bodyid[other])
            if body in self.feet:
                feet[self.feet[body]] = True
            elif body in self.hand_sides:
                hands[self.hand_sides[body]] = True
            else:
                nonfoot = True
        return feet, hands, nonfoot, obstacle


def _roll_degrees(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = quat.T
    return np.degrees(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))


class SeedReplayRunner:
    """One isolated MuJoCo + frozen SONIC instance, safe to own in one worker process."""

    def __init__(self, scene_path: Path = C.FLAT_SCENE):
        self.env = G1FlatEnv(scene_path)
        self.controller = SonicController()
        self.contacts = _ContactMonitor(self.env.model)
        joint_ids = self.env.model.actuator_trnid[self.env.body_act, 0]
        self.reference_lower = self.env.model.jnt_range[joint_ids, 0][C.MUJOCO_TO_ISAACLAB]
        self.reference_upper = self.env.model.jnt_range[joint_ids, 1][C.MUJOCO_TO_ISAACLAB]

    def replay(self, motion: SeedMotion, cfg: ReplayConfig, *, stratum: str,
               initial_base_height_m: float | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        # Starting at the first kinematic pose avoids treating every source clip as a large
        # static posture jump.  The warmup then lets the policy history fill while the first
        # reference frame remains fixed.
        if initial_base_height_m is None:
            self.env.reset(joint_offset=motion.joint_pos_hw[0] - C.DEFAULT_ANGLES)
        else:
            if not np.isfinite(initial_base_height_m) or initial_base_height_m <= 0.0:
                raise ValueError("initial_base_height_m must be finite and positive")
            self.env.reset(height=float(initial_base_height_m),
                           joint_offset=motion.joint_pos_hw[0] - C.DEFAULT_ANGLES)
        self.controller.reset()
        reference = motion.reference()
        warmup_ticks = int(round(cfg.warmup_seconds / C.CONTROL_DT))

        log: dict[str, list[Any]] = {key: [] for key in (
            "t", "source_frame", "q_ref", "dq_ref", "root_ref_pos_seed_m", "root_ref_quat",
            "q_exec", "dq_exec", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel",
            "action", "foot_contact", "hand_contact", "nonfoot_floor_contact", "obstacle_contact",
        )}
        for tick in range(warmup_ticks + motion.T):
            state = self.env.state()
            self.controller.append_state(state["q_hw"], state["dq_hw"], state["base_quat"],
                                         state["base_ang_vel"])
            action, q_target, _ = self.controller.act(reference, state["base_quat"])
            if tick >= warmup_ticks:
                frame = reference.cursor
                log["source_frame"].append(motion.source_frames[frame])
                log["q_ref"].append(reference.joint_pos[frame].copy())
                log["dq_ref"].append(reference.joint_vel[frame].copy())
                log["root_ref_pos_seed_m"].append(reference.root_pos[frame].copy())
                log["root_ref_quat"].append(reference.root_quat[frame].copy())
                log["action"].append(action.copy())
            self.env.set_target(q_target)
            self.env.step()
            if tick < warmup_ticks:
                continue

            executed = self.env.state()
            feet, hands, nonfoot, obstacle = self.contacts.flags(self.env.data)
            log["t"].append(self.env.time)
            log["q_exec"].append(executed["q_hw"][C.MUJOCO_TO_ISAACLAB].copy())
            log["dq_exec"].append(executed["dq_hw"][C.MUJOCO_TO_ISAACLAB].copy())
            log["base_pos"].append(executed["base_pos"].copy())
            log["base_quat"].append(executed["base_quat"].copy())
            log["base_lin_vel"].append(executed["base_lin_vel"].copy())
            log["base_ang_vel"].append(executed["base_ang_vel"].copy())
            log["foot_contact"].append(feet)
            log["hand_contact"].append(hands)
            log["nonfoot_floor_contact"].append(nonfoot)
            log["obstacle_contact"].append(obstacle)
            reference.advance()

        data = {key: np.asarray(value) for key, value in log.items()}
        return data, self._summary(data, cfg, stratum, self.reference_lower, self.reference_upper)

    @staticmethod
    def _summary(data: dict[str, np.ndarray], cfg: ReplayConfig, stratum: str,
                 reference_lower: np.ndarray, reference_upper: np.ndarray) -> dict[str, Any]:
        metric_start = min(len(data["t"]) - 1, int(round(cfg.ignore_metrics_seconds / C.CONTROL_DT)))
        view = slice(metric_start, None)
        err = np.abs(data["q_ref"] - data["q_exec"])
        roll = _roll_degrees(data["base_quat"])
        foot_ratio = data["foot_contact"][view].mean(axis=0)
        hand_ratio = data["hand_contact"][view].mean(axis=0)
        nonfoot_ratio = float(data["nonfoot_floor_contact"][view].mean())
        obstacle_ratio = float(data["obstacle_contact"][view].mean()) if "obstacle_contact" in data else 0.0
        airborne = ~np.asarray(data["foot_contact"], dtype=bool).any(axis=1)
        airborne_runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for frame, value in enumerate(airborne):
            if value and run_start is None:
                run_start = frame
            if run_start is not None and (not value or frame == len(airborne) - 1):
                stop = frame if not value else frame + 1
                airborne_runs.append((run_start, stop))
                run_start = None
        longest_airborne = max((stop - start for start, stop in airborne_runs), default=0)
        if airborne_runs:
            start, stop = max(airborne_runs, key=lambda item: item[1] - item[0])
            before = data["base_pos"][:max(start, 1), 2]
            baseline_z = float(np.median(before))
            peak_z = float(data["base_pos"][start:stop, 2].max())
            # A valid landing requires bilateral support after the airborne interval.  The
            # source can end during a flight, which is unsuitable for a motion primitive.
            landed = bool(np.any(np.asarray(data["foot_contact"], dtype=bool)[stop:].all(axis=1)))
        else:
            baseline_z = float(np.median(data["base_pos"][:max(metric_start, 1), 2]))
            peak_z, landed = baseline_z, False
        jump_lift = peak_z - baseline_z
        ref_out_of_range = ((data["q_ref"] < reference_lower[None, :]) |
                            (data["q_ref"] > reference_upper[None, :]))
        reference_path = float(np.linalg.norm(np.diff(data["root_ref_pos_seed_m"][:, :2], axis=0), axis=1).sum())
        executed_path = float(np.linalg.norm(np.diff(data["base_pos"][:, :2], axis=0), axis=1).sum())
        progress_ratio = executed_path / max(reference_path, 1e-8)
        fell = bool(data["base_pos"][view, 2].min() < cfg.fall_height) or bool(np.abs(roll[view]).max() > 60.0)

        checks: list[str] = []
        if fell:
            checks.append("fall_or_extreme_roll")
        if float(err[view].mean()) > cfg.max_track_err_rad:
            checks.append("tracking_error")
        if float(err[view, :12].mean()) > cfg.max_leg_track_err_rad:
            checks.append("leg_tracking_error")
        if float(np.abs(roll[view]).max()) > cfg.max_roll_deg:
            checks.append("roll_limit")
        if bool(ref_out_of_range.any()):
            checks.append("reference_joint_limit")
        if obstacle_ratio > 0.0:
            checks.append("scene_obstacle_contact")
        if stratum == "jump":
            if longest_airborne < cfg.jump_min_airborne_ticks:
                checks.append("jump_no_sustained_takeoff")
            if jump_lift < cfg.jump_min_lift_m:
                checks.append("jump_insufficient_lift")
            if not landed:
                checks.append("jump_no_verified_landing")

        # Crawl/all-fours references legitimately ask for hand and other low-body contacts.
        # Normal walk and crouch clips must retain foot support and stay off non-foot bodies.
        low_contact_mode = stratum in {"all_fours", "crawl", "low_transition"}
        if not low_contact_mode:
            if float(foot_ratio.min()) < cfg.min_foot_contact_ratio:
                checks.append("insufficient_foot_contact")
            if nonfoot_ratio > cfg.max_nonfoot_floor_ratio:
                checks.append("nonfoot_floor_contact")
        # A pure in-place turn or posture transition has no displacement task.  For a reference
        # that does ask for appreciable planar travel, stability alone is insufficient: require
        # the frozen executor to make at least a fraction of that progress.  The SEED and
        # generated root axes are only used through their invariant planar path length.
        locomotion = stratum in {"walk_nominal", "walk_turn", "walk_lateral_reverse", "crouch",
                                 "all_fours", "crawl", "generated"}
        progress_required = locomotion and reference_path >= cfg.min_reference_planar_path_m
        if progress_required and progress_ratio < cfg.min_progress_ratio:
            checks.append("insufficient_motion_progress")

        return {
            "stratum": stratum,
            "ticks": int(len(data["t"])),
            "control_hz": float(1.0 / C.CONTROL_DT),
            "metrics_ignore_ticks": int(metric_start),
            "fell": fell,
            "track_err_mean_rad": float(err[view].mean()),
            "track_err_legs_rad": float(err[view, :12].mean()),
            "track_err_waist_rad": float(err[view, 12:15].mean()),
            "track_err_arms_rad": float(err[view, 15:].mean()),
            "base_z_min_m": float(data["base_pos"][view, 2].min()),
            "base_z_mean_m": float(data["base_pos"][view, 2].mean()),
            "roll_abs_max_deg": float(np.abs(roll[view]).max()),
            "foot_contact_ratio": [float(x) for x in foot_ratio],
            "hand_contact_ratio": [float(x) for x in hand_ratio],
            "nonfoot_floor_ratio": nonfoot_ratio,
            "obstacle_contact_ratio": obstacle_ratio,
            "airborne_ticks_max": int(longest_airborne),
            "jump_lift_m": float(jump_lift),
            "jump_landing_verified": bool(landed),
            "reference_joint_limit_violations": int(ref_out_of_range.sum()),
            "reference_joint_limit_frames": int(ref_out_of_range.any(axis=1).sum()),
            "reference_joint_limit_names": [C.MOTOR_NAMES[index] for index in
                                              np.flatnonzero(ref_out_of_range.any(axis=0))],
            "reference_planar_path_m": reference_path,
            "exec_path_m": executed_path,
            "motion_progress_ratio": progress_ratio,
            "motion_progress_required": progress_required,
            "ref_root_displacement_seed_m": [float(x) for x in
                                               (data["root_ref_pos_seed_m"][-1] - data["root_ref_pos_seed_m"][0])],
            "accepted": not checks,
            "failed_checks": checks,
        }


def _manifest_rows(path: Path, data_root: Path, selected_ids: set[str], strata: set[str], limit: int,
                   per_stratum_limit: int, selection_seed: int) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if not row.get("motion_id") or not row.get("move_g1_path") or not row.get("pilot_stratum"):
            raise ValueError(f"{path} must contain motion_id, move_g1_path and pilot_stratum")
    if selected_ids:
        rows = [row for row in rows if row["motion_id"] in selected_ids]
    if strata:
        rows = [row for row in rows if row["pilot_stratum"] in strata]
    if per_stratum_limit:
        selected, counts = [], {}
        # The manifest is path-sorted for reproducibility, which would make a prefix highly
        # correlated by capture day/actor.  Hash order provides a stable, seed-controlled
        # pilot draw without changing the manifest or relying on process scheduling.
        ordered = sorted(rows, key=lambda row: hashlib.sha256(
            f"{selection_seed}:{row['motion_id']}".encode("utf-8")).digest())
        for row in ordered:
            key = row["pilot_stratum"]
            if counts.get(key, 0) >= per_stratum_limit:
                continue
            selected.append(row)
            counts[key] = counts.get(key, 0) + 1
        rows = selected
    # A capability slice may be extracted incrementally from the large HF tarball and invoked
    # with --motion-id.  Validate only the rows selected for this run; checking every manifest
    # member before applying that filter made valid partial replays impossible.
    for row in rows:
        if not (data_root / row["move_g1_path"]).is_file():
            raise FileNotFoundError(f"manifest path is missing: {data_root / row['move_g1_path']}")
    return rows[:limit] if limit > 0 else rows


def _save_result(out: Path, row: dict[str, str], data: dict[str, np.ndarray], summary: dict[str, Any], cfg: ReplayConfig) -> dict[str, Any]:
    out = Path(out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "summaries").mkdir(parents=True, exist_ok=True)
    motion_id = row["motion_id"]
    record = out / "records" / f"{motion_id}.npz"
    np.savez_compressed(record, **data)
    summary = dict(summary)
    summary.update({
        "motion_id": motion_id,
        "source_csv": row["move_g1_path"],
        "source_move_name": row.get("move_name", ""),
        "record": str(record.relative_to(out)),
        "replay_config": asdict(cfg),
    })
    (out / "summaries" / f"{motion_id}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


_WORKER: SeedReplayRunner | None = None
_WORKER_DATA_ROOT: Path | None = None
_WORKER_OUT: Path | None = None
_WORKER_CFG: ReplayConfig | None = None
_WORKER_EULER_SEQUENCE = "XYZ"


def _init_worker(data_root: str, out: str, config: dict[str, Any], euler_sequence: str) -> None:
    global _WORKER, _WORKER_DATA_ROOT, _WORKER_OUT, _WORKER_CFG, _WORKER_EULER_SEQUENCE
    _WORKER = SeedReplayRunner()
    _WORKER_DATA_ROOT = Path(data_root)
    _WORKER_OUT = Path(out)
    _WORKER_CFG = ReplayConfig(**config)
    _WORKER_EULER_SEQUENCE = euler_sequence


def _run_row(row: dict[str, str]) -> dict[str, Any]:
    assert _WORKER is not None and _WORKER_DATA_ROOT is not None and _WORKER_OUT is not None and _WORKER_CFG is not None
    try:
        motion = SeedMotion.from_csv(_WORKER_DATA_ROOT / row["move_g1_path"],
                                     euler_sequence=_WORKER_EULER_SEQUENCE)
        data, summary = _WORKER.replay(motion, _WORKER_CFG, stratum=row["pilot_stratum"])
        return _save_result(_WORKER_OUT, row, data, summary, _WORKER_CFG)
    except Exception as error:  # keep a long batch resumable if one source CSV is malformed
        return {
            "motion_id": row["motion_id"],
            "stratum": row["pilot_stratum"],
            "accepted": False,
            "failed_checks": ["replay_error"],
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def _write_error_summary(out: Path, result: dict[str, Any]) -> None:
    path = Path(out) / "summaries"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{result['motion_id']}.json").write_text(json.dumps(result, indent=2) + "\n")


def _batch_summary(out: Path, rows: Iterable[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for row in rows:
        path = Path(out) / "summaries" / f"{row['motion_id']}.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text()))
    accepted = [item for item in summaries if item.get("accepted")]
    counts: dict[str, dict[str, int]] = {}
    for item in summaries:
        key = item.get("stratum", "unknown")
        entry = counts.setdefault(key, {"replayed": 0, "accepted": 0})
        entry["replayed"] += 1
        entry["accepted"] += int(bool(item.get("accepted")))
    result = {
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "requested": len(list(rows)),
        "replayed": len(summaries),
        "accepted": len(accepted),
        "acceptance_rate": (len(accepted) / len(summaries)) if summaries else 0.0,
        "by_stratum": counts,
        "args": {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items()},
    }
    if accepted:
        result["accepted_track_err_mean_rad"] = float(np.mean([x["track_err_mean_rad"] for x in accepted]))
    return result


def main() -> int:
    default_root = Path("data/seed_stage2_pilot")
    parser = argparse.ArgumentParser(description="Replay BONES-SEED G1 clips through frozen SONIC")
    parser.add_argument("--manifest", type=Path, default=default_root / "seed_stage2_pilot_manifest_v004.csv")
    parser.add_argument("--data-root", type=Path, default=default_root,
                        help="directory containing g1/csv/... from the selected archive")
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/seed_replay"))
    parser.add_argument("--motion-id", nargs="*", default=[], help="specific manifest motion IDs")
    parser.add_argument("--strata", nargs="*", default=[], help="one or more pilot_stratum values")
    parser.add_argument("--limit", type=int, default=0, help="maximum selected rows (0 = all)")
    parser.add_argument("--per-stratum-limit", type=int, default=0,
                        help="cap each selected primitive stratum before applying --limit (0 = no cap)")
    parser.add_argument("--selection-seed", type=int, default=20260917,
                        help="stable hash seed used with --per-stratum-limit")
    parser.add_argument("--jobs", type=int, default=1, help="MuJoCo/ONNX worker processes")
    parser.add_argument("--no-resume", action="store_true", help="replay even when a summary already exists")
    parser.add_argument("--euler-sequence", choices=("XYZ", "xyz", "ZYX", "zyx"), default="XYZ")
    parser.add_argument("--warmup-seconds", type=float, default=ReplayConfig.warmup_seconds)
    parser.add_argument("--ignore-metrics-seconds", type=float, default=ReplayConfig.ignore_metrics_seconds)
    parser.add_argument("--fall-height", type=float, default=ReplayConfig.fall_height)
    parser.add_argument("--max-roll-deg", type=float, default=ReplayConfig.max_roll_deg)
    parser.add_argument("--max-track-err", type=float, default=ReplayConfig.max_track_err_rad)
    parser.add_argument("--max-leg-track-err", type=float, default=ReplayConfig.max_leg_track_err_rad)
    parser.add_argument("--min-foot-contact", type=float, default=ReplayConfig.min_foot_contact_ratio)
    parser.add_argument("--max-nonfoot-floor", type=float, default=ReplayConfig.max_nonfoot_floor_ratio)
    parser.add_argument("--min-reference-planar-path", type=float, default=ReplayConfig.min_reference_planar_path_m,
                        help="only reference paths at least this long require motion completion")
    parser.add_argument("--min-progress-ratio", type=float, default=ReplayConfig.min_progress_ratio,
                        help="minimum executed/reference planar path ratio for locomotion")
    parser.add_argument("--jump-min-lift", type=float, default=ReplayConfig.jump_min_lift_m,
                        help="minimum measured pelvis lift for admitting a jump primitive")
    parser.add_argument("--jump-min-airborne-ticks", type=int, default=ReplayConfig.jump_min_airborne_ticks,
                        help="minimum consecutive 50 Hz ticks with neither foot supported")
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be positive")
    config = ReplayConfig(
        warmup_seconds=args.warmup_seconds,
        ignore_metrics_seconds=args.ignore_metrics_seconds,
        fall_height=args.fall_height,
        max_roll_deg=args.max_roll_deg,
        max_track_err_rad=args.max_track_err,
        max_leg_track_err_rad=args.max_leg_track_err,
        min_foot_contact_ratio=args.min_foot_contact,
        max_nonfoot_floor_ratio=args.max_nonfoot_floor,
        min_reference_planar_path_m=args.min_reference_planar_path,
        min_progress_ratio=args.min_progress_ratio,
        jump_min_lift_m=args.jump_min_lift,
        jump_min_airborne_ticks=args.jump_min_airborne_ticks,
    )
    if args.limit < 0 or args.per_stratum_limit < 0 or args.jump_min_lift < 0 or args.jump_min_airborne_ticks < 1:
        parser.error("limits must be non-negative and jump-min-airborne-ticks must be positive")
    rows = _manifest_rows(args.manifest, args.data_root, set(args.motion_id), set(args.strata), args.limit,
                          args.per_stratum_limit, args.selection_seed)
    if not rows:
        parser.error("the selected manifest query has no rows")
    args.out.mkdir(parents=True, exist_ok=True)
    todo = rows if args.no_resume else [row for row in rows if not (args.out / "summaries" / f"{row['motion_id']}.json").is_file()]
    print(f"selected {len(rows)} clips; replaying {len(todo)} with {args.jobs} worker(s)", flush=True)

    if todo:
        worker_args = (str(args.data_root), str(args.out), asdict(config), args.euler_sequence)
        if args.jobs == 1:
            _init_worker(*worker_args)
            results = map(_run_row, todo)
        else:
            context = mp.get_context("spawn")
            pool = context.Pool(processes=min(args.jobs, len(todo)), initializer=_init_worker, initargs=worker_args)
            results = pool.imap_unordered(_run_row, todo)
        try:
            for i, result in enumerate(results, start=1):
                if "error" in result:
                    _write_error_summary(args.out, result)
                    verdict = result["error"]
                else:
                    verdict = "accepted" if result["accepted"] else ",".join(result["failed_checks"])
                print(f"[{i:4d}/{len(todo)}] {result['motion_id']} {result['stratum']}: {verdict}", flush=True)
        finally:
            if args.jobs > 1:
                pool.close()
                pool.join()

    summary = _batch_summary(args.out, rows, args)
    (args.out / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
