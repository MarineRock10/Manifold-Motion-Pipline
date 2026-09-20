"""Turn accepted SONIC replays into leakage-safe Stage-2 trajectory windows.

The replay stage records the source reference ``R_ref`` and the MuJoCo execution
``R_exec`` at SONIC's 50 Hz tick.  This command turns accepted records into a compact
30 Hz supervised dataset.  It deliberately keeps both trajectories:

* ``target_ref`` is the future reference a dynamic model must generate for SONIC;
* ``target_exec`` is the future behaviour used to evaluate whether that reference was
  actually executable.

The selected SEED clips were recorded on flat ground.  By default this command therefore uses
the explicit ``reverse_corridor`` mode: a body-envelope-consistent safe corridor and SDF are
reconstructed around successful execution.  They are labelled as synthetic reverse data, never
as a sensor-derived map.  ``--environment-mode flat`` preserves the original neutral condition
for an ablation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import constants as C
from .corridor import CorridorConfig, ExecutedEnvelopeEstimator, corridor_from_window


PRIMITIVE_NAMES = (
    "all_fours", "crawl", "crouch", "low_transition",
    "walk_lateral_reverse", "walk_nominal", "walk_turn", "jump",
)
PRIMITIVE_TO_ID = {name: index for index, name in enumerate(PRIMITIVE_NAMES)}
# [forward, lateral, vertical half-widths, yaw, pitch, roll].  This only denotes the
# open flat scene used during replay; it is intentionally not claimed to be M^E.
FLAT_MANIFOLD = np.array([2.0, 2.0, 1.5, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class WindowConfig:
    output_hz: float = 30.0
    horizon_seconds: float = 1.60
    history_seconds: float = 0.40
    stride_seconds: float = 0.20

    @property
    def horizon_frames(self) -> int:
        return int(round(self.horizon_seconds * self.output_hz))

    @property
    def history_frames(self) -> int:
        return int(round(self.history_seconds * self.output_hz))

    @property
    def stride_frames(self) -> int:
        return max(1, int(round(self.stride_seconds * self.output_hz)))


def _linear_resample(values: np.ndarray, source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return np.interp(target_t, source_t, values)
    flat = values.reshape(len(values), -1)
    result = np.stack([np.interp(target_t, source_t, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return result.reshape((len(target_t),) + values.shape[1:])


def _quat_resample(values: np.ndarray, source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """Normalized interpolation is sufficient for the small 50->30 Hz interval here."""
    quat = _linear_resample(values, source_t, target_t)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return quat


def _relative_pose(pos: np.ndarray, quat: np.ndarray, origin: int, future: slice) -> tuple[np.ndarray, np.ndarray]:
    """Future root pose in the root-local frame at ``origin`` (xyz + 6D rotation)."""
    p0 = pos[origin]
    q0 = quat[origin]
    rot0 = C.quat_to_matrix(q0)
    delta_local = (pos[future] - p0) @ rot0
    orientations = []
    for q in quat[future]:
        relative = C.quat_mul(C.quat_conj(q0), q)
        rot = C.quat_to_matrix(relative)
        orientations.append([rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], rot[2, 0], rot[2, 1]])
    return delta_local, np.asarray(orientations, dtype=np.float64)


def _gravity_local(quat: np.ndarray) -> np.ndarray:
    return np.asarray([C.quat_rotate(C.quat_conj(q), np.array([0.0, 0.0, -1.0])) for q in quat])


def _state_features(data: dict[str, np.ndarray]) -> np.ndarray:
    """Executed state given to the dynamic model: 69 floats at each timestamp."""
    gravity = _gravity_local(data["base_quat"])
    local_velocity = np.asarray([C.quat_rotate(C.quat_conj(q), velocity)
                                 for q, velocity in zip(data["base_quat"], data["base_lin_vel"])])
    return np.concatenate([
        data["q_exec"], data["dq_exec"], gravity, local_velocity,
        data["foot_contact"].astype(np.float64), data["hand_contact"].astype(np.float64),
        data["nonfoot_floor_contact"].astype(np.float64)[:, None],
    ], axis=1)


def _actor_split(actor_uid: str) -> int:
    """Actor-disjoint split ID: 0=train, 1=validation, 2=test."""
    bucket = hashlib.sha256(actor_uid.encode("utf-8")).digest()[0] % 10
    return 0 if bucket < 8 else (1 if bucket == 8 else 2)


def _manifest(path: Path) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = {row["motion_id"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def _accepted_record_ids(replay_root: Path) -> list[str]:
    records = []
    for summary_path in sorted((Path(replay_root) / "summaries").glob("*.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("accepted") and (Path(replay_root) / summary.get("record", "")).is_file():
            records.append(str(summary["motion_id"]))
    return records


def _resample_record(data: dict[str, np.ndarray], output_hz: float) -> dict[str, np.ndarray]:
    source_t = np.asarray(data["t"], dtype=np.float64)
    if len(source_t) < 2 or np.any(np.diff(source_t) <= 0):
        raise ValueError("record has an invalid time vector")
    target_t = np.arange(source_t[0], source_t[-1] + 1e-9, 1.0 / output_hz)
    if len(target_t) < 2:
        raise ValueError("record is too short after resampling")
    linear = ("q_ref", "dq_ref", "root_ref_pos_seed_m", "q_exec", "dq_exec", "base_pos",
              "base_lin_vel", "base_ang_vel", "action", "source_frame")
    quat = ("root_ref_quat", "base_quat")
    output: dict[str, np.ndarray] = {"t": target_t}
    for key in linear:
        output[key] = _linear_resample(data[key], source_t, target_t)
    for key in quat:
        output[key] = _quat_resample(data[key], source_t, target_t)
    for key in ("foot_contact", "hand_contact", "nonfoot_floor_contact"):
        nearest = np.searchsorted(source_t, target_t, side="left").clip(0, len(source_t) - 1)
        before = np.maximum(nearest - 1, 0)
        nearest = np.where(np.abs(source_t[nearest] - target_t) < np.abs(source_t[before] - target_t), nearest, before)
        output[key] = data[key][nearest]
    return output


def _window_keys(environment_mode: str) -> tuple[str, ...]:
    base = ("state", "history", "primitive", "manifold", "command", "target_ref", "target_exec", "split", "clip_index", "source_origin")
    return base if environment_mode == "flat" else base + ("corridor", "sdf")


def _windows_for_record(data: dict[str, np.ndarray], primitive: int, actor_split: int,
                        clip_index: int, cfg: WindowConfig, *, environment_mode: str,
                        corridor_config: CorridorConfig | None, clip_seed: int) -> dict[str, list[np.ndarray]]:
    state = _state_features(data)
    horizon, history = cfg.horizon_frames, cfg.history_frames
    keys = _window_keys(environment_mode)
    if len(state) < history + horizon + 1:
        return {key: [] for key in keys}
    if environment_mode == "reverse_corridor" and (corridor_config is None or "envelope_semi" not in data):
        raise ValueError("reverse_corridor windows require a corridor config and executed body envelopes")
    output: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for origin in range(history - 1, len(state) - horizon, cfg.stride_frames):
        future = slice(origin + 1, origin + 1 + horizon)
        ref_pos, ref_rot6 = _relative_pose(data["root_ref_pos_seed_m"], data["root_ref_quat"], origin, future)
        exec_pos, exec_rot6 = _relative_pose(data["base_pos"], data["base_quat"], origin, future)
        target_ref = np.concatenate([data["q_ref"][future], ref_pos, ref_rot6], axis=1)
        target_exec = np.concatenate([data["q_exec"][future], exec_pos, exec_rot6], axis=1)
        # In deployment this comes from the navigation layer: desired displacement and facing
        # at the planning horizon.  It is derived from R_ref here solely to supervise the
        # condition on this offline flat-ground corpus.
        command = np.concatenate([ref_pos[-1], ref_rot6[-1]])
        output["state"].append(state[origin])
        output["history"].append(state[origin - history + 1:origin + 1])
        output["primitive"].append(np.asarray(primitive, dtype=np.int64))
        output["manifold"].append(FLAT_MANIFOLD.copy())
        if environment_mode == "reverse_corridor":
            corridor, sdf = corridor_from_window(
                data["base_pos"], data["base_quat"], data["envelope_semi"],
                origin, future, seed=clip_seed + origin, config=corridor_config)
            output["corridor"].append(corridor)
            output["sdf"].append(sdf)
        output["command"].append(command)
        output["target_ref"].append(target_ref)
        output["target_exec"].append(target_exec)
        output["split"].append(np.asarray(actor_split, dtype=np.uint8))
        output["clip_index"].append(np.asarray(clip_index, dtype=np.int32))
        output["source_origin"].append(np.asarray(origin, dtype=np.int32))
    return output


def _concat(parts: dict[str, list[np.ndarray]], key: str, dtype: np.dtype) -> np.ndarray:
    values = parts[key]
    if not values:
        raise ValueError("no windows were produced; replay more accepted clips or shorten the window")
    return np.asarray(values, dtype=dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build actor-disjoint Stage-2 windows from accepted SEED replays")
    parser.add_argument("--replay-root", type=Path, default=Path("reports/manifold_motion/seed_replay"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/seed_stage2_pilot/seed_stage2_pilot_manifest_v004.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/seed_windows"))
    parser.add_argument("--output-hz", type=float, default=WindowConfig.output_hz)
    parser.add_argument("--horizon-seconds", type=float, default=WindowConfig.horizon_seconds)
    parser.add_argument("--history-seconds", type=float, default=WindowConfig.history_seconds)
    parser.add_argument("--stride-seconds", type=float, default=WindowConfig.stride_seconds)
    parser.add_argument("--limit-clips", type=int, default=0, help="debug limit (0 = every accepted record)")
    parser.add_argument("--environment-mode", choices=("flat", "reverse_corridor"), default="reverse_corridor")
    parser.add_argument("--corridor-clearance-min", type=float, default=CorridorConfig.clearance_min_m)
    parser.add_argument("--corridor-clearance-max", type=float, default=CorridorConfig.clearance_max_m)
    parser.add_argument("--sdf-shape", type=int, nargs=3, default=CorridorConfig.sdf_shape,
                        metavar=("NX", "NY", "NZ"))
    parser.add_argument("--corridor-seed", type=int, default=20260917)
    parser.add_argument("--primitive-override", choices=PRIMITIVE_NAMES, default=None,
                        help="explicit semantic label for every selected clip (keeps source label in metadata)")
    args = parser.parse_args()
    cfg = WindowConfig(args.output_hz, args.horizon_seconds, args.history_seconds, args.stride_seconds)
    if min(cfg.output_hz, cfg.horizon_seconds, cfg.history_seconds, cfg.stride_seconds) <= 0:
        parser.error("window rates and durations must be positive")
    corridor_config = None
    if args.environment_mode == "reverse_corridor":
        try:
            corridor_config = CorridorConfig(args.corridor_clearance_min, args.corridor_clearance_max,
                                             tuple(args.sdf_shape))
            if min(corridor_config.sdf_shape) < 2:
                raise ValueError("each SDF dimension must be at least 2")
            if not (0 <= corridor_config.clearance_min_m <= corridor_config.clearance_max_m):
                raise ValueError("clearance must satisfy 0 <= min <= max")
        except ValueError as error:
            parser.error(str(error))

    manifest = _manifest(args.manifest)
    motion_ids = _accepted_record_ids(args.replay_root)
    if args.limit_clips:
        motion_ids = motion_ids[:args.limit_clips]
    if not motion_ids:
        parser.error("no accepted replay records found")

    keys = _window_keys(args.environment_mode)
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    clips: list[dict[str, str]] = []
    envelope_estimator = ExecutedEnvelopeEstimator() if args.environment_mode == "reverse_corridor" else None
    for motion_number, motion_id in enumerate(motion_ids, start=1):
        row = manifest.get(motion_id)
        if row is None:
            raise KeyError(f"accepted replay {motion_id} is absent from {args.manifest}")
        source_primitive_name = row["pilot_stratum"]
        primitive_name = args.primitive_override or source_primitive_name
        if primitive_name not in PRIMITIVE_TO_ID:
            raise ValueError(f"unknown primitive '{primitive_name}' for {motion_id}")
        path = Path(args.replay_root) / "records" / f"{motion_id}.npz"
        with np.load(path) as record:
            data = {key: record[key] for key in record.files}
        data = _resample_record(data, cfg.output_hz)
        if envelope_estimator is not None:
            data["envelope_semi"] = envelope_estimator.sequence(data["q_exec"], data["base_pos"], data["base_quat"])
        clip_index = len(clips)
        windows = _windows_for_record(data, PRIMITIVE_TO_ID[primitive_name],
                                      _actor_split(row.get("actor_uid", "unknown")), clip_index, cfg,
                                      environment_mode=args.environment_mode, corridor_config=corridor_config,
                                      clip_seed=int.from_bytes(hashlib.sha256(
                                          f"{args.corridor_seed}:{motion_id}".encode("utf-8")).digest()[:8], "little"))
        count = len(windows["state"])
        if count:
            for key in pieces:
                pieces[key].extend(windows[key])
            clips.append({"motion_id": motion_id, "actor_uid": row.get("actor_uid", ""),
                          "primitive": primitive_name, "source_primitive": source_primitive_name,
                          "windows": str(count)})
        if args.environment_mode == "reverse_corridor":
            print(f"[{motion_number:4d}/{len(motion_ids)}] {motion_id} {primitive_name}: {count} windows", flush=True)

    arrays = {
        "state": _concat(pieces, "state", np.float32),
        "history": _concat(pieces, "history", np.float32),
        "primitive": _concat(pieces, "primitive", np.int64),
        "manifold": _concat(pieces, "manifold", np.float32),
        "command": _concat(pieces, "command", np.float32),
        "target_ref": _concat(pieces, "target_ref", np.float32),
        "target_exec": _concat(pieces, "target_exec", np.float32),
        "split": _concat(pieces, "split", np.uint8),
        "clip_index": _concat(pieces, "clip_index", np.int32),
        "source_origin": _concat(pieces, "source_origin", np.int32),
        # Kept inside the NPZ (rather than inferred from the selected subset) so a jump-only
        # archive still has the same one-hot width as a future merged router/model dataset.
        "primitive_count": np.asarray(len(PRIMITIVE_NAMES), dtype=np.int64),
    }
    if args.environment_mode == "reverse_corridor":
        arrays["corridor"] = _concat(pieces, "corridor", np.float32)
        arrays["sdf"] = _concat(pieces, "sdf", np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "seed_stage2_windows.npz", **arrays)
    split_counts = {name: int((arrays["split"] == value).sum())
                    for value, name in enumerate(("train", "validation", "test"))}
    metadata: dict[str, Any] = {
        "config": asdict(cfg),
        "primitive_names": PRIMITIVE_NAMES,
        "environment": {
            "mode": args.environment_mode,
            "legacy_flat_manifold": FLAT_MANIFOLD.tolist(),
            "corridor_config": asdict(corridor_config) if corridor_config else None,
            "provenance": ("reverse_synthesized_from_R_exec" if args.environment_mode == "reverse_corridor"
                           else "flat_placeholder"),
        },
        "condition": {"state_dim": int(arrays["state"].shape[1]),
                      "history_shape": list(arrays["history"].shape[1:]),
                      "primitive_encoding": "integer index into primitive_names",
                      "primitive_override": args.primitive_override,
                      "manifold_dim": int(arrays["manifold"].shape[1]),
                      "corridor_shape": list(arrays["corridor"].shape[1:]) if "corridor" in arrays else None,
                      "sdf_shape": list(arrays["sdf"].shape[1:]) if "sdf" in arrays else None,
                      "source_origin": "30 Hz index in the accepted replay record, for traceability",
                      "command_dim": int(arrays["command"].shape[1])},
        "target": {"shape": list(arrays["target_ref"].shape[1:]),
                   "layout_per_frame": "29 q_ref(policy order) + root xyz local + root rotation 6D"},
        "windows": int(len(arrays["state"])),
        "split_counts": split_counts,
        "clips": clips,
        "warning": ("Corridor/SDF labels are reverse-synthesized from successful execution, not sensor-derived maps."
                    if args.environment_mode == "reverse_corridor"
                    else "manifold is a neutral flat-ground placeholder; it is not obstacle conditioning."),
    }
    (args.out / "seed_stage2_windows_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"windows": metadata["windows"], "split_counts": split_counts,
                      "clips_used": len(clips), "target_shape": metadata["target"]["shape"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
