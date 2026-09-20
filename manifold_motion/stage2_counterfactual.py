"""Run the fixed-state, fixed-command manifold intervention for Stage 2.

The normal routed demo proves that recorded conditions map to different primitives, but its
windows also differ in robot state and command.  This experiment locks one held-out walking
window's ``state``, ``history`` and ``command``.  It then replaces only its environment tensors
``(corridor, sdf)`` with aperture conditions representative of normal, low and narrow passages.
The geometry-only router is used deliberately: this is an ablation whose causal question is
"what changed because M changed?", not a replacement for the deployment ``p(z_p|M,c)`` router.

Each result is replayed in a corresponding MuJoCo scene with named physical obstacles.  Any
robot-obstacle contact is a hard failure, in addition to the usual SONIC tracking and corridor
checks.  Conditions still originate in the current reverse-synthesized pilot corpus; this module
therefore validates a *controlled simulator intervention*, not sensor generalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import constants as C
from . import stage2_flow
from .corridor import CorridorConfig, corridor_sdf
from .primitive_router import _features, _load_router, _normalize
from .seed_replay import ReplayConfig, SeedReplayRunner
from .seed_windows import PRIMITIVE_NAMES
from .stage2_validate import validate_trajectory


# Global row IDs in the checked-in v2 window archive.  The normal case is also the fixed base
# state/command window.  Crouch is presently a training actor because the pilot selection has no
# actor-disjoint crouch windows; its status remains explicit in the report.
PILOT_SCENARIOS = {
    "normal": {"condition_row": 2072, "scene": "data/g1_flat/scene_counterfactual_normal.xml", "split": "test"},
    "crouch": {"condition_row": 554, "scene": "data/g1_flat/scene_counterfactual_low.xml", "split": "train"},
    "low_transition": {"condition_row": 1015, "scene": "data/g1_flat/scene_counterfactual_low.xml", "split": "validation"},
    "side_escape": {"condition_row": 1546, "scene": "data/g1_flat/scene_counterfactual_side.xml", "split": "test"},
}


def _parse_models(values: list[str]) -> dict[int, Path]:
    models: dict[int, Path] = {}
    for value in values:
        try:
            primitive, path = value.split("=", 1)
            primitive_id = int(primitive)
        except ValueError as error:
            raise ValueError("--model must have the form PRIMITIVE_ID=CHECKPOINT") from error
        if not 0 <= primitive_id < len(PRIMITIVE_NAMES):
            raise ValueError(f"primitive ID {primitive_id} is out of range")
        if primitive_id in models:
            raise ValueError(f"duplicate model for primitive ID {primitive_id}")
        models[primitive_id] = Path(path)
    return models


def _base_row(raw: dict[str, np.ndarray], split: int, index: int) -> int:
    candidates = np.flatnonzero(raw["split"] == split)
    if not len(candidates):
        raise ValueError(f"window archive has no split {split}")
    return int(candidates[index % len(candidates)])


def _condition(raw: dict[str, np.ndarray], base_row: int, source_row: int,
               bridge_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Return an aperture intervention with a wide entry for the common start pose.

    A robot beginning in the common standing state must have room to lower itself *before*
    entering the lower aperture.  The leading elements take the base corridor; remaining
    elements are the intervened condition.  SDF is static over the short horizon and comes from
    the intervention row.
    """
    corridor = np.asarray(raw["corridor"][source_row], dtype=np.float32).copy()
    if source_row != base_row:
        # Keep the common route/task while changing the aperture.  Centres and yaw are not router
        # features (they would leak a recorded future), yet using the identical centreline lets
        # the hard corridor check assess every primitive against one physical task rather than
        # against its source clip's different displacement.
        corridor[:, :3] = raw["corridor"][base_row, :, :3]
        corridor[:, 6] = raw["corridor"][base_row, :, 6]
    # ``sdf`` is a deterministic rasterization of this exact corridor.  Reusing the source
    # grid after transplanting its aperture onto a common centreline would make the two pieces
    # of M disagree, creating an artificial out-of-distribution input.  Re-rasterize after the
    # intervention so the model sees one coherent counterfactual environment.
    return corridor, corridor_sdf(corridor, CorridorConfig())


def _router_decision(router_path: Path, corridor: np.ndarray, sdf: np.ndarray,
                     command: np.ndarray) -> tuple[int, float, list[dict[str, float | str]]]:
    router, checkpoint = _load_router(router_path)
    if bool(checkpoint.get("include_command", True)):
        raise ValueError("counterfactual routing requires a --geometry-only router checkpoint")
    raw = {"corridor": corridor[None, ...], "sdf": sdf[None, ...], "command": command[None, ...]}
    feature = _features(raw, include_command=False)
    normalized = _normalize(feature, checkpoint["feature_mean"], checkpoint["feature_std"])
    with torch.no_grad():
        probability = torch.softmax(router(torch.as_tensor(normalized)), dim=1)[0].numpy()
    active = np.asarray(checkpoint["active_primitive_ids"], dtype=np.int64)
    order = np.argsort(probability)[::-1]
    return (int(active[order[0]]), float(probability[order[0]]),
            [{"primitive": PRIMITIVE_NAMES[int(active[i])], "probability": float(probability[i])} for i in order])


def _initial_reference(q_policy: np.ndarray) -> np.ndarray:
    """The identical physical/reference start supplied to every intervention."""
    result = np.zeros(38, dtype=np.float32)
    result[:29] = q_policy.astype(np.float32)
    # Local root reference at the current pose: zero translation and identity 6-D rotation.
    result[32:38] = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    return result


def _prepare_transition(initial: np.ndarray, generated: np.ndarray, bridge_frames: int) -> np.ndarray:
    if bridge_frames <= 0:
        return np.asarray(generated, dtype=np.float32)
    alpha = np.arange(bridge_frames, dtype=np.float32) / float(bridge_frames)
    # Cubic ease-in/out avoids a discontinuous first SONIC reference when a selected primitive
    # begins from a crouched/low posture rather than the common standing state.
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    bridge = initial[None, :] + alpha[:, None] * (generated[0:1] - initial[None, :])
    return np.concatenate([bridge.astype(np.float32), np.asarray(generated, dtype=np.float32)], axis=0)


def _sample(args: argparse.Namespace, model: Path, primitive: int, condition_path: Path, out: Path) -> Path:
    sample_args = argparse.Namespace(
        windows=args.windows, autoencoder=Path("reports/manifold_motion/stage2_flow/autoencoder.pt"),
        flow=Path("reports/manifold_motion/stage2_flow/flow.pt"), out=out, split=args.base_split,
        index=args.base_index, steps=32, num_candidates=1, sampler="conditional_mean",
        mean_model=model, residual_flow=Path("reports/manifold_motion/stage2_residual_flow/residual_flow.pt"),
        seed=args.seed, device=args.device, condition_npz=condition_path, primitive_id=primitive)
    stage2_flow.sample(sample_args)
    return out / "sample.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="fixed-state/counterfactual manifold Stage-2 experiment")
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True, help="geometry-only primitive router checkpoint")
    parser.add_argument("--model", action="append", required=True, help="PRIMITIVE_ID=conditional_mean.pt")
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/stage2_counterfactual"))
    parser.add_argument("--base-split", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--base-index", type=int, default=120,
                        help="index within base split; default is the held-out normal-walk window")
    parser.add_argument("--scenarios", nargs="+", choices=tuple(PILOT_SCENARIOS), default=list(PILOT_SCENARIOS))
    parser.add_argument("--bridge-frames", type=int, default=18, help="30 Hz transition frames from the common start")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.bridge_frames < 0:
        parser.error("--bridge-frames must be non-negative")
    models = _parse_models(args.model)
    with np.load(args.windows) as archive:
        needed = {"state", "history", "command", "corridor", "sdf", "split", "target_ref", "primitive"}
        missing = needed - set(archive.files)
        if missing:
            parser.error(f"windows is missing {sorted(missing)}")
        raw = {key: np.asarray(archive[key]) for key in needed}
    base = _base_row(raw, args.base_split, args.base_index)
    fixed_q = np.asarray(raw["state"][base, :29], dtype=np.float32)
    initial = _initial_reference(fixed_q)
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.scenarios:
        spec = PILOT_SCENARIOS[name]
        source_row = int(spec["condition_row"])
        if not 0 <= source_row < len(raw["state"]):
            raise ValueError(f"{name} condition row {source_row} is outside {args.windows}")
        case_out = args.out / name
        case_out.mkdir(parents=True, exist_ok=True)
        corridor, sdf = _condition(raw, base, source_row, args.bridge_frames)
        condition_path = case_out / "condition.npz"
        np.savez_compressed(condition_path, corridor=corridor, sdf=sdf)
        primitive, confidence, probabilities = _router_decision(args.router, corridor, sdf, raw["command"][base])
        if primitive not in models:
            raise ValueError(f"{name}: router selected {PRIMITIVE_NAMES[primitive]} but no --model was supplied")
        sample_path = _sample(args, models[primitive], primitive, condition_path, case_out)
        with np.load(sample_path) as archive:
            saved = {key: np.asarray(archive[key]) for key in archive.files if key not in {"generated_ref_candidates", "candidate_indices"}}
            generated = np.asarray(archive["generated_ref"], dtype=np.float32)
        prepared = _prepare_transition(initial, generated, args.bridge_frames)
        saved["generated_ref"] = prepared
        saved["initial_q_policy"] = fixed_q
        saved["transition_bridge_frames"] = np.asarray(args.bridge_frames, dtype=np.int64)
        prepared_path = case_out / "prepared_sample.npz"
        np.savez_compressed(prepared_path, **saved)
        scene = C.REPO / str(spec["scene"])
        runner = SeedReplayRunner(scene)
        executed, summary = validate_trajectory(prepared, source=prepared_path, source_hz=30.0,
                                                 config=ReplayConfig(), corridor=corridor,
                                                 stratum=PRIMITIVE_NAMES[primitive], runner=runner)
        np.savez_compressed(case_out / "executed.npz", **executed)
        report = {
            "scenario": name, "scene": str(scene), "base_window_row": base,
            "condition_window_row": source_row, "condition_split": str(spec["split"]),
            "fixed_state_history_command": True, "fixed_initial_q_policy": fixed_q.tolist(),
            "router_checkpoint": str(args.router), "routed_primitive": PRIMITIVE_NAMES[primitive],
            "routed_primitive_id": primitive, "router_confidence": confidence,
            "probabilities": probabilities, "dynamic_checkpoint": str(models[primitive]),
            "transition_bridge_frames": args.bridge_frames, "accepted": bool(summary["accepted"]),
            "failed_checks": summary["failed_checks"], "execution": summary,
            "condition_provenance": "reverse_synthesized_pilot_M; physical MuJoCo obstacle scene is explicit",
        }
        (case_out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        print(json.dumps({key: report[key] for key in ("scenario", "routed_primitive", "router_confidence", "accepted", "failed_checks")}), flush=True)
    result = {"base_window_row": base, "base_split": args.base_split, "base_index": args.base_index,
              "all_accepted": bool(all(item["accepted"] for item in reports)), "scenarios": reports,
              "note": "The intervention holds stored state/history/command fixed and changes only M; low/side conditions use the current reverse-synthesized pilot corpus."}
    (args.out / "counterfactual_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"all_accepted": result["all_accepted"], "scenarios": len(reports)}, indent=2))
    return 0 if result["all_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
