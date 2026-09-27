"""Evaluate a same-state environment counterfactual for the latent prior.

The state, history, command and requested skill family are held fixed.  Only the normalized
environment block (M_e corridor/SDF/M_self) is replaced by another held-out window.  This is the
minimal causal check that the latent decoder uses environment geometry rather than merely
replaying a family label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from manifold_motion.dataio.seed_replay import ReplayConfig, SeedReplayRunner
from manifold_motion.stage2.flow import Normalizer, WindowData, _environment_vector, _target_from_model, _torch_load
from manifold_motion.stage2.latent_prior import StateConditionedSkillVAE
from manifold_motion.stage2.projection import project_reference
from manifold_motion.stage2.validate import validate_trajectory
from manifold_motion.planning.corridor import ExecutedEnvelopeEstimator


def _condition_with_environment(data: WindowData, base_index: int, environment_index: int,
                                family_id: int) -> np.ndarray:
    condition = data.condition_for_primitive(base_index, family_id)
    start = data.raw["state"].shape[1] + int(np.prod(data.raw["history"].shape[1:])) + data.primitive_count
    raw = {key: data.raw[key][environment_index:environment_index + 1]
           for key in ("state", "manifold", "corridor", "sdf", "self_manifold")}
    environment = _environment_vector(raw)[0]
    condition[start:start + len(environment)] = data.normalizer.manifold(environment)
    return condition


def _decode(model: StateConditionedSkillVAE, data: WindowData, condition: np.ndarray,
            corridor: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        value = torch.as_tensor(condition[None], device=device)
        mean, _ = model.prior_stats(value)
        decoded = model.decode(mean, value).cpu().numpy().reshape(data.target_shape)
    trajectory = _target_from_model(data.normalizer.inverse_target(decoded),
                                    data.joint_lower, data.joint_upper)
    return project_reference(trajectory, corridor)[0]


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    checkpoint = _torch_load(args.prior_checkpoint, device)
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    target_field = str(checkpoint.get("model_target_field", "target_ref"))
    data = WindowData.load(args.windows, normalizer=normalizer, model_target_field=target_field)
    model = StateConditionedSkillVAE(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    for index in (args.base_index, args.environment_index):
        if not 0 <= index < len(data.raw["state"]):
            raise ValueError(f"window index {index} is outside the archive")
    runner = SeedReplayRunner()
    envelope = ExecutedEnvelopeEstimator()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = {}
    for name, environment_index in (("base_environment", args.base_index),
                                     ("counterfactual_environment", args.environment_index)):
        condition = _condition_with_environment(data, args.base_index, environment_index, args.family_id)
        corridor = np.asarray(data.raw["corridor"][environment_index], dtype=np.float32)
        trajectory = _decode(model, data, condition, corridor, device)
        sample_path = args.out / f"{name}.npz"
        np.savez_compressed(sample_path, generated_ref=trajectory,
                            condition_corridor=corridor,
                            base_state_index=np.asarray(args.base_index),
                            environment_index=np.asarray(environment_index),
                            primitive_id=np.asarray(args.family_id))
        executed, summary = validate_trajectory(
            trajectory, source=sample_path, source_hz=30.0, config=ReplayConfig(),
            corridor=corridor, max_corridor_radius=args.max_corridor_radius,
            stratum=f"latent_counterfactual_{name}", runner=runner)
        np.savez_compressed(args.out / f"{name}_executed.npz", **executed)
        self_semi = envelope.sequence(executed["q_exec"], executed["base_pos"], executed["base_quat"])
        cases[name] = {
            "sample": str(sample_path), "environment_index": int(environment_index),
            "summary": summary,
            "mean_self_manifold": self_semi.mean(axis=0).tolist(),
            "min_self_manifold": self_semi.min(axis=0).tolist(),
            "base_z_min_m": float(executed["base_pos"][:, 2].min()),
        }
    base = np.load(args.out / "base_environment.npz")["generated_ref"]
    counter = np.load(args.out / "counterfactual_environment.npz")["generated_ref"]
    report = {
        "schema": "manifold-motion.latent-environment-counterfactual.v1",
        "checkpoint": str(args.prior_checkpoint), "windows": str(args.windows),
        "base_state_index": args.base_index, "environment_index": args.environment_index,
        "family_id": args.family_id, "family": str(data.raw["primitive_names"][args.family_id]),
        "held_fixed": ["state", "history", "primitive_family", "command"],
        "changed": ["M_e corridor", "corridor SDF", "M_self condition"],
        "target_field": target_field,
        "decoded_reference_delta_l2": float(np.linalg.norm(base - counter)),
        "decoded_root_delta_max_m": float(np.max(np.abs(base[:, 29:32] - counter[:, 29:32]))),
        "cases": cases,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-index", type=int, default=1278)
    parser.add_argument("--environment-index", type=int, default=0)
    parser.add_argument("--family-id", type=int, default=6)
    parser.add_argument("--max-corridor-radius", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_corridor_radius <= 0:
        parser.error("max-corridor-radius must be positive")
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
