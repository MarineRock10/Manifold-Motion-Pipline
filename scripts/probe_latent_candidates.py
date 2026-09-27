"""Probe multiple environment-conditioned latent samples through the physical corridor gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from manifold_motion.dataio.seed_replay import ReplayConfig, SeedReplayRunner
from manifold_motion.stage2.flow import Normalizer, WindowData, _target_from_model, _torch_load
from manifold_motion.stage2.latent_prior import StateConditionedSkillVAE, latent_barrier
from manifold_motion.stage2.projection import project_reference
from manifold_motion.stage2.validate import validate_trajectory


def probe(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    checkpoint = _torch_load(args.prior_checkpoint, device)
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    target_field = str(checkpoint.get("model_target_field", "target_ref"))
    data = WindowData.load(args.windows, normalizer=normalizer, model_target_field=target_field)
    family_id = int(args.family_id)
    if not 0 <= family_id < data.primitive_count:
        raise ValueError(f"family id must be in [0,{data.primitive_count})")
    model = StateConditionedSkillVAE(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    condition = torch.as_tensor(
        data.condition_for_primitive(args.source_index, family_id)[None], device=device)
    corridor = np.asarray(data.raw["corridor"][args.source_index], dtype=np.float32)
    with torch.no_grad():
        prior_mean, prior_logvar = model.prior_stats(condition)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    runner = SeedReplayRunner()
    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    for candidate in range(args.candidates):
        with torch.no_grad():
            latent = prior_mean + torch.exp(0.5 * prior_logvar) * torch.randn(
                prior_mean.shape, generator=generator, device=device)
            latent = latent_barrier(latent, prior_mean, prior_logvar, args.latent_radius)
            decoded = model.decode(latent, condition).cpu().numpy().reshape(data.target_shape)
        trajectory = _target_from_model(normalizer.inverse_target(decoded),
                                        data.joint_lower, data.joint_upper)
        trajectory, projection = project_reference(trajectory, corridor)
        source = args.out / f"candidate_{candidate}.npz"
        np.savez_compressed(source, generated_ref=trajectory,
                            condition_corridor=corridor,
                            source_index=np.asarray(args.source_index),
                            primitive_id=np.asarray(family_id))
        _, summary = validate_trajectory(
            trajectory, source=source, source_hz=30.0, config=ReplayConfig(),
            corridor=corridor, max_corridor_radius=args.max_corridor_radius,
            stratum=f"latent_family_{family_id}", runner=runner)
        rows.append({"candidate": candidate, "accepted": bool(summary["accepted"]),
                     "failed_checks": summary["failed_checks"],
                     "corridor_radius_max": summary.get("corridor_radius_max"),
                     "corridor_radius_spatial_union_max": summary.get("corridor_radius_spatial_union_max"),
                     "corridor_radius_temporal_max": summary.get("corridor_radius_temporal_max"),
                     "track_err_mean_rad": summary.get("track_err_mean_rad"),
                     "base_z_min_m": summary.get("base_z_min_m"),
                     "reference_planar_path_m": summary.get("reference_planar_path_m"),
                     "exec_path_m": summary.get("exec_path_m"),
                     "projection_objective_before": projection["objective_before"]["total"],
                     "projection_objective_after": projection["objective_after"]["total"],
                     "sample": str(source)})
    report = {"schema": "manifold-motion.latent-candidate-probe.v1",
              "prior_checkpoint": str(args.prior_checkpoint), "windows": str(args.windows),
              "source_index": args.source_index, "family_id": family_id,
              "model_target_field": target_field,
              "family": str(data.raw["primitive_names"][family_id]), "candidates": rows,
              "accepted_count": int(sum(row["accepted"] for row in rows)),
              "max_corridor_radius": args.max_corridor_radius}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--family-id", type=int, required=True)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--latent-radius", type=float, default=3.0)
    parser.add_argument("--max-corridor-radius", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.candidates < 1 or args.latent_radius <= 0 or args.max_corridor_radius <= 0:
        parser.error("candidates and latent/corridor radii must be positive")
    probe(args); return 0


if __name__ == "__main__":
    raise SystemExit(main())
