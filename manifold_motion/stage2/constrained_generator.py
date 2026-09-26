"""Constrained Stage-2 trajectory hand-off.

This module joins the three safety/adaptation mechanisms requested for deployment:

1. a trainable, zero-initialized residual restricted to family-relevant body joints;
2. a state-conditioned latent ellipsoid barrier; and
3. the optimization-embedded reference projection before frozen SONIC execution.

The MuJoCo/SONIC physical selector still runs after this layer.  A learned or differentiable
constraint is a proposal guard, not evidence that a motion is physically executable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from manifold_motion.core import constants as C
from manifold_motion.stage2.composer import ComposerData, ComposerNormalizer, EnvironmentSkillComposer, _forward
from manifold_motion.stage2.flow import Normalizer, WindowData, _device, _target_from_model, _torch_load
from manifold_motion.stage2.latent_prior import StateConditionedSkillVAE, latent_barrier
from manifold_motion.stage2.projection import ProjectionConfig, project_reference


def _policy_indices(hardware_indices: Sequence[int]) -> np.ndarray:
    return np.asarray(C.ISAACLAB_TO_MUJOCO[np.asarray(hardware_indices, dtype=np.int64)], dtype=np.int64)


POLICY_GROUPS = {
    "legs": _policy_indices(range(0, 12)),
    "waist": _policy_indices(range(12, 15)),
    "left_arm": _policy_indices(range(15, 22)),
    "right_arm": _policy_indices(range(22, 29)),
}


def family_joint_mask(family_name: str) -> np.ndarray:
    """Return the local 29-DOF support of a skill in SONIC/policy joint order."""
    name = family_name.lower()
    if any(token in name for token in ("all_fours", "ladder", "recovery", "roll", "vault", "inchworm", "spider")):
        groups = ("legs", "waist", "left_arm", "right_arm")
    elif any(token in name for token in ("door", "button", "lever", "carry", "hands_back")):
        groups = ("waist", "left_arm", "right_arm")
    elif any(token in name for token in ("walk", "jog", "turn", "crouch", "dodge", "lunge", "hop",
                                              "jump", "step", "kneel")):
        groups = ("legs", "waist")
    else:
        # Unknown families are not granted a full-body residual implicitly.
        groups = ("waist",)
    mask = np.zeros(29, dtype=np.float32)
    for group in groups:
        mask[POLICY_GROUPS[group]] = 1.0
    return mask


class LocalBodyResidualHead(nn.Module):
    """Zero-parity trajectory correction with an explicit per-family joint mask."""

    def __init__(self, feature_dim: int, horizon: int, hidden: int = 128, bound_rad: float = 0.08):
        super().__init__()
        if min(feature_dim, horizon, hidden) <= 0 or bound_rad <= 0:
            raise ValueError("local residual dimensions/bound must be positive")
        self.horizon = int(horizon)
        self.bound_rad = float(bound_rad)
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, horizon * 29))
        # Before targeted fine-tuning this module is exactly identity, preserving SONIC behavior.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base: torch.Tensor, feature: torch.Tensor, joint_mask: torch.Tensor) -> torch.Tensor:
        if base.ndim != 3 or base.shape[1:] != (self.horizon, 38):
            raise ValueError(f"base must be [B,{self.horizon},38]")
        if joint_mask.shape != (len(base), 29):
            raise ValueError("joint_mask must be [B,29]")
        residual = self.bound_rad * torch.tanh(self.net(feature)).reshape(-1, self.horizon, 29)
        # Ramp in over four frames so a skill switch cannot introduce a joint discontinuity.
        ramp = torch.linspace(0.0, 1.0, min(4, self.horizon), device=base.device, dtype=base.dtype)
        temporal = torch.ones(self.horizon, device=base.device, dtype=base.dtype)
        temporal[:len(ramp)] = ramp
        q = base[..., :29] + residual * joint_mask[:, None, :] * temporal[None, :, None]
        return torch.cat((q, base[..., 29:]), dim=-1)


def apply_local_body_residual(trajectory: np.ndarray, residual: np.ndarray,
                              family_name: str, bound_rad: float = 0.08) -> tuple[np.ndarray, dict[str, Any]]:
    """Numpy inference equivalent of :class:`LocalBodyResidualHead`."""
    value = np.asarray(trajectory, dtype=np.float32).copy()
    correction = np.asarray(residual, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 38:
        raise ValueError("trajectory must have shape [T,38]")
    if correction.shape == (29,):
        correction = np.repeat(correction[None], len(value), axis=0)
    if correction.shape != (len(value), 29):
        raise ValueError(f"residual must have shape [29] or [{len(value)},29]")
    mask = family_joint_mask(family_name)
    correction = np.clip(correction, -bound_rad, bound_rad) * mask[None]
    ramp = np.ones((len(value), 1), dtype=np.float32)
    ramp[:min(4, len(value)), 0] = np.linspace(0.0, 1.0, min(4, len(value)))
    correction *= ramp
    value[:, :29] += correction
    return value, {"family": family_name, "active_joint_count": int(mask.sum()),
                   "bound_rad": float(bound_rad), "delta_max_rad": float(np.max(np.abs(correction))),
                   "joint_mask_policy_order": mask.astype(int).tolist()}


def generate_constrained_reference(
        model: StateConditionedSkillVAE, condition: torch.Tensor, proposed_latent: torch.Tensor,
        normalizer: Normalizer, joint_lower: np.ndarray, joint_upper: np.ndarray,
        target_shape: tuple[int, int], family_name: str, corridor: np.ndarray,
        *, local_residual: np.ndarray | None = None, latent_radius: float = 3.0,
        residual_bound_rad: float = 0.08, projection_config: ProjectionConfig | None = None,
        handoff_q: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode and constrain one latent proposal, returning a complete audit trail."""
    if len(condition) != 1 or len(proposed_latent) != 1:
        raise ValueError("constrained single-reference hand-off expects batch size one")
    model.eval()
    with torch.no_grad():
        prior_mean, prior_logvar = model.prior_stats(condition)
        safe_latent = latent_barrier(proposed_latent, prior_mean, prior_logvar, latent_radius)
        decoded = model.decode(safe_latent, condition).cpu().numpy().reshape(target_shape)
    model_target = normalizer.inverse_target(decoded)
    trajectory = _target_from_model(model_target, joint_lower, joint_upper)
    residual_report: dict[str, Any] = {"applied": False}
    if local_residual is not None:
        trajectory, residual_report = apply_local_body_residual(
            trajectory, local_residual, family_name, residual_bound_rad)
        residual_report["applied"] = True
    projected, projection_report = project_reference(
        trajectory, corridor, projection_config, handoff_q=handoff_q)
    scale = torch.exp(0.5 * prior_logvar).clamp_min(1e-4)
    before = torch.linalg.vector_norm((proposed_latent - prior_mean) / scale, dim=-1)
    after = torch.linalg.vector_norm((safe_latent - prior_mean) / scale, dim=-1)
    audit = {
        "family": family_name,
        "latent_barrier": {"radius": float(latent_radius),
                           "mahalanobis_before": float(before.item()),
                           "mahalanobis_after": float(after.item()),
                           "clipped": bool(before.item() > latent_radius + 1e-6)},
        "local_body_residual": residual_report,
        "optimization_projection": projection_report,
        "final_authority": "frozen SONIC + MuJoCo physical selector",
    }
    return projected, audit


__all__ = ["LocalBodyResidualHead", "apply_local_body_residual", "family_joint_mask",
           "generate_constrained_reference", "POLICY_GROUPS"]


def smoke(args: argparse.Namespace) -> int:
    """Run composer -> prior -> barrier/residual/projection on one aligned held-out window."""
    device = _device(args.device)
    prior_checkpoint = _torch_load(args.prior_checkpoint, device)
    prior_normalizer = Normalizer.from_state_dict(prior_checkpoint["normalizer"])
    prior_data = WindowData.load(args.prior_windows, normalizer=prior_normalizer,
                                 model_target_field=prior_checkpoint.get("model_target_field", "target_ref"))
    composer_checkpoint = _torch_load(args.composer_checkpoint, device)
    composer_data = ComposerData.load(
        args.environment_windows,
        normalizer=ComposerNormalizer.from_state_dict(composer_checkpoint["normalizer"]),
    )
    if len(prior_data.split) != len(composer_data.split):
        raise ValueError("prior and environment window archives are not aligned")
    for key in ("split", "primitive", "clip_index", "source_origin"):
        if key in prior_data.raw and key in composer_data.raw and not np.array_equal(prior_data.raw[key], composer_data.raw[key]):
            raise ValueError(f"prior/environment archives disagree on {key}")
    candidates = np.flatnonzero(composer_data.split == args.split)
    if args.sample_index >= 0:
        index = int(args.sample_index)
    elif args.true_family:
        family_id = composer_data.primitive_names.index(args.true_family)
        matching = candidates[composer_data.labels[candidates] == family_id]
        if not len(matching):
            raise ValueError(f"no {args.true_family} sample in requested split")
        index = int(matching[0])
    else:
        index = int(candidates[0])

    composer = EnvironmentSkillComposer(**composer_checkpoint["architecture"]).to(device)
    composer.load_state_dict(composer_checkpoint["model_state"]); composer.eval()
    with torch.no_grad():
        logits = _forward(composer, composer_data, np.asarray([index]), device)
        supported = torch.as_tensor(composer_checkpoint["supported_mask"], device=device, dtype=torch.bool)
        logits[:, ~supported] = -1e9
        probability = torch.softmax(logits, -1)[0]
        top_probability, top_ids = torch.topk(probability, k=min(3, int(supported.sum())))
    selected_id = int(top_ids[0].item())
    selected_name = composer_data.primitive_names[selected_id]

    prior = StateConditionedSkillVAE(**prior_checkpoint["architecture"]).to(device)
    prior.load_state_dict(prior_checkpoint["model_state"]); prior.eval()
    condition = torch.as_tensor(prior_data.condition_for_primitive(index, selected_id)[None], device=device)
    with torch.no_grad():
        mean, logvar = prior.prior_stats(condition)
        direction = torch.ones_like(mean) / np.sqrt(mean.shape[-1])
        proposal = mean + args.latent_offset * torch.exp(0.5 * logvar) * direction
    corridor = composer_data.raw["corridor"][index]
    residual = np.zeros((len(corridor), 29), dtype=np.float32)
    if args.residual_amplitude:
        phase = np.linspace(0.0, np.pi * 2.0, len(corridor), endpoint=False)
        residual[:] = args.residual_amplitude * np.sin(phase)[:, None]
    generated, audit = generate_constrained_reference(
        prior, condition, proposal, prior_data.normalizer, prior_data.joint_lower,
        prior_data.joint_upper, prior_data.target_shape, selected_name, corridor,
        local_residual=residual, latent_radius=args.latent_radius,
        residual_bound_rad=args.residual_bound)
    true_id = int(composer_data.labels[index])
    result = {
        "schema": "manifold-motion.constrained-generator-smoke.v1",
        "source_index": index, "split": int(composer_data.split[index]),
        "true_family": composer_data.primitive_names[true_id],
        "selected_family": selected_name, "selected_correct": selected_id == true_id,
        "top3": [{"family": composer_data.primitive_names[int(i)], "probability": float(p)}
                 for i, p in zip(top_ids.cpu().numpy(), top_probability.cpu().numpy())],
        "audit": audit,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "constrained_sample.npz", generated_ref=generated,
                        condition_corridor=corridor, condition_sdf=composer_data.raw["sdf"][index],
                        condition_self_manifold=composer_data.raw["self_manifold"][index],
                        selected_primitive=np.asarray(selected_id), primitive_name=np.asarray(selected_name),
                        source_index=np.asarray(index))
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("smoke")
    command.add_argument("--prior-checkpoint", type=Path, required=True)
    command.add_argument("--prior-windows", type=Path, required=True)
    command.add_argument("--composer-checkpoint", type=Path, required=True)
    command.add_argument("--environment-windows", type=Path, required=True)
    command.add_argument("--out", type=Path, default=Path("reports/manifold_motion/constrained_generator_smoke_v1"))
    command.add_argument("--sample-index", type=int, default=-1)
    command.add_argument("--true-family", default="walk_forward")
    command.add_argument("--split", type=int, choices=(0, 1, 2), default=2)
    command.add_argument("--latent-offset", type=float, default=3.5)
    command.add_argument("--latent-radius", type=float, default=3.0)
    command.add_argument("--residual-amplitude", type=float, default=0.01)
    command.add_argument("--residual-bound", type=float, default=0.08)
    command.add_argument("--device", default="cpu")
    command.set_defaults(handler=smoke)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
