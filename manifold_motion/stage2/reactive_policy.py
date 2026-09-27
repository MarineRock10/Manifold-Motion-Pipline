"""Learned reactive hazard-action prior for online projectile avoidance.

The teacher does not prescribe a trajectory.  For every randomized relative pose/velocity it
rolls out a small library of *action families* (keep, sidestep, crouch, retreat, hop) through
the measured self-manifold and scores swept ellipsoid clearance, boundary clearance and effort.
An MLP is then trained to imitate that optimization score.  At deployment the MLP receives the
current radar track and selects one action; the regular Flow candidate/shadow/surface gates
remain authoritative before any SONIC reference is committed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ACTION_NAMES = ("keep", "sidestep", "crouch", "retreat", "hop")
ACTION_DIM = len(ACTION_NAMES)
FEATURE_DIM = 18


def _features(rng: np.random.Generator, count: int) -> np.ndarray:
    """Sample sensor-level inputs; no schedule/event labels are exposed to the model."""
    rel = np.column_stack([
        rng.uniform(0.35, 3.2, count), rng.uniform(-1.15, 1.15, count),
        rng.uniform(-0.90, 0.90, count)])
    # Most examples are approaching hazards, with a smaller set of lateral/receding cases.
    vel = np.column_stack([
        rng.uniform(-3.6, 0.8, count), rng.uniform(-1.2, 1.2, count),
        rng.uniform(-0.35, 0.35, count)])
    robot_vel = np.column_stack([rng.uniform(-0.15, 0.55, count), rng.uniform(-0.18, 0.18, count)])
    self_semi = np.column_stack([
        rng.uniform(0.38, 0.52, count), rng.uniform(0.28, 0.48, count),
        rng.uniform(0.82, 1.12, count)])
    free = np.column_stack([
        rng.uniform(0.28, 1.25, count), rng.uniform(0.28, 1.25, count),
        rng.uniform(0.18, 1.25, count), rng.uniform(0.20, 0.90, count)])
    narrow = rng.random(count) < 0.45
    free[narrow, :2] = rng.uniform(0.25, 0.58, (int(np.sum(narrow)), 2))
    radius = rng.uniform(0.06, 0.18, (count, 1))
    closing = np.maximum(-vel[:, 0:1], 0.0)
    ttc = np.clip(rel[:, 0:1] / np.maximum(closing, 0.05), 0.0, 8.0)
    return np.concatenate([rel, vel, robot_vel, self_semi, free, radius, ttc, closing], axis=1).astype(np.float32)


def _action_centres(action: int, t: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Action-family root displacement in robot coordinates over the reaction horizon."""
    u = np.clip(t / 1.10, 0.0, 1.0)
    smooth = u * u * (3.0 - 2.0 * u)
    out = np.zeros((len(t), 3), dtype=np.float32)
    if action == 0:                         # continue current gait
        out[:, 0] = 0.28 * u
    elif action == 1:                       # lateral dodge, sign selected by free space
        out[:, 0] = 0.16 * u; out[:, 1] = sign * 0.62 * smooth
    elif action == 2:                       # crouch in place, preserve forward progress
        out[:, 0] = 0.12 * u; out[:, 2] = -0.42 * smooth
    elif action == 3:                       # short retreat
        out[:, 0] = -0.34 * smooth
    else:                                   # bounded hop; only valid with vertical aperture
        out[:, 0] = 0.20 * u; out[:, 2] = 0.55 * np.sin(np.pi * u)
    return out


def teacher_scores(features: np.ndarray, *, horizon_s: float = 1.10,
                   steps: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Return optimization-derived utility and the collision-free action label."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"features must be [N,{FEATURE_DIM}], got {x.shape}")
    # Inputs: rel(3), velocity(3), robot velocity(2), semi(3), free left/right/up/back(4), r.
    rel, vel, self_semi = x[:, :3], x[:, 3:6], x[:, 8:11]
    free_left, free_right, free_up, free_back = [x[:, 11 + i] for i in range(4)]
    radius = x[:, 15]
    times = np.linspace(0.0, float(horizon_s), int(steps), dtype=np.float32)
    utilities = np.empty((len(x), ACTION_DIM), dtype=np.float32)
    for action in range(ACTION_DIM):
        sign = np.where(free_left >= free_right, 1.0, -1.0).astype(np.float32)
        # _action_centres accepts one sign per sample; broadcast time/sample dimensions here.
        root = np.stack([_action_centres(action, times, np.full(len(times), s))
                         for s in sign], axis=0)
        projectile = rel[:, None, :] + vel[:, None, :] * times[None, :, None]
        delta = projectile - root
        semi = np.repeat(self_semi[:, None, :], len(times), axis=1)
        # A conservative swept ellipsoid clearance. Positive means outside the body envelope.
        ellipsoid = np.sqrt(np.sum((delta / (semi + radius[:, None, None] + 1e-5)) ** 2, axis=2)) - 1.0
        clearance = np.min(ellipsoid * np.min(semi, axis=2), axis=1)
        lateral_limit = np.where(sign > 0.0, free_left, free_right)
        lateral = np.max(np.abs(root[:, :, 1]), axis=1)
        feasible = np.ones(len(x), dtype=np.float32)
        if action == 1:
            feasible *= (lateral_limit > lateral + 0.05).astype(np.float32)
        elif action == 2:
            feasible *= (self_semi[:, 2] - 0.42 > 0.46).astype(np.float32)
        elif action == 3:
            feasible *= (free_back > 0.38).astype(np.float32)
        elif action == 4:
            feasible *= (free_up > 0.68).astype(np.float32)
        effort = np.asarray([0.00, 0.10, 0.08, 0.09, 0.12], dtype=np.float32)[action]
        # Keep is preferred when safe; under threat the best clearance dominates.
        threat = np.clip(1.0 - np.minimum(clearance / 0.20, 1.0), 0.0, 1.0)
        utilities[:, action] = clearance + 0.18 * (1.0 - threat) - effort
        utilities[:, action] -= (1.0 - feasible) * 2.0
    labels = np.argmax(utilities, axis=1).astype(np.int64)
    return utilities, labels


class ReactiveHazardPolicy(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, ACTION_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class ReactivePolicy:
    model: ReactiveHazardPolicy
    mean: np.ndarray
    std: np.ndarray
    checkpoint: Path

    @classmethod
    def load(cls, checkpoint: Path, device: str = "cpu") -> "ReactivePolicy":
        value = torch.load(checkpoint, map_location=device, weights_only=False)
        model = ReactiveHazardPolicy(**value["architecture"])
        model.load_state_dict(value["model_state"]); model.eval()
        return cls(model, np.asarray(value["mean"], dtype=np.float32),
                   np.asarray(value["std"], dtype=np.float32), Path(checkpoint))

    def predict(self, features: np.ndarray) -> dict[str, Any]:
        x = (np.asarray(features, dtype=np.float32) - self.mean) / self.std
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x.reshape(1, -1))).numpy()[0]
        logits = logits - np.max(logits); p = np.exp(logits); p /= np.sum(p)
        index = int(np.argmax(p))
        return {"action_id": index, "action": ACTION_NAMES[index],
                "probability": float(p[index]), "probabilities": p.tolist()}


def train(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(int(args.seed))
    x = _features(rng, int(args.samples))
    # Mirror augmentation forces left/right symmetry without hand-coding a direction policy.
    x_mirror = x.copy(); x_mirror[:, 1] *= -1; x_mirror[:, 4] *= -1
    x_mirror[:, 11], x_mirror[:, 12] = x[:, 12], x[:, 11]
    x = np.concatenate([x, x_mirror], axis=0)
    utilities, labels = teacher_scores(x)
    order = rng.permutation(len(x)); n_test = max(1, len(x) // 10); n_val = max(1, len(x) // 10)
    test, val, train_idx = order[:n_test], order[n_test:n_test + n_val], order[n_test + n_val:]
    mean, std = x[train_idx].mean(0).astype(np.float32), (x[train_idx].std(0) + 1e-5).astype(np.float32)
    tx = torch.from_numpy((x[train_idx] - mean) / std); ty = torch.from_numpy(labels[train_idx])
    model = ReactiveHazardPolicy(hidden=int(args.hidden)); opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    counts = np.bincount(labels[train_idx], minlength=ACTION_DIM).astype(np.float32)
    class_weight = torch.from_numpy(np.sqrt(np.max(counts) / np.maximum(counts, 1.0)))
    generator = torch.Generator().manual_seed(int(args.seed))
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for batch in torch.randperm(len(tx), generator=generator).split(int(args.batch_size)):
            logits = model(tx[batch]); loss = nn.functional.cross_entropy(
                logits, ty[batch], weight=class_weight)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy((x[test] - mean) / std)).argmax(1).numpy()
    accuracy = float(np.mean(pred == labels[test]))
    distribution = {name: int(np.sum(labels == i)) for i, name in enumerate(ACTION_NAMES)}
    per_action = {name: (float(np.mean(pred[labels[test] == i] == i))
                         if np.any(labels[test] == i) else None)
                  for i, name in enumerate(ACTION_NAMES)}
    report = {"schema": "manifold-motion.reactive-hazard-policy.v1", "samples": int(len(x)),
              "train_samples": int(len(train_idx)), "test_accuracy": accuracy,
              "label_distribution": distribution, "test_recall_by_action": per_action,
              "action_names": list(ACTION_NAMES), "teacher": "swept self-manifold clearance + feasibility + effort",
              "seed": int(args.seed), "epochs": int(args.epochs)}
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": report["schema"], "architecture": {"hidden": int(args.hidden)},
                "model_state": model.state_dict(), "mean": mean, "std": std,
                "action_names": ACTION_NAMES}, args.out / "reactive_policy.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)); return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", nargs="?")
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=40); parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args(); return train(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ACTION_NAMES", "ReactivePolicy", "teacher_scores"]
