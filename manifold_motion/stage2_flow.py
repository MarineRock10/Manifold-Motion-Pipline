"""Conditional latent Flow Matching baseline for Stage 2.

This is the first dynamic model for the data contract emitted by :mod:`seed_windows`:

``(M_e corridor/SDF, primitive, state, history, command) -> future R_ref``.

It is intentionally a *latent* flow: a conditional VAE first maps the 1.6-second
trajectory into a compact latent, then Flow Matching learns the conditional latent vector
field.  That is substantially more stable than learning a vector field over 48 x 38 raw
trajectory coordinates as the first implementation.

The dataset also contains ``target_exec``.  It is never substituted for the generated
reference: SONIC consumes R_ref, while R_exec remains the executor-grounded evaluation
target.

Typical use::

    python3 -m manifold_motion.stage2_flow train-ae --windows reports/.../seed_stage2_windows.npz
    python3 -m manifold_motion.stage2_flow train-flow --windows reports/.../seed_stage2_windows.npz
    python3 -m manifold_motion.stage2_flow sample --windows reports/.../seed_stage2_windows.npz
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import constants as C
from .env import G1FlatEnv


TARGET_PARAMETERIZATION = "bounded_joint_logit_v1"
# The SEED selection taxonomy is fixed.  Subsets such as walk-only training must retain the
# same one-hot width as the full dataset or their checkpoints become silently incompatible.
LEGACY_PRIMITIVE_COUNT = 7
PRIMITIVE_NAMES = (
    "all_fours", "crawl", "crouch", "low_transition",
    "walk_lateral_reverse", "walk_nominal", "walk_turn", "jump",
)
PRIMITIVE_COUNT = len(PRIMITIVE_NAMES)


def _policy_joint_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Physical G1 joint ranges in SONIC's policy order, read from the active MJCF."""
    env = G1FlatEnv()
    joint_ids = env.model.actuator_trnid[env.body_act, 0]
    return (env.model.jnt_range[joint_ids, 0][C.MUJOCO_TO_ISAACLAB].astype(np.float32),
            env.model.jnt_range[joint_ids, 1][C.MUJOCO_TO_ISAACLAB].astype(np.float32))


def _target_to_model(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Map physical q to logits, so the decoder's inverse is *always* joint-range valid."""
    value = np.asarray(target, dtype=np.float32).copy()
    ratio = (value[..., :29] - lower) / (upper - lower)
    ratio = np.clip(ratio, 1e-5, 1.0 - 1e-5)
    value[..., :29] = np.log(ratio / (1.0 - ratio))
    return value


def _target_from_model(value: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    output = np.asarray(value, dtype=np.float32).copy()
    logits = np.clip(output[..., :29], -40.0, 40.0)
    ratio = 1.0 / (1.0 + np.exp(-logits))
    output[..., :29] = lower + ratio * (upper - lower)
    return output


def _safe_std(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    std = values.std(axis=axis)
    return np.where(std < 1e-6, 1.0, std)


def _environment_vector(raw: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten the versioned environment condition while retaining legacy window support."""
    count = len(raw["state"])
    parts = [raw["manifold"].reshape(count, -1)]
    # New reverse-corridor data adds a time-aligned ellipsoid sequence and a fixed local SDF.
    # Their order is part of the data contract and must not be inferred from archive key order.
    for key in ("corridor", "sdf"):
        if key in raw:
            parts.append(raw[key].reshape(count, -1))
    return np.concatenate(parts, axis=1).astype(np.float32)


@dataclass
class Normalizer:
    """Train-split statistics only; validation/test actors never influence them."""

    state_mean: np.ndarray
    state_std: np.ndarray
    manifold_mean: np.ndarray
    manifold_std: np.ndarray
    command_mean: np.ndarray
    command_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    @classmethod
    def fit(cls, raw: dict[str, np.ndarray], train: np.ndarray,
            model_target: np.ndarray | None = None, environment: np.ndarray | None = None) -> "Normalizer":
        if not train.any():
            raise ValueError("the window file has no training split")
        state_values = np.concatenate([raw["state"][train], raw["history"][train].reshape(-1, raw["state"].shape[1])])
        target_source = raw["target_ref"] if model_target is None else model_target
        target_values = target_source[train].reshape(-1, target_source.shape[-1])
        environment = _environment_vector(raw) if environment is None else environment
        return cls(
            state_values.mean(axis=0), _safe_std(state_values, axis=0),
            environment[train].mean(axis=0), _safe_std(environment[train], axis=0),
            raw["command"][train].mean(axis=0), _safe_std(raw["command"][train], axis=0),
            target_values.mean(axis=0), _safe_std(target_values, axis=0),
        )

    def state(self, value: np.ndarray) -> np.ndarray:
        return (value - self.state_mean) / self.state_std

    def manifold(self, value: np.ndarray) -> np.ndarray:
        return (value - self.manifold_mean) / self.manifold_std

    def command(self, value: np.ndarray) -> np.ndarray:
        return (value - self.command_mean) / self.command_std

    def target(self, value: np.ndarray) -> np.ndarray:
        return (value - self.target_mean) / self.target_std

    def inverse_target(self, value: np.ndarray) -> np.ndarray:
        return value * self.target_std + self.target_mean

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name), dtype=np.float32)
                for name in ("state_mean", "state_std", "manifold_mean", "manifold_std",
                             "command_mean", "command_std", "target_mean", "target_std")}

    @classmethod
    def from_state_dict(cls, values: dict[str, np.ndarray]) -> "Normalizer":
        return cls(**{name: np.asarray(values[name], dtype=np.float32)
                      for name in ("state_mean", "state_std", "manifold_mean", "manifold_std",
                                   "command_mean", "command_std", "target_mean", "target_std")})


@dataclass
class WindowData:
    condition: np.ndarray      # [N, C]
    target: np.ndarray         # [N, horizon * trajectory_dim], normalized R_ref
    split: np.ndarray          # [N], 0=train, 1=validation, 2=test
    target_shape: tuple[int, int]
    normalizer: Normalizer
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    primitive_count: int
    raw: dict[str, np.ndarray]

    @classmethod
    def load(cls, path: Path, normalizer: Normalizer | None = None,
             model_target_field: str = "target_ref") -> "WindowData":
        with np.load(path) as archive:
            required = {"state", "history", "primitive", "manifold", "command", "target_ref", "target_exec", "split"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{path} is not a Stage-2 window file; missing {sorted(missing)}")
            optional = {key for key in ("corridor", "sdf", "clip_index", "source_origin", "primitive_count") if key in archive.files}
            raw = {key: np.asarray(archive[key]) for key in required | optional}
        if model_target_field not in {"target_ref", "target_exec"}:
            raise ValueError(f"model_target_field must be target_ref or target_exec, got {model_target_field}")
        continuous = ("state", "history", "manifold", "command", "target_ref", "target_exec", "corridor", "sdf")
        if not all(np.isfinite(raw[key]).all() for key in continuous if key in raw):
            raise ValueError(f"{path} contains non-finite continuous values")
        train = raw["split"] == 0
        joint_lower, joint_upper = _policy_joint_bounds()
        model_target = _target_to_model(raw[model_target_field], joint_lower, joint_upper)
        environment = _environment_vector(raw)
        normalizer = normalizer or Normalizer.fit(raw, train, model_target=model_target, environment=environment)
        if "primitive_count" in raw:
            primitive_count = int(np.asarray(raw["primitive_count"]).reshape(-1)[0])
        else:
            # Archives written before the jump extension have a seven-wide categorical
            # condition even if they contain only a crouch or transition subset.
            primitive_count = LEGACY_PRIMITIVE_COUNT if int(raw["primitive"].max()) < LEGACY_PRIMITIVE_COUNT else int(raw["primitive"].max()) + 1
        if primitive_count < int(raw["primitive"].max()) + 1 or primitive_count > PRIMITIVE_COUNT:
            raise ValueError(f"{path} has invalid primitive_count {primitive_count}")
        one_hot = np.eye(primitive_count, dtype=np.float32)[raw["primitive"].astype(np.int64)]
        condition = np.concatenate([
            normalizer.state(raw["state"]).astype(np.float32),
            normalizer.state(raw["history"]).reshape(len(raw["state"]), -1).astype(np.float32),
            one_hot,
            normalizer.manifold(environment).astype(np.float32),
            normalizer.command(raw["command"]).astype(np.float32),
        ], axis=1)
        target_normalized = normalizer.target(model_target).reshape(len(raw["state"]), -1).astype(np.float32)
        return cls(condition, target_normalized, raw["split"].astype(np.int64),
                   tuple(raw["target_ref"].shape[1:]), normalizer, joint_lower, joint_upper, primitive_count, raw)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int, layers: int = 3):
        super().__init__()
        blocks: list[nn.Module] = []
        dim = input_dim
        for _ in range(layers):
            blocks.extend((nn.Linear(dim, hidden), nn.SiLU()))
            dim = hidden
        blocks.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ConditionalVAE(nn.Module):
    def __init__(self, target_dim: int, condition_dim: int, latent_dim: int = 64,
                 condition_hidden: int = 256, hidden: int = 512):
        super().__init__()
        self.target_dim = target_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.condition_hidden = condition_hidden
        self.hidden = hidden
        self.condition_encoder = _MLP(condition_dim, condition_hidden, condition_hidden, layers=2)
        self.encoder = _MLP(target_dim + condition_hidden, 2 * latent_dim, hidden, layers=3)
        self.decoder = _MLP(latent_dim + condition_hidden, target_dim, hidden, layers=3)

    def encode(self, target: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.encoder(torch.cat([target, self.condition_encoder(condition)], dim=-1))
        return stats.chunk(2, dim=-1)

    def decode(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([latent, self.condition_encoder(condition)], dim=-1))

    def forward(self, target: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(target, condition)
        latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return self.decode(latent, condition), mean, logvar

    def architecture(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in
                ("target_dim", "condition_dim", "latent_dim", "condition_hidden", "hidden")}


class ConditionalTrajectoryMean(nn.Module):
    """Deterministic conditional trajectory proposal used as a Stage-2 ablation/baseline.

    Its data contract is exactly the latent-flow contract.  It is useful both as a measurable
    conditional-mean baseline and as the future anchor for residual Flow Matching; it must not
    be confused with the stochastic Flow sampler.
    """

    def __init__(self, target_dim: int, condition_dim: int, hidden: int = 512):
        super().__init__()
        self.target_dim = target_dim
        self.condition_dim = condition_dim
        self.hidden = hidden
        self.net = _MLP(condition_dim, target_dim, hidden, layers=4)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.net(condition)

    def architecture(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in ("target_dim", "condition_dim", "hidden")}


def _time_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(torch.linspace(0.0, math.log(1000.0), half, device=t.device))
    phase = t[:, None] * frequencies[None, :]
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class LatentFlow(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, condition_hidden: int = 256, hidden: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.condition_hidden = condition_hidden
        self.hidden = hidden
        self.condition_encoder = _MLP(condition_dim, condition_hidden, condition_hidden, layers=2)
        self.vector_field = _MLP(latent_dim + 32 + condition_hidden, latent_dim, hidden, layers=4)

    def forward(self, latent: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.vector_field(torch.cat([latent, _time_embedding(t), self.condition_encoder(condition)], dim=-1))

    def architecture(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in
                ("latent_dim", "condition_dim", "condition_hidden", "hidden")}


class ResidualTrajectoryFlow(nn.Module):
    """Flow Matching model for residuals around a known executable proposal.

    The residual is represented in the same normalized bounded-joint target space as the
    conditional-mean model.  Sampling keeps candidate 0 as the exact mean proposal and only
    perturbs subsequent candidates, so the stochastic model cannot erase a passing baseline.
    """

    def __init__(self, target_dim: int, condition_dim: int, condition_hidden: int = 256,
                 hidden: int = 256):
        super().__init__()
        self.target_dim = target_dim
        self.condition_dim = condition_dim
        self.condition_hidden = condition_hidden
        self.hidden = hidden
        self.condition_encoder = _MLP(condition_dim, condition_hidden, condition_hidden, layers=2)
        self.vector_field = _MLP(target_dim + 32 + condition_hidden, target_dim, hidden, layers=3)

    def forward(self, residual: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.vector_field(torch.cat([residual, _time_embedding(t),
                                             self.condition_encoder(condition)], dim=-1))

    def architecture(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in
                ("target_dim", "condition_dim", "condition_hidden", "hidden")}


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator):
    shuffled = rng.permutation(indices)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start:start + batch_size]


def _evaluate_vae(model: ConditionalVAE, data: WindowData, indices: np.ndarray, device: torch.device, batch_size: int) -> float:
    if not len(indices):
        return float("nan")
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            target = torch.as_tensor(data.target[batch], device=device)
            condition = torch.as_tensor(data.condition[batch], device=device)
            mean, _ = model.encode(target, condition)
            reconstruction = model.decode(mean, condition)
            losses.append(float(torch.mean((reconstruction - target) ** 2).cpu()))
    return float(np.mean(losses))


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before weights_only was added
        return torch.load(path, map_location=device)


def train_ae(args: argparse.Namespace) -> int:
    device = _device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = WindowData.load(args.windows)
    train = np.flatnonzero(data.split == 0)
    validation = np.flatnonzero(data.split == 1)
    model = ConditionalVAE(data.target.shape[1], data.condition.shape[1], args.latent_dim,
                           args.condition_hidden, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best = float("inf")
    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in _batches(train, args.batch_size, rng):
            target = torch.as_tensor(data.target[batch], device=device)
            condition = torch.as_tensor(data.condition[batch], device=device)
            reconstructed, mean, logvar = model(target, condition)
            reconstruction = torch.mean((reconstructed - target) ** 2)
            kl = -0.5 * torch.mean(1.0 + logvar - mean.square() - logvar.exp())
            loss = reconstruction + args.kl_weight * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_loss = _evaluate_vae(model, data, validation if len(validation) else train, device, args.batch_size)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_reconstruction": validation_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best:
            best = validation_loss
            torch.save({"kind": "conditional_vae", "architecture": model.architecture(),
                        "model_state": model.state_dict(), "normalizer": data.normalizer.state_dict(),
                        "target_shape": data.target_shape, "primitive_count": data.primitive_count,
                        "target_parameterization": TARGET_PARAMETERIZATION,
                        "joint_lower": data.joint_lower, "joint_upper": data.joint_upper,
                        "windows": str(args.windows)}, args.out / "autoencoder.pt")
    (args.out / "autoencoder_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return 0


def _evaluate_mean(model: ConditionalTrajectoryMean, data: WindowData, indices: np.ndarray,
                   device: torch.device, batch_size: int) -> float:
    if not len(indices):
        return float("nan")
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            target = torch.as_tensor(data.target[batch], device=device)
            condition = torch.as_tensor(data.condition[batch], device=device)
            losses.append(float(torch.mean((model(condition) - target) ** 2).cpu()))
    return float(np.mean(losses))


def train_mean(args: argparse.Namespace) -> int:
    """Fit a deterministic conditional proposal under the exact Flow data contract."""
    device = _device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = WindowData.load(args.windows, model_target_field=args.model_target_field)
    train = np.flatnonzero(data.split == 0)
    validation = np.flatnonzero(data.split == 1)
    if args.primitive_id >= 0:
        if args.primitive_id > int(data.raw["primitive"].max()):
            raise ValueError(f"primitive-id {args.primitive_id} is absent from {args.windows}")
        train = train[data.raw["primitive"][train] == args.primitive_id]
        validation = validation[data.raw["primitive"][validation] == args.primitive_id]
        if not len(train):
            raise ValueError(f"no training windows for primitive-id {args.primitive_id}")
    model = ConditionalTrajectoryMean(data.target.shape[1], data.condition.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best = float("inf")
    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in _batches(train, args.batch_size, rng):
            target = torch.as_tensor(data.target[batch], device=device)
            condition = torch.as_tensor(data.condition[batch], device=device)
            loss = torch.mean((model(condition) - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_loss = _evaluate_mean(model, data, validation if len(validation) else train, device, args.batch_size)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_trajectory_mse": validation_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best:
            best = validation_loss
            torch.save({"kind": "conditional_trajectory_mean", "architecture": model.architecture(),
                        "model_state": model.state_dict(), "normalizer": data.normalizer.state_dict(),
                        "target_shape": data.target_shape, "primitive_count": data.primitive_count,
                        "target_parameterization": TARGET_PARAMETERIZATION,
                        "joint_lower": data.joint_lower, "joint_upper": data.joint_upper,
                        "primitive_id": int(args.primitive_id),
                        "windows": str(args.windows), "model_target_field": args.model_target_field},
                       args.out / "conditional_mean.pt")
    (args.out / "conditional_mean_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return 0


def _mean_predictions(model: ConditionalTrajectoryMean, data: WindowData,
                      indices: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    """Predict normalized targets without retaining an autograd graph."""
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            condition = torch.as_tensor(data.condition[batch], device=device)
            outputs.append(model(condition).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, data.target.shape[1]), np.float32)


def _evaluate_residual_flow(flow: ResidualTrajectoryFlow, mean: ConditionalTrajectoryMean,
                            data: WindowData, indices: np.ndarray, residual_std: np.ndarray,
                            device: torch.device, batch_size: int, seed: int) -> float:
    if not len(indices):
        return float("nan")
    generator = torch.Generator(device=device).manual_seed(seed)
    flow.eval()
    mean.eval()
    losses = []
    std = torch.as_tensor(residual_std, device=device)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            target = torch.as_tensor(data.target[batch], device=device)
            condition = torch.as_tensor(data.condition[batch], device=device)
            residual = (target - mean(condition)) / std
            base = torch.randn(residual.shape, device=device, generator=generator)
            t = torch.rand(len(batch), device=device, generator=generator)
            current = (1.0 - t[:, None]) * base + t[:, None] * residual
            losses.append(float(torch.mean((flow(current, t, condition) - (residual - base)) ** 2).cpu()))
    return float(np.mean(losses))


def train_residual_flow(args: argparse.Namespace) -> int:
    """Train a stochastic residual field around a previously validated mean proposal."""
    device = _device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    mean, mean_checkpoint = _load_mean(args.mean_model, device)
    normalizer = Normalizer.from_state_dict(mean_checkpoint["normalizer"])
    data = WindowData.load(args.windows, normalizer=normalizer)
    if not (np.allclose(data.joint_lower, mean_checkpoint["joint_lower"]) and
            np.allclose(data.joint_upper, mean_checkpoint["joint_upper"]) and
            mean.condition_dim == data.condition.shape[1] and mean.target_dim == data.target.shape[1]):
        raise ValueError("conditional-mean checkpoint is incompatible with the active G1/window condition")
    train = np.flatnonzero(data.split == 0)
    validation = np.flatnonzero(data.split == 1)
    if args.primitive_id >= 0:
        train = train[data.raw["primitive"][train] == args.primitive_id]
        validation = validation[data.raw["primitive"][validation] == args.primitive_id]
        if not len(train):
            raise ValueError(f"no training windows for primitive-id {args.primitive_id}")
    # Standardize residual coordinates using training actors only.  The mean proposal remains
    # the exact zero-residual anchor at sampling time.
    mean_train = _mean_predictions(mean, data, train, device, args.batch_size)
    residual_train = data.target[train] - mean_train
    residual_std = _safe_std(residual_train, axis=0).astype(np.float32)
    flow = ResidualTrajectoryFlow(data.target.shape[1], data.condition.shape[1],
                                  args.condition_hidden, args.hidden).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best = float("inf")
    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    std = torch.as_tensor(residual_std, device=device)
    for epoch in range(1, args.epochs + 1):
        flow.train()
        mean.eval()
        losses = []
        for batch in _batches(train, args.batch_size, rng):
            target = torch.as_tensor(data.target[batch], device=device)
            condition = torch.as_tensor(data.condition[batch], device=device)
            with torch.no_grad():
                residual = (target - mean(condition)) / std
            base = torch.randn_like(residual)
            t = torch.rand(len(batch), device=device)
            current = (1.0 - t[:, None]) * base + t[:, None] * residual
            loss = torch.mean((flow(current, t, condition) - (residual - base)) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_loss = _evaluate_residual_flow(
            flow, mean, data, validation if len(validation) else train, residual_std,
            device, args.batch_size, args.seed + epoch)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "validation_residual_flow_mse": validation_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best:
            best = validation_loss
            torch.save({"kind": "residual_trajectory_flow", "architecture": flow.architecture(),
                        "model_state": flow.state_dict(), "mean_model": str(args.mean_model),
                        "normalizer": mean_checkpoint["normalizer"], "target_shape": data.target_shape,
                        "target_parameterization": TARGET_PARAMETERIZATION,
                        "joint_lower": data.joint_lower, "joint_upper": data.joint_upper,
                        "residual_std": residual_std, "primitive_id": int(args.primitive_id),
                        "windows": str(args.windows)}, args.out / "residual_flow.pt")
    (args.out / "residual_flow_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return 0


def _load_ae(path: Path, device: torch.device) -> tuple[ConditionalVAE, dict[str, Any]]:
    checkpoint = _torch_load(path, device)
    if checkpoint.get("kind") != "conditional_vae":
        raise ValueError(f"{path} is not a conditional VAE checkpoint")
    if checkpoint.get("target_parameterization") != TARGET_PARAMETERIZATION:
        raise ValueError(f"{path} does not use the current bounded G1 joint target parameterization; retrain it")
    model = ConditionalVAE(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _load_mean(path: Path, device: torch.device) -> tuple[ConditionalTrajectoryMean, dict[str, Any]]:
    checkpoint = _torch_load(path, device)
    if checkpoint.get("kind") != "conditional_trajectory_mean":
        raise ValueError(f"{path} is not a conditional trajectory-mean checkpoint")
    if checkpoint.get("target_parameterization") != TARGET_PARAMETERIZATION:
        raise ValueError(f"{path} does not use the current bounded G1 joint target parameterization; retrain it")
    model = ConditionalTrajectoryMean(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _load_residual_flow(path: Path, device: torch.device) -> tuple[ResidualTrajectoryFlow, dict[str, Any]]:
    checkpoint = _torch_load(path, device)
    if checkpoint.get("kind") != "residual_trajectory_flow":
        raise ValueError(f"{path} is not a residual trajectory Flow checkpoint")
    if checkpoint.get("target_parameterization") != TARGET_PARAMETERIZATION:
        raise ValueError(f"{path} does not use the current bounded G1 joint target parameterization; retrain it")
    model = ResidualTrajectoryFlow(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _evaluate_flow(flow: LatentFlow, ae: ConditionalVAE, data: WindowData, indices: np.ndarray,
                   device: torch.device, batch_size: int, seed: int) -> float:
    if not len(indices):
        return float("nan")
    generator = torch.Generator(device=device).manual_seed(seed)
    flow.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            target = torch.as_tensor(data.target[batch], device=device)
            condition = torch.as_tensor(data.condition[batch], device=device)
            latent, _ = ae.encode(target, condition)
            base = torch.randn(latent.shape, device=device, generator=generator)
            t = torch.rand(len(batch), device=device, generator=generator)
            current = (1.0 - t[:, None]) * base + t[:, None] * latent
            loss = torch.mean((flow(current, t, condition) - (latent - base)) ** 2)
            losses.append(float(loss.cpu()))
    return float(np.mean(losses))


def train_flow(args: argparse.Namespace) -> int:
    device = _device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    autoencoder_path = args.autoencoder or (args.out / "autoencoder.pt")
    ae, ae_checkpoint = _load_ae(autoencoder_path, device)
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    normalizer = Normalizer.from_state_dict(ae_checkpoint["normalizer"])
    data = WindowData.load(args.windows, normalizer=normalizer)
    if not (np.allclose(data.joint_lower, ae_checkpoint["joint_lower"]) and
            np.allclose(data.joint_upper, ae_checkpoint["joint_upper"])):
        raise ValueError("the active G1 MJCF joint ranges differ from the autoencoder checkpoint")
    if tuple(data.target_shape) != tuple(ae_checkpoint["target_shape"]):
        raise ValueError("window target shape does not match the autoencoder checkpoint")
    train = np.flatnonzero(data.split == 0)
    validation = np.flatnonzero(data.split == 1)
    flow = LatentFlow(ae.latent_dim, data.condition.shape[1], args.condition_hidden, args.hidden).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best = float("inf")
    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        flow.train()
        losses = []
        for batch in _batches(train, args.batch_size, rng):
            target = torch.as_tensor(data.target[batch], device=device)
            condition = torch.as_tensor(data.condition[batch], device=device)
            with torch.no_grad():
                latent, _ = ae.encode(target, condition)
            base = torch.randn_like(latent)
            t = torch.rand(len(batch), device=device)
            current = (1.0 - t[:, None]) * base + t[:, None] * latent
            loss = torch.mean((flow(current, t, condition) - (latent - base)) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_loss = _evaluate_flow(flow, ae, data, validation if len(validation) else train,
                                         device, args.batch_size, args.seed + epoch)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_flow_mse": validation_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best:
            best = validation_loss
            torch.save({"kind": "latent_flow_matching", "architecture": flow.architecture(),
                        "model_state": flow.state_dict(), "autoencoder": str(autoencoder_path),
                        "normalizer": ae_checkpoint["normalizer"], "target_shape": data.target_shape,
                        "target_parameterization": TARGET_PARAMETERIZATION,
                        "joint_lower": data.joint_lower, "joint_upper": data.joint_upper,
                        "windows": str(args.windows)}, args.out / "flow.pt")
    (args.out / "flow_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return 0


def _load_flow(path: Path, device: torch.device) -> tuple[LatentFlow, dict[str, Any]]:
    checkpoint = _torch_load(path, device)
    if checkpoint.get("kind") != "latent_flow_matching":
        raise ValueError(f"{path} is not a latent Flow Matching checkpoint")
    if checkpoint.get("target_parameterization") != TARGET_PARAMETERIZATION:
        raise ValueError(f"{path} does not use the current bounded G1 joint target parameterization; retrain it")
    model = LatentFlow(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def sample(args: argparse.Namespace) -> int:
    device = _device(args.device)
    ae: ConditionalVAE | None = None
    ae_checkpoint: dict[str, Any] | None = None
    flow: LatentFlow | None = None
    residual_flow: ResidualTrajectoryFlow | None = None
    mean: ConditionalTrajectoryMean | None = None
    flow_checkpoint: dict[str, Any] | None = None
    residual_checkpoint: dict[str, Any] | None = None
    mean_checkpoint: dict[str, Any] | None = None
    if args.sampler in ("conditional_mean", "residual_flow"):
        if args.mean_model is None:
            raise ValueError("--mean-model is required with --sampler conditional_mean/residual_flow")
        mean, mean_checkpoint = _load_mean(args.mean_model, device)
        normalizer = Normalizer.from_state_dict(mean_checkpoint["normalizer"])
        data = WindowData.load(args.windows, normalizer=normalizer)
        if not (np.allclose(data.joint_lower, mean_checkpoint["joint_lower"]) and
                np.allclose(data.joint_upper, mean_checkpoint["joint_upper"]) and
                mean.condition_dim == data.condition.shape[1] and mean.target_dim == data.target.shape[1]):
            raise ValueError("conditional-mean checkpoint is incompatible with the active G1/window condition")
    else:
        ae, ae_checkpoint = _load_ae(args.autoencoder, device)
        normalizer = Normalizer.from_state_dict(ae_checkpoint["normalizer"])
        data = WindowData.load(args.windows, normalizer=normalizer)
        if not (np.allclose(data.joint_lower, ae_checkpoint["joint_lower"]) and
                np.allclose(data.joint_upper, ae_checkpoint["joint_upper"]) and
                ae.condition_dim == data.condition.shape[1]):
            raise ValueError("autoencoder checkpoint is incompatible with the active G1/window condition")
    if args.sampler == "flow":
        flow, flow_checkpoint = _load_flow(args.flow, device)
        if not (np.allclose(data.joint_lower, flow_checkpoint["joint_lower"]) and
                np.allclose(data.joint_upper, flow_checkpoint["joint_upper"]) and
                flow.condition_dim == data.condition.shape[1]):
            raise ValueError("flow checkpoint is incompatible with the active G1/window condition")
    if args.sampler == "residual_flow":
        residual_flow, residual_checkpoint = _load_residual_flow(args.residual_flow, device)
        if not (np.allclose(data.joint_lower, residual_checkpoint["joint_lower"]) and
                np.allclose(data.joint_upper, residual_checkpoint["joint_upper"]) and
                residual_flow.condition_dim == data.condition.shape[1] and
                residual_flow.target_dim == data.target.shape[1]):
            raise ValueError("residual flow checkpoint is incompatible with the active G1/window condition")
    candidates = np.flatnonzero(data.split == args.split)
    if not len(candidates):
        raise ValueError(f"no windows in requested split {args.split}")
    index = int(candidates[args.index % len(candidates)])
    actual_primitive = int(data.raw["primitive"][index])
    conditioned_primitive = actual_primitive if args.primitive_id < 0 else int(args.primitive_id)
    if not 0 <= conditioned_primitive < data.primitive_count:
        raise ValueError(f"primitive-id must be in [0, {data.primitive_count - 1}] for this archive")
    if mean_checkpoint is not None and int(mean_checkpoint.get("primitive_id", -1)) >= 0:
        expected_primitive = int(mean_checkpoint["primitive_id"])
        if conditioned_primitive != expected_primitive:
            raise ValueError(f"conditional-mean checkpoint is restricted to primitive-id {expected_primitive}, "
                             f"but the conditioned primitive-id is {conditioned_primitive}")
    # A perception producer can override only the environment tensors while reusing the held-out
    # state/history/command.  This preserves the learned condition layout and keeps the resulting
    # sample traceable to the same SEED window.
    condition_overrides: dict[str, np.ndarray] = {}
    if args.condition_npz is not None:
        with np.load(args.condition_npz) as condition_archive:
            for key in ("corridor", "sdf"):
                if key not in condition_archive.files:
                    raise ValueError(f"{args.condition_npz} must contain '{key}'")
                value = np.asarray(condition_archive[key], dtype=np.float32)
                if key not in data.raw or value.shape != data.raw[key][index].shape:
                    raise ValueError(f"perception {key} shape {value.shape} does not match {data.raw.get(key, np.empty(0)).shape[1:]}")
                condition_overrides[key] = value
        environment_parts = [data.raw["manifold"][index:index + 1].reshape(1, -1)]
        for key in ("corridor", "sdf"):
            if key in data.raw:
                value = condition_overrides.get(key, data.raw[key][index])
                environment_parts.append(value.reshape(1, -1))
        environment = np.concatenate(environment_parts, axis=1).astype(np.float32)
    if args.condition_npz is not None or conditioned_primitive != actual_primitive:
        if args.condition_npz is None:
            environment_parts = [data.raw["manifold"][index:index + 1].reshape(1, -1)]
            for key in ("corridor", "sdf"):
                if key in data.raw:
                    environment_parts.append(data.raw[key][index:index + 1].reshape(1, -1))
            environment = np.concatenate(environment_parts, axis=1).astype(np.float32)
        one_hot = np.eye(data.primitive_count, dtype=np.float32)[[conditioned_primitive]]
        condition_row = np.concatenate([
            normalizer.state(data.raw["state"][index:index + 1]).astype(np.float32),
            normalizer.state(data.raw["history"][index:index + 1]).reshape(1, -1).astype(np.float32),
            one_hot,
            normalizer.manifold(environment).astype(np.float32),
            normalizer.command(data.raw["command"][index:index + 1]).astype(np.float32),
        ], axis=1)
    else:
        condition_row = data.condition[index:index + 1]
    # Produce several independently seeded flow trajectories for the same condition.  The
    # subsequent SONIC/MuJoCo selector is deliberately responsible for deciding which one is
    # executable; candidate 0 is retained as ``generated_ref`` for backwards compatibility.
    condition = torch.as_tensor(condition_row, device=device).expand(args.num_candidates, -1)
    with torch.no_grad():
        if args.sampler == "flow":
            assert flow is not None
            generator = torch.Generator(device=device).manual_seed(args.seed)
            latent = torch.randn((args.num_candidates, flow.latent_dim), device=device, generator=generator)
            dt = 1.0 / args.steps
            for step in range(args.steps):
                t = torch.full((args.num_candidates,), step * dt, device=device)
                latent = latent + dt * flow(latent, t, condition)
            normalized = ae.decode(latent, condition).cpu().numpy().reshape((args.num_candidates,) + data.target_shape)
        elif args.sampler == "residual_flow":
            assert mean is not None and residual_flow is not None and residual_checkpoint is not None
            # Candidate zero is an exact copy of the validated conditional mean.  Remaining
            # candidates are integrated residual fields around that anchor.
            mean_flat = mean(condition[:1]).cpu().numpy()
            mean_normalized = mean_flat.reshape((1,) + data.target_shape)
            if args.num_candidates == 1:
                normalized = mean_normalized
            else:
                generator = torch.Generator(device=device).manual_seed(args.seed)
                residual = torch.randn((args.num_candidates - 1, residual_flow.target_dim),
                                        device=device, generator=generator)
                residual_condition = condition[1:]
                dt = 1.0 / args.steps
                for step in range(args.steps):
                    t = torch.full((args.num_candidates - 1,), step * dt, device=device)
                    residual = residual + dt * residual_flow(residual, t, residual_condition)
                residual = residual * torch.as_tensor(residual_checkpoint["residual_std"], device=device)
                residual = residual.cpu().numpy().reshape((args.num_candidates - 1,) + data.target_shape)
                generated_residual = mean_normalized + residual
                normalized = np.concatenate([mean_normalized, generated_residual], axis=0)
        elif args.sampler == "posterior_mean":
            # Diagnostic upper bound only: it encodes the held-out target before decoding it.
            # This cannot be used at deployment, but separates a poor latent representation
            # from a poor learned flow prior during model development.
            assert ae is not None
            target = torch.as_tensor(data.target[index:index + 1], device=device)
            latent, _ = ae.encode(target, condition[:1])
            latent = latent.expand(args.num_candidates, -1)
            normalized = ae.decode(latent, condition).cpu().numpy().reshape((args.num_candidates,) + data.target_shape)
        else:
            assert mean is not None
            normalized = mean(condition).cpu().numpy().reshape((args.num_candidates,) + data.target_shape)
    generated_candidates = _target_from_model(normalizer.inverse_target(normalized),
                                               data.joint_lower, data.joint_upper).astype(np.float32)
    generated = generated_candidates[0]
    expected_ref = data.raw["target_ref"][index].astype(np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    sample_arrays: dict[str, np.ndarray] = {
        "generated_ref": generated,
        "generated_ref_candidates": generated_candidates,
        "candidate_indices": np.arange(args.num_candidates, dtype=np.int64),
        "expected_ref": expected_ref,
        "expected_exec": data.raw["target_exec"][index],
        "condition": condition_row[0],
        "source_index": np.asarray(index, dtype=np.int64),
        "primitive_id": np.asarray(conditioned_primitive, dtype=np.int64),
        "primitive_name": np.asarray(PRIMITIVE_NAMES[conditioned_primitive]),
        "source_primitive_id": np.asarray(actual_primitive, dtype=np.int64),
    }
    # Keep the environmental condition beside the generated trajectory.  This makes controller
    # validation check the exact corridor/SDF the model was conditioned on, rather than a later
    # lookup that could silently use a different window.
    for key in ("corridor", "sdf"):
        if key in data.raw:
            sample_arrays[f"condition_{key}"] = condition_overrides.get(key, data.raw[key][index])
    # A sample must be traceable back through its window to the accepted SEED replay record.
    # ``clip_index`` resolves through window metadata's clips list; ``source_origin`` is the
    # 30 Hz origin in that record.  Neither is part of the learned condition.
    for key in ("clip_index", "source_origin"):
        if key in data.raw:
            sample_arrays[key] = data.raw[key][index]
    np.savez_compressed(args.out / "sample.npz", **sample_arrays)
    report = {"source_index": index, "split": int(args.split), "steps": args.steps,
              "num_candidates": args.num_candidates, "sampler": args.sampler,
              "primitive": str(sample_arrays["primitive_name"]),
              "source_primitive": PRIMITIVE_NAMES[actual_primitive],
              "generated_shape": list(generated.shape), "flow_checkpoint": (str(args.flow) if flow_checkpoint else None),
              "mean_checkpoint": (str(args.mean_model) if mean_checkpoint else None),
              "residual_flow_checkpoint": (str(args.residual_flow) if residual_checkpoint else None),
              "autoencoder_checkpoint": (str(args.autoencoder) if ae_checkpoint else None),
              "condition_npz": (str(args.condition_npz) if args.condition_npz else None),
              "note": "Generated trajectory is R_ref for SONIC; validate it through seed_replay-style execution before use."}
    for key in ("clip_index", "source_origin"):
        if key in sample_arrays:
            report[key] = int(sample_arrays[key])
    (args.out / "sample.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def _common_training(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/stage2_flow"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--condition-hidden", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", default="auto")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/sample the Stage-2 conditional latent Flow Matching baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    ae = sub.add_parser("train-ae", help="fit the conditional trajectory VAE")
    _common_training(ae)
    ae.add_argument("--latent-dim", type=int, default=64)
    ae.add_argument("--kl-weight", type=float, default=1e-4)
    ae.set_defaults(handler=train_ae)

    mean = sub.add_parser("train-mean", help="fit deterministic conditional trajectory-mean baseline")
    _common_training(mean)
    mean.add_argument("--primitive-id", type=int, default=-1,
                      help="optional raw primitive ID; -1 uses all primitives")
    mean.add_argument("--model-target-field", choices=("target_ref", "target_exec"),
                      default="target_ref",
                      help="training target; target_exec learns SONIC-achieved references")
    mean.set_defaults(handler=train_mean)

    residual = sub.add_parser("train-residual-flow",
                              help="fit stochastic Flow Matching residuals around a mean proposal")
    _common_training(residual)
    residual.add_argument("--mean-model", type=Path, required=True,
                          help="validated conditional_mean.pt used as the zero-residual anchor")
    residual.add_argument("--primitive-id", type=int, default=-1,
                          help="optional raw primitive ID; -1 uses all primitives")
    residual.set_defaults(handler=train_residual_flow)

    flow = sub.add_parser("train-flow", help="fit Flow Matching in the frozen VAE latent space")
    _common_training(flow)
    flow.add_argument("--autoencoder", type=Path, default=None,
                      help="VAE checkpoint (defaults to <out>/autoencoder.pt)")
    flow.set_defaults(handler=train_flow)

    sampler = sub.add_parser("sample", help="sample one R_ref trajectory conditioned on a stored window")
    sampler.add_argument("--windows", type=Path, required=True)
    sampler.add_argument("--autoencoder", type=Path, default=Path("reports/manifold_motion/stage2_flow/autoencoder.pt"))
    sampler.add_argument("--flow", type=Path, default=Path("reports/manifold_motion/stage2_flow/flow.pt"))
    sampler.add_argument("--out", type=Path, default=Path("reports/manifold_motion/stage2_sample"))
    sampler.add_argument("--split", type=int, choices=(0, 1, 2), default=2)
    sampler.add_argument("--index", type=int, default=0, help="index within the selected split")
    sampler.add_argument("--steps", type=int, default=32)
    sampler.add_argument("--num-candidates", type=int, default=1,
                         help="independent flow samples for subsequent SONIC/MuJoCo selection")
    sampler.add_argument("--sampler", choices=("flow", "conditional_mean", "residual_flow", "posterior_mean"), default="flow",
                         help="flow is latent generation; residual_flow samples around a mean anchor; posterior_mean is diagnostic")
    sampler.add_argument("--mean-model", type=Path, default=None,
                         help="conditional_mean.pt required with --sampler conditional_mean/residual_flow")
    sampler.add_argument("--residual-flow", type=Path,
                         default=Path("reports/manifold_motion/stage2_residual_flow/residual_flow.pt"),
                         help="residual_flow.pt required with --sampler residual_flow")
    sampler.add_argument("--seed", type=int, default=20260917)
    sampler.add_argument("--device", default="auto")
    sampler.add_argument("--condition-npz", type=Path, default=None,
                         help="optional perception condition.npz containing corridor and sdf")
    sampler.add_argument("--primitive-id", type=int, default=-1,
                         help="router-selected z_p override; -1 retains the stored window primitive")
    sampler.set_defaults(handler=sample)
    args = parser.parse_args()
    if getattr(args, "epochs", 1) < 1 or getattr(args, "batch_size", 1) < 1:
        parser.error("epochs and batch size must be positive")
    if getattr(args, "num_candidates", 1) < 1:
        parser.error("num-candidates must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
