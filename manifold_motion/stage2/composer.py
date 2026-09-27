"""SLAM- and self-manifold-conditioned high-level skill composer.

The composer deliberately does *not* receive the ground-truth primitive as an input.  It predicts
the next SEED skill family from the live robot state/history, route command, the environment
ellipsoid corridor and SDF supplied by SLAM, and the robot's executed self-manifold.  This avoids
the label leakage present in trajectory models where a primitive one-hot is part of the condition.

The reverse-corridor SEED archive is a pre-training source.  Its fields have the same shapes as
the online perception contract, but its corridor/SDF were reconstructed from successful MuJoCo
execution and are therefore reported as reverse-synthetic rather than real sensor observations.
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
from torch.nn import functional as F

from manifold_motion.stage2.flow import _MLP, _batches, _device, _torch_load


@dataclass
class ComposerNormalizer:
    state_mean: np.ndarray
    state_std: np.ndarray
    corridor_mean: np.ndarray
    corridor_std: np.ndarray
    self_mean: np.ndarray
    self_std: np.ndarray
    sdf_mean: np.ndarray
    sdf_std: np.ndarray
    manifold_mean: np.ndarray
    manifold_std: np.ndarray
    command_mean: np.ndarray
    command_std: np.ndarray

    @staticmethod
    def _statistics(value: np.ndarray, axes: int | tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        mean = value.mean(axis=axes)
        std = value.std(axis=axes)
        return mean.astype(np.float32), np.where(std < 1e-6, 1.0, std).astype(np.float32)

    @classmethod
    def fit(cls, raw: dict[str, np.ndarray], train: np.ndarray) -> "ComposerNormalizer":
        state = np.concatenate((raw["state"][train], raw["history"][train].reshape(-1, raw["state"].shape[-1])))
        state_mean, state_std = cls._statistics(state, 0)
        corridor_mean, corridor_std = cls._statistics(raw["corridor"][train], (0, 1))
        self_mean, self_std = cls._statistics(raw["self_manifold"][train], (0, 1))
        # A scalar SDF normalization preserves metric relationships between neighboring voxels.
        sdf_mean, sdf_std = cls._statistics(raw["sdf"][train], tuple(range(raw["sdf"].ndim)))
        manifold_mean, manifold_std = cls._statistics(raw["manifold"][train], 0)
        command_mean, command_std = cls._statistics(raw["command"][train], 0)
        return cls(state_mean, state_std, corridor_mean, corridor_std, self_mean, self_std,
                   np.asarray(sdf_mean), np.asarray(sdf_std), manifold_mean, manifold_std,
                   command_mean, command_std)

    def normalize(self, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "state": ((raw["state"] - self.state_mean) / self.state_std).astype(np.float32),
            "history": ((raw["history"] - self.state_mean) / self.state_std).astype(np.float32),
            "corridor": ((raw["corridor"] - self.corridor_mean) / self.corridor_std).astype(np.float32),
            "self_manifold": ((raw["self_manifold"] - self.self_mean) / self.self_std).astype(np.float32),
            "sdf": ((raw["sdf"] - self.sdf_mean) / self.sdf_std).astype(np.float32),
            "manifold": ((raw["manifold"] - self.manifold_mean) / self.manifold_std).astype(np.float32),
            "command": ((raw["command"] - self.command_mean) / self.command_std).astype(np.float32),
        }

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name), dtype=np.float32) for name in self.__dataclass_fields__}

    @classmethod
    def from_state_dict(cls, values: dict[str, np.ndarray]) -> "ComposerNormalizer":
        return cls(**{name: np.asarray(values[name], dtype=np.float32) for name in cls.__dataclass_fields__})


@dataclass
class ComposerData:
    inputs: dict[str, np.ndarray]
    labels: np.ndarray
    split: np.ndarray
    primitive_count: int
    primitive_names: list[str]
    supported_mask: np.ndarray
    normalizer: ComposerNormalizer
    raw: dict[str, np.ndarray]

    @classmethod
    def load(cls, path: Path, normalizer: ComposerNormalizer | None = None) -> "ComposerData":
        required = {"state", "history", "corridor", "sdf", "self_manifold", "manifold",
                    "command", "primitive", "split", "primitive_count", "primitive_names"}
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{path} lacks composer fields {sorted(missing)}")
            raw = {key: np.asarray(archive[key]) for key in required}
        continuous = required - {"primitive", "split", "primitive_count", "primitive_names"}
        if not all(np.isfinite(raw[key]).all() for key in continuous):
            raise ValueError(f"{path} contains non-finite composer inputs")
        count = len(raw["state"])
        if any(len(raw[key]) != count for key in required - {"primitive_count", "primitive_names"}):
            raise ValueError("composer arrays disagree on their sample count")
        primitive_count = int(np.asarray(raw["primitive_count"]).reshape(-1)[0])
        labels = raw["primitive"].astype(np.int64)
        if labels.min() < 0 or labels.max() >= primitive_count:
            raise ValueError("primitive IDs fall outside primitive_count")
        names = [str(value) for value in raw["primitive_names"].tolist()]
        if len(names) != primitive_count:
            raise ValueError("primitive_names width disagrees with primitive_count")
        train = raw["split"] == 0
        if not train.any():
            raise ValueError("composer windows have no training split")
        normalizer = normalizer or ComposerNormalizer.fit(raw, train)
        supported = np.zeros(primitive_count, dtype=bool)
        supported[np.unique(labels)] = True
        return cls(normalizer.normalize(raw), labels, raw["split"].astype(np.int64),
                   primitive_count, names, supported, normalizer, raw)


class EnvironmentSkillComposer(nn.Module):
    """Multi-branch encoder for state, temporal geometry and local SLAM voxels."""

    def __init__(self, state_dim: int, command_dim: int, manifold_dim: int,
                 primitive_count: int, sdf_shape: tuple[int, int, int], hidden: int = 96):
        super().__init__()
        self.state_dim = state_dim
        self.command_dim = command_dim
        self.manifold_dim = manifold_dim
        self.primitive_count = primitive_count
        self.sdf_shape = tuple(int(value) for value in sdf_shape)
        self.hidden = hidden
        self.state_encoder = _MLP(state_dim, hidden, hidden, layers=2)
        self.history_encoder = nn.GRU(state_dim, hidden, batch_first=True)
        self.corridor_encoder = nn.GRU(7, hidden, batch_first=True)
        self.self_encoder = nn.GRU(3, hidden // 2, batch_first=True)
        pooled_shape = tuple((value - 2) // 2 + 1 for value in self.sdf_shape)
        self.sdf_encoder = _MLP(int(np.prod(pooled_shape)), hidden // 2, hidden, layers=2)
        self.manifold_encoder = _MLP(manifold_dim, hidden // 4, hidden // 2, layers=2)
        self.command_encoder = _MLP(command_dim, hidden // 2, hidden, layers=2)
        fused = hidden * 4 + hidden // 2 + hidden // 4
        self.fusion = nn.Sequential(nn.Linear(fused, hidden * 2), nn.SiLU(), nn.Dropout(0.1),
                                    nn.Linear(hidden * 2, hidden), nn.SiLU())
        self.classifier = nn.Linear(hidden, primitive_count)

    def forward(self, state: torch.Tensor, history: torch.Tensor, corridor: torch.Tensor,
                sdf: torch.Tensor, self_manifold: torch.Tensor, manifold: torch.Tensor,
                command: torch.Tensor) -> torch.Tensor:
        state_feature = self.state_encoder(state)
        _, history_feature = self.history_encoder(history)
        _, corridor_feature = self.corridor_encoder(corridor)
        _, self_feature = self.self_encoder(self_manifold)
        sdf_pooled = F.avg_pool3d(sdf.unsqueeze(1), kernel_size=2, stride=2).flatten(1)
        sdf_feature = self.sdf_encoder(sdf_pooled)
        manifold_feature = self.manifold_encoder(manifold)
        command_feature = self.command_encoder(command)
        fused = torch.cat((state_feature, history_feature[-1], corridor_feature[-1],
                           self_feature[-1], sdf_feature, manifold_feature, command_feature), dim=-1)
        return self.classifier(self.fusion(fused))

    def architecture(self) -> dict[str, Any]:
        return {"state_dim": self.state_dim, "command_dim": self.command_dim,
                "manifold_dim": self.manifold_dim, "primitive_count": self.primitive_count,
                "sdf_shape": self.sdf_shape, "hidden": self.hidden}


def _forward(model: EnvironmentSkillComposer, data: ComposerData, indices: np.ndarray,
             device: torch.device) -> torch.Tensor:
    return _forward_inputs(model, data.inputs, indices, device)


def _forward_inputs(model: EnvironmentSkillComposer, value: dict[str, np.ndarray],
                    indices: np.ndarray, device: torch.device) -> torch.Tensor:
    tensors = [torch.as_tensor(value[key][indices], device=device) for key in
               ("state", "history", "corridor", "sdf", "self_manifold", "manifold", "command")]
    return model(*tensors)


def _batch_forward(model: EnvironmentSkillComposer, value: dict[str, np.ndarray],
                   device: torch.device) -> torch.Tensor:
    count = len(value["state"])
    return _forward_inputs(model, value, np.arange(count), device)


def _metrics(model: EnvironmentSkillComposer, data: ComposerData, indices: np.ndarray,
             device: torch.device, batch_size: int) -> dict[str, Any]:
    if not len(indices):
        return {"windows": 0}
    model.eval()
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            logits = _forward(model, data, batch, device)
            # Families rejected by physical replay can never be routed online.
            logits[:, ~torch.as_tensor(data.supported_mask, device=device)] = -1e9
            probabilities.append(torch.softmax(logits, -1).cpu().numpy())
            predictions.append(torch.argmax(logits, -1).cpu().numpy())
    probability = np.concatenate(probabilities)
    prediction = np.concatenate(predictions)
    target = data.labels[indices]
    top_k = min(3, data.primitive_count)
    top3 = np.argpartition(probability, -top_k, axis=1)[:, -top_k:]
    classes = sorted(np.unique(target).tolist())
    per_family = []
    f1_values = []
    for family_id in classes:
        true = target == family_id
        pred = prediction == family_id
        tp = int(np.sum(true & pred)); fp = int(np.sum(~true & pred)); fn = int(np.sum(true & ~pred))
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_family.append({"family_id": int(family_id), "name": data.primitive_names[family_id],
                           "windows": int(true.sum()), "accuracy": float(np.mean(prediction[true] == family_id)),
                           "precision": precision, "recall": recall, "f1": f1})
    return {"windows": int(len(indices)), "top1_accuracy": float(np.mean(prediction == target)),
            "top3_accuracy": float(np.mean(np.any(top3 == target[:, None], axis=1))),
            "macro_f1": float(np.mean(f1_values)), "families": per_family}


def _environment_ablation(model: EnvironmentSkillComposer, data: ComposerData,
                          indices: np.ndarray, device: torch.device,
                          batch_size: int, seed: int) -> dict[str, Any]:
    """Quantify whether predictions actually depend on M_e/M_self rather than proprioception.

    Zero is the train-split normalized mean.  The shuffled case preserves every marginal but
    breaks the state-to-environment pairing, which is a stronger leakage test than deleting one
    scalar aperture feature.
    """
    if not len(indices):
        return {"windows": 0}

    def shadow(inputs: dict[str, np.ndarray]) -> Any:
        return type("_ComposerAblation", (), {
            "inputs": inputs, "labels": data.labels, "supported_mask": data.supported_mask,
            "primitive_count": data.primitive_count, "primitive_names": data.primitive_names,
        })()

    geometry_keys = ("corridor", "sdf", "self_manifold", "manifold")
    masked = {key: value.copy() for key, value in data.inputs.items()}
    for key in geometry_keys:
        masked[key][indices] = 0.0
    environment_only = {key: value.copy() for key, value in data.inputs.items()}
    environment_only["state"][indices] = 0.0
    environment_only["history"][indices] = 0.0
    rng = np.random.default_rng(seed)
    donor = rng.permutation(indices)
    shuffled = {key: value.copy() for key, value in data.inputs.items()}
    for key in geometry_keys:
        shuffled[key][indices] = data.inputs[key][donor]
    full = _metrics(model, data, indices, device, batch_size)
    no_geometry = _metrics(model, shadow(masked), indices, device, batch_size)
    environment = _metrics(model, shadow(environment_only), indices, device, batch_size)
    permuted = _metrics(model, shadow(shuffled), indices, device, batch_size)
    return {
        "windows": int(len(indices)),
        "full_top1": full["top1_accuracy"],
        "geometry_masked_top1": no_geometry["top1_accuracy"],
        "environment_only_top1": environment["top1_accuracy"],
        "geometry_permuted_top1": permuted["top1_accuracy"],
        "full_minus_masked": full["top1_accuracy"] - no_geometry["top1_accuracy"],
        "full_minus_permuted": full["top1_accuracy"] - permuted["top1_accuracy"],
    }


def train(args: argparse.Namespace) -> int:
    device = _device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data = ComposerData.load(args.windows)
    train_idx = np.flatnonzero(data.split == 0); val_idx = np.flatnonzero(data.split == 1)
    test_idx = np.flatnonzero(data.split == 2)
    model = EnvironmentSkillComposer(data.raw["state"].shape[-1], data.raw["command"].shape[-1],
                                     data.raw["manifold"].shape[-1], data.primitive_count,
                                     tuple(data.raw["sdf"].shape[1:]), args.hidden).to(device)
    counts = np.bincount(data.labels[train_idx], minlength=data.primitive_count).astype(np.float64)
    weights = np.zeros(data.primitive_count, dtype=np.float32)
    observed = counts > 0
    weights[observed] = np.sqrt(counts[observed].sum() / counts[observed])
    weights[observed] /= weights[observed].mean()
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed); args.out.mkdir(parents=True, exist_ok=True)
    history = []; best = -float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in _batches(train_idx, args.batch_size, rng):
            logits = _forward(model, data, batch, device)
            target = torch.as_tensor(data.labels[batch], device=device)
            full_loss = criterion(logits, target)
            # Counterfactual pairing keeps the receiver's proprioceptive state/history while
            # replacing its geometry and task command with a donor window.  The donor family
            # becomes the target.  This explicitly prevents a classifier from treating M_e as
            # decorative context while memorising motion phase from state alone.
            donor = rng.permutation(batch)
            counterfactual = {key: data.inputs[key][batch].copy() for key in data.inputs}
            for key in ("corridor", "sdf", "self_manifold", "manifold", "command"):
                counterfactual[key] = data.inputs[key][donor].copy()
            counterfactual_logits = _batch_forward(model, counterfactual, device)
            counterfactual_loss = criterion(
                counterfactual_logits, torch.as_tensor(data.labels[donor], device=device))
            environment_only = {key: data.inputs[key][batch].copy() for key in data.inputs}
            environment_only["state"][:] = 0.0
            environment_only["history"][:] = 0.0
            environment_logits = _batch_forward(model, environment_only, device)
            environment_loss = criterion(environment_logits, target)
            loss = (full_loss + args.counterfactual_weight * counterfactual_loss
                    + args.environment_only_weight * environment_loss)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append((float(loss.detach().cpu()), float(full_loss.detach().cpu()),
                           float(counterfactual_loss.detach().cpu()),
                           float(environment_loss.detach().cpu())))
        validation = _metrics(model, data, val_idx if len(val_idx) else test_idx, device, args.batch_size)
        row = {"epoch": epoch, "train_loss": float(np.mean([value[0] for value in losses])),
               "train_full_loss": float(np.mean([value[1] for value in losses])),
               "train_counterfactual_loss": float(np.mean([value[2] for value in losses])),
               "train_environment_only_loss": float(np.mean([value[3] for value in losses])),
               "validation_top1": validation.get("top1_accuracy"),
               "validation_top3": validation.get("top3_accuracy"),
               "validation_macro_f1": validation.get("macro_f1")}
        history.append(row)
        score = float(validation.get("macro_f1", -float("inf")))
        if score > best:
            best = score
            torch.save({"schema": "manifold-motion.environment-skill-composer.v2",
                        "architecture": model.architecture(), "model_state": model.state_dict(),
                        "normalizer": data.normalizer.state_dict(), "primitive_names": data.primitive_names,
                        "supported_mask": data.supported_mask, "epoch": epoch,
                        "counterfactual_weight": args.counterfactual_weight,
                        "environment_only_weight": args.environment_only_weight,
                        "input_contract": "state+history+M_e(corridor,SDF)+M_self+command; no primitive input"},
                       args.out / "composer.pt")
        if epoch == 1 or epoch % max(1, args.log_every) == 0 or epoch == args.epochs:
            print(json.dumps(row), flush=True)
    checkpoint = _torch_load(args.out / "composer.pt", device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = {"train": _metrics(model, data, train_idx, device, args.batch_size),
               "validation": _metrics(model, data, val_idx, device, args.batch_size),
               "test": _metrics(model, data, test_idx, device, args.batch_size)}
    report = {"schema": "manifold-motion.environment-skill-composer-report.v2",
              "windows": str(args.windows), "environment_provenance": "reverse_synthesized_from_R_exec",
              "best_epoch": int(checkpoint["epoch"]), "epochs": args.epochs,
              "primitive_count": data.primitive_count, "supported_family_count": int(data.supported_mask.sum()),
              "supported_families": [data.primitive_names[i] for i in np.flatnonzero(data.supported_mask)],
              "split_counts": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
              "training_objective": {
                  "counterfactual_weight": args.counterfactual_weight,
                  "environment_only_weight": args.environment_only_weight,
                  "counterfactual_contract": "hold state/history; swap M_e/SDF/M_self/manifold/command and donor label",
              },
              "environment_ablation": _environment_ablation(
                  model, data, test_idx, device, args.batch_size, args.seed + 17),
              "metrics": metrics, "history": history, "checkpoint": "composer.pt"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"best_epoch": report["best_epoch"], "supported_family_count": report["supported_family_count"],
                      "test": metrics["test"] | {"families": "see report.json"}}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--windows", type=Path, required=True)
    train_parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/composer_v1"))
    train_parser.add_argument("--epochs", type=int, default=60)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--hidden", type=int, default=96)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--counterfactual-weight", type=float, default=0.0)
    train_parser.add_argument("--environment-only-weight", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=20260927)
    train_parser.add_argument("--log-every", type=int, default=5)
    train_parser.add_argument("--device", default="cpu")
    train_parser.set_defaults(handler=train)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
