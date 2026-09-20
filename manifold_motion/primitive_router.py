"""Learn the Stage-2 primitive decision ``p(z_p | M, c)``.

The trajectory generators consume a primitive token but do not choose it themselves.  This
module is the missing router between an environment manifold/safe corridor and the corresponding
per-primitive dynamic model.  It deliberately trains only on primitives with accepted
SONIC/MuJoCo motion windows; unsupported labels such as crawl are absent from its output space
instead of being sampled optimistically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .seed_windows import PRIMITIVE_NAMES


def _features(raw: dict[str, np.ndarray], *, include_command: bool = True) -> np.ndarray:
    """Summarize the environment manifold, optionally appending task command ``c``.

    The normal router learns :math:`p(z_p|M,c)`.  A geometry-only ablation is deliberately
    supported for counterfactual tests that hold the initial robot state and command fixed while
    changing just ``M``.  Neither variant ever receives joint state, future route centres, or a
    primitive label.
    """
    if "corridor" not in raw or "sdf" not in raw:
        raise ValueError("primitive routing requires corridor and sdf environment conditions")
    semi = raw["corridor"][:, :, 3:6].astype(np.float32)
    # Aperture changes are the manifold evidence for stand/crouch/low motion.  Route-centre
    # positions are deliberately excluded: knowing a recorded future path would leak the label.
    geometry = np.concatenate([
        semi.mean(axis=1), semi.min(axis=1), semi.max(axis=1),
        np.quantile(semi, 0.1, axis=1), np.quantile(semi, 0.9, axis=1),
        raw["sdf"].reshape(len(semi), -1).mean(axis=1, keepdims=True),
        (raw["sdf"] > 0).reshape(len(semi), -1).mean(axis=1, keepdims=True),
    ], axis=1)
    if include_command:
        return np.concatenate([geometry, raw["command"].astype(np.float32)], axis=1).astype(np.float32)
    return geometry.astype(np.float32)


class Router(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 96):
        super().__init__()
        self.input_dim, self.output_dim, self.hidden = input_dim, output_dim, hidden
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Linear(hidden, output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)

    def architecture(self) -> dict[str, int]:
        return {"input_dim": self.input_dim, "output_dim": self.output_dim, "hidden": self.hidden}


def _load_windows(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        required = {"primitive", "split", "corridor", "sdf", "command"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        return {key: np.asarray(archive[key]) for key in required}


def _normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / std


def _metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = logits.argmax(dim=1)
    return {"accuracy": float((prediction == target).float().mean().cpu()),
            "loss": float(nn.functional.cross_entropy(logits, target).cpu())}


def train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    raw = _load_windows(args.windows)
    feature = _features(raw, include_command=not args.geometry_only)
    train_mask, val_mask, test_mask = (raw["split"] == value for value in (0, 1, 2))
    active = np.unique(raw["primitive"][train_mask]).astype(np.int64)
    if not len(active):
        raise ValueError("no training windows")
    target_lookup = {int(primitive): index for index, primitive in enumerate(active.tolist())}
    target = np.asarray([target_lookup.get(int(value), -1) for value in raw["primitive"]], dtype=np.int64)
    train = np.flatnonzero(train_mask & (target >= 0))
    validation = np.flatnonzero(val_mask & (target >= 0))
    test = np.flatnonzero(test_mask & (target >= 0))
    mean, std = feature[train].mean(0), feature[train].std(0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    feature = _normalize(feature, mean, std).astype(np.float32)
    model = Router(feature.shape[1], len(active), args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best = float("inf")
    history: list[dict[str, float]] = []
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train)
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            x = torch.as_tensor(feature[indices])
            y = torch.as_tensor(target[indices])
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            evaluate = validation if len(validation) else train
            metric = _metrics(model(torch.as_tensor(feature[evaluate])), torch.as_tensor(target[evaluate]))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": metric["loss"],
               "validation_accuracy": metric["accuracy"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metric["loss"] < best:
            best = metric["loss"]
            torch.save({"kind": "manifold_primitive_router", "architecture": model.architecture(),
                        "model_state": model.state_dict(), "feature_mean": mean.astype(np.float32),
                        "feature_std": std, "active_primitive_ids": active,
                        "active_primitive_names": [PRIMITIVE_NAMES[int(value)] for value in active],
                        "include_command": bool(not args.geometry_only),
                        "windows": str(args.windows),
                        "note": "Only accepted-data primitive IDs are routable."}, args.out / "router.pt")
    model.eval()
    report: dict[str, Any] = {"active_primitive_ids": active.tolist(),
                              "active_primitive_names": [PRIMITIVE_NAMES[int(value)] for value in active],
                              "train_windows": int(len(train)), "validation_windows": int(len(validation)),
                              "test_windows": int(len(test))}
    with torch.no_grad():
        for name, indices in (("train", train), ("validation", validation), ("test", test)):
            if len(indices):
                report[name] = _metrics(model(torch.as_tensor(feature[indices])), torch.as_tensor(target[indices]))
    (args.out / "router_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.out / "router_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def _load_router(path: Path) -> tuple[Router, dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("kind") != "manifold_primitive_router":
        raise ValueError(f"{path} is not a manifold primitive router")
    model = Router(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def route(args: argparse.Namespace) -> int:
    raw = _load_windows(args.windows)
    model, checkpoint = _load_router(args.model)
    candidates = np.flatnonzero(raw["split"] == args.split)
    if not len(candidates):
        raise ValueError(f"no split {args.split} windows")
    index = int(candidates[args.index % len(candidates)])
    feature = _features(raw, include_command=bool(checkpoint.get("include_command", True)))[index:index + 1]
    x = torch.as_tensor(_normalize(feature, checkpoint["feature_mean"], checkpoint["feature_std"]))
    probability = torch.softmax(model(x), dim=1)[0].detach().numpy()
    active = np.asarray(checkpoint["active_primitive_ids"], dtype=np.int64)
    order = np.argsort(probability)[::-1]
    report = {"source_index": index, "split": int(args.split),
              "true_primitive": PRIMITIVE_NAMES[int(raw["primitive"][index])],
              "predicted_primitive_id": int(active[order[0]]),
              "predicted_primitive": PRIMITIVE_NAMES[int(active[order[0]])],
              "confidence": float(probability[order[0]]),
              "probabilities": [{"primitive": PRIMITIVE_NAMES[int(active[i])], "probability": float(probability[i])}
                                for i in order]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="learn or evaluate M,c -> primitive routing")
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--windows", type=Path, required=True)
    train_parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/primitive_router"))
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--hidden", type=int, default=96)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--geometry-only", action="store_true",
                              help="learn p(z_p|M) for a fixed-command counterfactual ablation")
    train_parser.add_argument("--seed", type=int, default=20260918)
    train_parser.set_defaults(handler=train)
    route_parser = sub.add_parser("route")
    route_parser.add_argument("--windows", type=Path, required=True)
    route_parser.add_argument("--model", type=Path, required=True)
    route_parser.add_argument("--split", type=int, choices=(0, 1, 2), default=2)
    route_parser.add_argument("--index", type=int, default=0)
    route_parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/primitive_route.json"))
    route_parser.set_defaults(handler=route)
    args = parser.parse_args()
    if getattr(args, "epochs", 1) < 1 or getattr(args, "batch_size", 1) < 1:
        parser.error("epochs and batch size must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
