"""Stage-1 imitation baseline trained from the reverse manifold/action catalogue.

This is deliberately a small, auditable baseline: a static descriptor of the local environment
manifold is mapped to the middle pose of an accepted SONIC execution window.  It tests the first
claim independently of Stage-2 history, Flow sampling and online routing.  The catalogue keeps
the full dynamic tensors for later experiments; this trainer only consumes the environment
manifold and the executed joint target.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manifold_motion.core import constants as C


def manifold_features(corridor: np.ndarray) -> np.ndarray:
    """Return a static, phase-independent descriptor for one ``M_e(t)`` window.

    The descriptor contains mean/min/max centre, semi-axis and yaw plus route displacement and
    yaw change.  It does not contain the action family or any joint target.
    """
    value = np.asarray(corridor, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 7:
        raise ValueError(f"corridor must be [N,T,7], got {value.shape}")
    center = value[:, :, :3]
    semi = value[:, :, 3:6]
    yaw = np.unwrap(value[:, :, 6], axis=1)
    stats = np.concatenate([
        center.mean(axis=1), center.min(axis=1), center.max(axis=1),
        semi.mean(axis=1), semi.min(axis=1), semi.max(axis=1),
        center[:, -1] - center[:, 0],
        (yaw[:, -1] - yaw[:, 0])[:, None],
    ], axis=1)
    return stats.astype(np.float32)


def _metrics(pred: np.ndarray, target: np.ndarray, names: np.ndarray,
             family_ids: np.ndarray) -> dict[str, object]:
    error = pred - target
    by_family: dict[str, dict[str, float]] = {}
    for family_id in sorted(set(int(value) for value in family_ids)):
        mask = family_ids == family_id
        family = str(names[family_id])
        by_family[family] = {
            "rows": int(mask.sum()),
            "mae_rad": float(np.abs(error[mask]).mean()),
            "rmse_rad": float(np.sqrt(np.mean(error[mask] ** 2))),
        }
    return {
        "rows": int(len(target)),
        "mae_rad": float(np.abs(error).mean()),
        "rmse_rad": float(np.sqrt(np.mean(error ** 2))),
        "p95_abs_rad": float(np.percentile(np.abs(error), 95)),
        "by_family": by_family,
    }


def _mlp(torch, input_dim: int, output_dim: int, hidden: int):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden), torch.nn.LayerNorm(hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, output_dim),
    )


def train(args) -> int:
    import torch

    with np.load(args.catalog, allow_pickle=False) as archive:
        corridor = np.asarray(archive["corridor"], dtype=np.float32)
        target = np.asarray(archive[args.target_field], dtype=np.float32)
        split = np.asarray(archive["split"], dtype=np.uint8)
        family = np.asarray(archive["primitive"], dtype=np.int64)
        names = np.asarray(archive["primitive_names"])
    if target.ndim != 3 or target.shape[2] < 29:
        raise ValueError(f"target must be [N,T,>=29], got {target.shape}")
    frame = target.shape[1] // 2 if args.target_frame < 0 else int(args.target_frame)
    if frame < 0 or frame >= target.shape[1]:
        raise ValueError(f"target frame {frame} is outside horizon {target.shape[1]}")
    x = manifold_features(corridor)
    default = C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB].astype(np.float32)
    y = target[:, frame, :29] - default[None]
    train_mask, val_mask, test_mask = split == 0, split == 1, split == 2
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("catalog must contain train, validation and test rows")
    x_mean, x_std = x[train_mask].mean(axis=0), x[train_mask].std(axis=0)
    x_std = np.maximum(x_std, 1e-5)
    x_norm = (x - x_mean) / x_std
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    model = _mlp(torch, x.shape[1], y.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    x_train = torch.as_tensor(x_norm[train_mask], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y[train_mask], dtype=torch.float32, device=device)
    best = float("inf")
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x_train))
        model.train()
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            pred = model(x_train[idx])
            loss = torch.nn.functional.smooth_l1_loss(pred, y_train[idx])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.as_tensor(x_norm[val_mask], dtype=torch.float32, device=device)).cpu().numpy()
        val_metrics = _metrics(val_pred, y[val_mask], names, family[val_mask])
        if val_metrics["mae_rad"] < best:
            best = float(val_metrics["mae_rad"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"epoch {epoch:4d} train_smooth_l1 {np.mean(losses):.5f} "
                  f"val_mae {val_metrics['mae_rad']:.4f} ({time.perf_counter() - started:.1f}s)", flush=True)
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        train_pred = model(torch.as_tensor(x_norm[train_mask], dtype=torch.float32, device=device)).cpu().numpy()
        val_pred = model(torch.as_tensor(x_norm[val_mask], dtype=torch.float32, device=device)).cpu().numpy()
        test_pred = model(torch.as_tensor(x_norm[test_mask], dtype=torch.float32, device=device)).cpu().numpy()
    metrics = {
        "train": _metrics(train_pred, y[train_mask], names, family[train_mask]),
        "validation": _metrics(val_pred, y[val_mask], names, family[val_mask]),
        "test": _metrics(test_pred, y[test_mask], names, family[test_mask]),
    }
    global_mean = y[train_mask].mean(axis=0)
    family_mean = {int(i): y[train_mask & (family == i)].mean(axis=0)
                   for i in sorted(set(int(v) for v in family[train_mask]))}
    metrics["test_global_mean_baseline"] = _metrics(
        np.repeat(global_mean[None], int(test_mask.sum()), axis=0), y[test_mask], names, family[test_mask])
    metrics["test_family_mean_baseline"] = _metrics(
        np.stack([family_mean.get(int(i), global_mean) for i in family[test_mask]]),
        y[test_mask], names, family[test_mask])
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "manifold-motion.stage1.manifold-imitation.v1",
        "model": model.state_dict(), "input_mean": x_mean, "input_std": x_std,
        "input_dim": int(x.shape[1]), "output_dim": int(y.shape[1]),
        "hidden": int(args.hidden),
        "target_field": args.target_field, "target_frame": frame,
        "primitive_names": names.tolist(), "default_pose": default,
    }
    torch.save(checkpoint, args.out / "manifold_imitation.pt")
    report = {
        "schema": "manifold-motion.stage1.manifold-imitation-report.v1",
        "catalog": str(args.catalog), "target_field": args.target_field, "target_frame": frame,
        "feature_dim": int(x.shape[1]), "train_rows": int(train_mask.sum()),
        "validation_rows": int(val_mask.sum()), "test_rows": int(test_mask.sum()),
        "hidden": args.hidden, "epochs": args.epochs, "seed": args.seed,
        "metrics": metrics,
        "interpretation": "M_e-only supervised imitation; no SONIC fine-tuning or Stage-2 history is used",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "test": metrics["test"],
                      "global_baseline": metrics["test_global_mean_baseline"],
                      "family_baseline": metrics["test_family_mean_baseline"]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/stage1_manifold_imitation_v1"))
    parser.add_argument("--target-field", choices=("target_ref", "target_exec"), default="target_exec")
    parser.add_argument("--target-frame", type=int, default=-1, help="middle frame by default")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden < 1:
        parser.error("epochs, batch-size and hidden must be positive")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
