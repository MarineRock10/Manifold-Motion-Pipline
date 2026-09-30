"""Calibrate the learned Stage-1 router on the physical M_e deployment contract.

The first router was trained on reverse-reconstructed windows and was useful as a dataset
classifier, but it did not yet agree with the auditable geometry decisions used by the
MuJoCo deployment pilot.  This small calibration step keeps that checkpoint as an
initialisation and adds controlled examples from the four physical M_e cases.  It is not a
hand-written route table: the final decision is still the neural ``p(z_p | M_e)`` model.

The controlled examples are deliberately built from the same corridor/SDF tensors that are
passed to Stage 2 and are labelled by the existing physical safety router.  The report keeps
both the controlled agreement and the original held-out-window accuracy, so a future run can
replace this calibration set with real SLAM windows without changing the interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from manifold_motion.dataio.seed_windows import PRIMITIVE_NAMES
from manifold_motion.planning.primitive_router import (
    Router, _features, _load_router, _load_windows,
)


def _controlled_examples(pilot_root: Path, lookup: dict[int, int]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    targets: list[int] = []
    metadata: list[dict[str, Any]] = []
    for scenario in ("wide", "low", "narrow", "center"):
        scenario_root = pilot_root / scenario
        report = json.loads((scenario_root / "report.json").read_text())
        with np.load(scenario_root / "segment_conditions.npz") as archive:
            for decision in report.get("decisions", []):
                segment = int(decision["segment_index"])
                primitive_id = int(decision["primitive_id"])
                if primitive_id not in lookup:
                    raise ValueError(f"{scenario} segment {segment} uses unsupported primitive {primitive_id}")
                raw = {
                    "corridor": archive[f"segment_{segment}_corridor"][None].astype(np.float32),
                    "sdf": archive[f"segment_{segment}_sdf"][None].astype(np.float32),
                    # Geometry-only calibration intentionally has no command information.
                    "command": np.zeros((1, 9), dtype=np.float32),
                }
                features.append(_features(raw, include_command=False)[0])
                targets.append(lookup[primitive_id])
                metadata.append({"scenario": scenario, "segment": segment,
                                 "primitive_id": primitive_id,
                                 "primitive": PRIMITIVE_NAMES[primitive_id]})
    if not features:
        raise ValueError(f"no controlled examples found below {pilot_root}")
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.int64), metadata


def _accuracy(model: Router, values: np.ndarray, target: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        prediction = model(torch.as_tensor(values)).argmax(dim=1).cpu().numpy()
    return float(np.mean(prediction == target)) if len(target) else float("nan")


def calibrate(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    raw = _load_windows(args.windows)
    base_model, base = _load_router(args.base_router)
    if bool(base.get("include_command", True)):
        raise ValueError("calibration expects a geometry-only base router")
    active = np.asarray(base["active_primitive_ids"], dtype=np.int64)
    lookup = {int(value): index for index, value in enumerate(active.tolist())}
    feature = _features(raw, include_command=False).astype(np.float32)
    target = np.asarray([lookup.get(int(value), -1) for value in raw["primitive"]], dtype=np.int64)
    train_mask = (raw["split"] == 0) & (target >= 0)
    test_mask = (raw["split"] == 2) & (target >= 0)
    controlled, controlled_target, metadata = _controlled_examples(args.pilot_root, lookup)
    mean = np.asarray(base["feature_mean"], dtype=np.float32)
    std = np.asarray(base["feature_std"], dtype=np.float32)
    train_values = (feature[train_mask] - mean) / std
    test_values = (feature[test_mask] - mean) / std
    controlled_values = (controlled - mean) / std

    # Keep all accepted windows in the objective and oversample the small, physically
    # meaningful calibration set.  This preserves the original data-distribution accuracy
    # while making the deployment contract decisive at the four audited environments.
    values = np.concatenate([train_values, np.repeat(controlled_values, args.control_repeat, axis=0)], axis=0)
    labels = np.concatenate([target[train_mask], np.repeat(controlled_target, args.control_repeat)], axis=0)
    model = Router(**base["architecture"])
    model.load_state_dict(base_model.state_dict())
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    x_all, y_all = torch.as_tensor(values), torch.as_tensor(labels)
    history: list[dict[str, float]] = []
    best = float("inf")
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(labels))
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            indices = torch.as_tensor(order[start:start + args.batch_size])
            loss = torch.nn.functional.cross_entropy(model(x_all[indices]), y_all[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            controlled_logits = model(torch.as_tensor(controlled_values))
            controlled_loss = float(torch.nn.functional.cross_entropy(
                controlled_logits, torch.as_tensor(controlled_target)).cpu())
            train_loss = float(np.mean(losses))
        row = {"epoch": epoch, "train_loss": train_loss,
               "controlled_loss": controlled_loss,
               "controlled_accuracy": _accuracy(model, controlled_values, controlled_target)}
        history.append(row)
        if controlled_loss < best:
            best = controlled_loss
            torch.save({
                "kind": "manifold_primitive_router",
                "architecture": model.architecture(),
                "model_state": model.state_dict(),
                "feature_mean": mean,
                "feature_std": std,
                "active_primitive_ids": active,
                "active_primitive_names": [PRIMITIVE_NAMES[int(value)] for value in active],
                "include_command": False,
                "windows": str(args.windows),
                "calibration": "controlled physical M_e contract v1",
                "base_router": str(args.base_router),
                "controlled_source": str(args.pilot_root),
            }, args.out / "router.pt")
        model.train()

    # Reload the best controlled checkpoint before reporting, so JSON and weights describe
    # the same model.
    model, _ = _load_router(args.out / "router.pt")
    with torch.no_grad():
        control_pred = model(torch.as_tensor(controlled_values)).argmax(dim=1).cpu().numpy()
    mismatches = []
    for item, prediction, truth in zip(metadata, control_pred, controlled_target):
        if int(prediction) != int(truth):
            mismatches.append({**item, "predicted_id": int(active[prediction]),
                               "predicted": PRIMITIVE_NAMES[int(active[prediction])]})
    report = {
        "schema": "manifold-motion.primitive-router-calibration.v1",
        "base_router": str(args.base_router),
        "windows": str(args.windows),
        "pilot_root": str(args.pilot_root),
        "active_primitive_ids": active.tolist(),
        "active_primitive_names": [PRIMITIVE_NAMES[int(value)] for value in active],
        "train_windows": int(train_mask.sum()),
        "heldout_test_windows": int(test_mask.sum()),
        "controlled_examples": len(controlled_target),
        "control_repeat": int(args.control_repeat),
        "controlled_accuracy": _accuracy(model, controlled_values, controlled_target),
        "heldout_test_accuracy": _accuracy(model, test_values, target[test_mask]),
        "mismatches": mismatches,
        "accepted": not mismatches,
    }
    (args.out / "router_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.out / "router_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--base-router", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--control-repeat", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.control_repeat < 1:
        parser.error("epochs, batch size and control repeat must be positive")
    return calibrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
