"""Train the first Stage-2 dynamic model with a predicted Stage-1 primitive distribution."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from manifold_motion.stage2.flow import (WindowData, ConditionalTrajectoryMean, _batches, _device,
                                         _target_from_model, _trajectory_mean_loss)
from manifold_motion.stage2.stage1_router_bridge import predicted_primitive_probabilities, replace_primitive_condition


def train(args) -> int:
    import torch

    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed); device = _device(args.device)
    data = WindowData.load(args.windows, model_target_field=args.model_target_field)
    probabilities = predicted_primitive_probabilities(args.windows, args.stage1_checkpoint, str(device))
    data = replace_primitive_condition(data, probabilities)
    train_ids = np.flatnonzero(data.split == 0); val_ids = np.flatnonzero(data.split == 1); test_ids = np.flatnonzero(data.split == 2)
    model = ConditionalTrajectoryMean(data.target.shape[1], data.condition.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best = float("inf"); best_state = None; best_epoch = 0; history = []; started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in _batches(train_ids, args.batch_size, rng):
            target = torch.as_tensor(data.target[batch], device=device)
            condition = torch.as_tensor(data.condition[batch], device=device)
            loss = _trajectory_mean_loss(model(condition), target, args.root_weight)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([float(_trajectory_mean_loss(model(torch.as_tensor(data.condition[b], device=device)), torch.as_tensor(data.target[b], device=device), args.root_weight).cpu()) for b in [val_ids]]))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_trajectory_mse": val_loss}
        history.append(row)
        if val_loss < best:
            best = val_loss; best_epoch = epoch; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0: print(f"epoch {epoch:3d} train {np.mean(losses):.5f} val {val_loss:.5f} ({time.perf_counter()-started:.1f}s)", flush=True)
    if best_state is None: raise RuntimeError("no Stage-2 checkpoint was selected")
    model.load_state_dict(best_state); model.eval()
    metrics = {}
    with torch.no_grad():
        for name, ids in (("train", train_ids), ("validation", val_ids), ("test", test_ids)):
            pred = model(torch.as_tensor(data.condition[ids], device=device)).cpu().numpy()
            target = data.target[ids]
            pred_model = data.normalizer.inverse_target(pred.reshape(len(ids), *data.target_shape))
            pred_physical = _target_from_model(pred_model, data.joint_lower, data.joint_upper)
            target_physical = data.raw[args.model_target_field][ids]
            error = pred_physical - target_physical
            metrics[name] = {"rows": int(len(ids)),
                             "normalized_mae": float(np.abs(pred - target).mean()),
                             "joint_mae_rad": float(np.abs(error[..., :29]).mean()),
                             "root_position_mae_m": float(np.abs(error[..., 29:32]).mean()),
                             "root_rotation6d_mae": float(np.abs(error[..., 32:]).mean())}
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "stage2_router_conditioned_mean", "architecture": model.architecture(), "model_state": best_state, "normalizer": data.normalizer.state_dict(), "target_shape": data.target_shape, "primitive_count": data.primitive_count, "joint_lower": data.joint_lower, "joint_upper": data.joint_upper, "stage1_checkpoint": str(args.stage1_checkpoint), "windows": str(args.windows), "model_target_field": args.model_target_field, "best_epoch": best_epoch, "router_contract": "predicted M_e(t) probability vector replaces oracle primitive one-hot"}, args.out / "router_conditioned_mean.pt")
    summary = {"schema": "manifold-motion.stage2.router-conditioned-mean.v1", "windows": str(args.windows), "stage1_checkpoint": str(args.stage1_checkpoint), "metrics": metrics, "best_epoch": best_epoch, "history": history, "router_probability_entropy": float(np.mean(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-8)), axis=1))), "oracle_primitive_not_used": True, "limitations": ["deterministic trajectory mean, not yet stochastic Flow Matching", "root tracking still requires the physical candidate gate"]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n"); print(json.dumps(summary, indent=2, ensure_ascii=False)); return 0


def main():
    p = argparse.ArgumentParser(); p.add_argument("--windows", type=Path, required=True); p.add_argument("--stage1-checkpoint", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--model-target-field", choices=("target_ref", "target_exec"), default="target_exec"); p.add_argument("--epochs", type=int, default=35); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--hidden", type=int, default=256); p.add_argument("--learning-rate", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=1e-5); p.add_argument("--root-weight", type=float, default=2.0); p.add_argument("--seed", type=int, default=20260931); p.add_argument("--device", default="cpu"); p.add_argument("--threads", type=int, default=2); args = p.parse_args(); raise SystemExit(train(args))


if __name__ == "__main__": main()
