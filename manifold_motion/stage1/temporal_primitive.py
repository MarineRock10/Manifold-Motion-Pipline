"""Stage-1 temporal primitive model.

This module is intentionally separate from the Stage-2 flow model.  It does not receive the
recorded primitive label, the future self-manifold, or the target pose.  Its only geometric
input is the time-aligned environment manifold ``M_e(t)``; it predicts a complete 36-frame
joint sequence and an auxiliary action-family label.  The label is used only for evaluation.

The physical check replays the predicted sequence through the real SONIC/MuJoCo loop.  It is a
small CPU experiment, not a claim that the frozen ONNX controller has been retrained.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from manifold_motion.core import constants as C


DEFAULT = C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB].astype(np.float32)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as a:
        required = ("corridor", "target_exec", "primitive", "primitive_names", "split", "clip_index")
        missing = [key for key in required if key not in a.files]
        if missing:
            raise ValueError(f"catalog is missing {missing}")
        out = {key: np.asarray(a[key]) for key in a.files}
    if out["corridor"].ndim != 3 or out["corridor"].shape[-1] != 7:
        raise ValueError(f"corridor must be [N,T,7], got {out['corridor'].shape}")
    if out["target_exec"].shape[:2] != out["corridor"].shape[:2] or out["target_exec"].shape[-1] < 29:
        raise ValueError("target_exec must align with corridor and contain 29 joints")
    return out


def confirmation_split(catalog: Path, metadata: Path, out: Path, modulo: int = 5) -> dict[str, Any]:
    """Create a deterministic actor-held-out confirmation archive.

    Original test rows stay test rows.  The original training actors are partitioned into a
    new train/validation/confirmation split by a stable hash; this prevents selecting the
    confirmation set after seeing the temporal model's result.
    """
    data = _load(catalog)
    meta = json.loads(metadata.read_text())
    clips = meta.get("clips", [])
    if not clips:
        raise ValueError("metadata has no clips")
    actors = [str(clips[int(i)]["actor_uid"]) for i in data["clip_index"]]
    new_split = np.asarray(data["split"], dtype=np.uint8).copy()
    for i, actor in enumerate(actors):
        if int(data["split"][i]) != 0:
            continue
        bucket = int(hashlib.sha256(actor.encode()).hexdigest()[:8], 16) % modulo
        new_split[i] = 3 if bucket == 0 else (1 if bucket == 1 else 0)
    payload = {key: value for key, value in data.items()}
    payload["split"] = new_split
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    counts = {str(int(k)): int(v) for k, v in zip(*np.unique(new_split, return_counts=True))}
    actor_counts: dict[str, int] = {}
    for i, actor in enumerate(actors):
        actor_counts.setdefault(actor, int(new_split[i]))
    manifest = {
        "schema": "manifold-motion.stage1.temporal-confirmation.v1",
        "source_catalog": str(catalog), "metadata": str(metadata), "modulo": modulo,
        "split_counts": counts, "confirmation_actors": sorted(a for a, s in actor_counts.items() if s == 3),
        "actor_count": len(actor_counts), "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


class TemporalPrimitiveNet:
    """Factory wrapper kept importable without importing torch during data-only commands."""

    @staticmethod
    def build(torch, geom_dim: int, hidden: int, classes: int):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.geom = nn.Sequential(nn.Linear(geom_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
                self.temporal = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=0.05)
                self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 29))
                self.classifier = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, classes))

            def forward(self, x):
                z = self.geom(x)
                h, _ = self.temporal(z)
                return self.head(h), self.classifier(h.mean(dim=1))

        return Net()


def _metrics(pred: np.ndarray, target: np.ndarray, family_pred: np.ndarray,
             family: np.ndarray, names: np.ndarray, split: str) -> dict[str, Any]:
    err = pred - target
    vel_err = np.diff(pred, axis=1) - np.diff(target, axis=1)
    out: dict[str, Any] = {
        "split": split, "rows": int(len(target)),
        "joint_mae_rad": float(np.abs(err).mean()),
        "joint_rmse_rad": float(np.sqrt(np.mean(err ** 2))),
        "joint_p95_abs_rad": float(np.percentile(np.abs(err), 95)),
        "velocity_mae_rad": float(np.abs(vel_err).mean()),
        "family_top1": float(np.mean(family_pred == family)),
        "by_family": {},
    }
    for f in sorted(set(int(x) for x in family)):
        mask = family == f
        out["by_family"][str(names[f])] = {
            "rows": int(mask.sum()),
            "joint_mae_rad": float(np.abs(err[mask]).mean()),
            "family_top1": float(np.mean(family_pred[mask] == f)),
        }
    return out


def _fit_stats(corridor: np.ndarray, target: np.ndarray, train: np.ndarray):
    g = corridor[train].astype(np.float32)
    gm = g.reshape(-1, g.shape[-1]).mean(0)
    gs = np.maximum(g.reshape(-1, g.shape[-1]).std(0), 1e-5)
    y = target[train].astype(np.float32)
    ym = y.reshape(-1, y.shape[-1]).mean(0)
    ys = np.maximum(y.reshape(-1, y.shape[-1]).std(0), 1e-4)
    return gm, gs, ym, ys


def _predict(model, x: np.ndarray, device):
    import torch
    model.eval()
    with torch.no_grad():
        outputs, logits = model(torch.as_tensor(x, dtype=torch.float32, device=device))
    return outputs.cpu().numpy(), logits.cpu().numpy()


def train(args: argparse.Namespace) -> int:
    import torch
    torch.set_num_threads(max(1, int(args.threads)))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = _load(args.catalog)
    corridor = data["corridor"].astype(np.float32)
    target = data["target_exec"][..., :29].astype(np.float32) - DEFAULT[None, None, :]
    split = data["split"].astype(np.int64)
    train_ids = np.flatnonzero(split == 0)
    val_ids = np.flatnonzero(split == 1)
    eval_ids = np.flatnonzero(split == args.eval_split)
    if not len(train_ids) or not len(val_ids) or not len(eval_ids):
        raise ValueError(f"need non-empty train/validation/eval splits; got {np.unique(split, return_counts=True)}")
    gm, gs, ym, ys = _fit_stats(corridor, target, train_ids)
    x = (corridor - gm[None, None]) / gs[None, None]
    y = (target - ym[None, None]) / ys[None, None]
    device = torch.device(args.device)
    classes = int(data["primitive_names"].shape[0])
    class_counts = np.bincount(data["primitive"][train_ids], minlength=classes).astype(np.float32)
    # The catalogue is deliberately diverse and long-tailed.  Inverse-square-root weighting
    # avoids letting carry/walk families erase rare jump/interaction families while remaining
    # less noisy than full inverse-frequency weighting.
    class_weights = np.zeros(classes, dtype=np.float32)
    present = class_counts > 0
    class_weights[present] = np.sqrt(class_counts[present].sum() / class_counts[present])
    class_weights[present] /= class_weights[present].mean()
    torch_class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    model = TemporalPrimitiveNet.build(torch, 7, args.hidden, classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best = float("inf"); best_state = None; best_epoch = 0; history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); order = rng.permutation(train_ids); losses = []
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            tx = torch.as_tensor(x[ids], dtype=torch.float32, device=device)
            ty = torch.as_tensor(y[ids], dtype=torch.float32, device=device)
            tf = torch.as_tensor(data["primitive"][ids], dtype=torch.long, device=device)
            pred, logits = model(tx)
            pose = torch.nn.functional.smooth_l1_loss(pred, ty)
            vel = torch.nn.functional.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], ty[:, 1:] - ty[:, :-1])
            cls = torch.nn.functional.cross_entropy(logits, tf, weight=torch_class_weights)
            loss = pose + args.velocity_weight * vel + args.class_weight * cls
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        vp, vl = _predict(model, x[val_ids], device)
        vpose = vp * ys[None, None] + ym[None, None]
        vmetrics = _metrics(vpose, target[val_ids], vl.argmax(1), data["primitive"][val_ids], data["primitive_names"], "validation")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **vmetrics})
        selection_score = vmetrics["joint_mae_rad"] + args.selection_class_weight * (1.0 - vmetrics["family_top1"])
        history[-1]["selection_score"] = float(selection_score)
        if selection_score < best:
            best = selection_score; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"epoch {epoch:3d} loss {np.mean(losses):.4f} val_mae {vmetrics['joint_mae_rad']:.4f} val_acc {vmetrics['family_top1']:.3f} ({time.perf_counter()-started:.1f}s)", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    report_metrics = {}
    for label, ids in (("train", train_ids), ("validation", val_ids), ("eval", eval_ids)):
        yp, yl = _predict(model, x[ids], device)
        report_metrics[label] = _metrics(yp * ys[None, None] + ym[None, None], target[ids], yl.argmax(1), data["primitive"][ids], data["primitive_names"], label)
    mean_seq = target[train_ids].mean(axis=0)
    baseline = _metrics(np.repeat(mean_seq[None], len(eval_ids), axis=0), target[eval_ids],
                        np.full(len(eval_ids), int(np.bincount(data["primitive"][train_ids]).argmax())),
                        data["primitive"][eval_ids], data["primitive_names"], "global_mean")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = {"schema": "manifold-motion.stage1.temporal-primitive.v1", "model": best_state,
                  "geom_mean": gm, "geom_std": gs, "target_mean": ym, "target_std": ys,
                  "hidden": int(args.hidden), "classes": classes, "horizon": int(target.shape[1]),
                  "primitive_names": data["primitive_names"].tolist(), "best_epoch": best_epoch,
                  "catalog": str(args.catalog), "eval_split": int(args.eval_split)}
    torch.save(checkpoint, args.out / "temporal_primitive.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n")
    summary = {"schema": "manifold-motion.stage1.temporal-primitive-report.v1", "catalog": str(args.catalog),
               "eval_split": int(args.eval_split), "seed": args.seed, "epochs": args.epochs,
               "best_epoch": best_epoch, "metrics": report_metrics, "global_mean_baseline": baseline,
               "input_contract": "M_e(t) corridor only; primitive/self-manifold/target are not inputs",
               "limitations": ["target root translation/rotation are not predicted in this Stage-1 head",
                               "physical replay uses frozen SONIC; no ONNX weights are changed"]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _load_model(checkpoint: Path, device):
    import torch
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model = TemporalPrimitiveNet.build(torch, 7, int(ck["hidden"]), int(ck["classes"])).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck


def _limits(key):
    model = key.env.model
    ids = model.actuator_trnid[key.env.body_act, 0]
    limits = model.jnt_range[ids][C.MUJOCO_TO_ISAACLAB]
    return limits[:, 0].astype(np.float32) - DEFAULT + 0.005, limits[:, 1].astype(np.float32) - DEFAULT - 0.005


def _rollout(sequence: np.ndarray, semi: np.ndarray, repeat: int = 2, capture: bool = False):
    import mujoco
    from manifold_motion.simulation.keyframe_env import KeyframeEnv
    key = KeyframeEnv(); key.reset(); low, high = _limits(key)
    sequence = np.clip(np.asarray(sequence, dtype=np.float32), low, high)
    for _ in range(10): key.step()
    origin = key.state()["base_pos"].copy(); max_radius = 0.; max_drift = 0.; fallen = False
    errors = []; trace = {"q": [], "pos": [], "quat": [], "t": [], "semi": np.asarray(semi)}
    for frame in range(len(sequence)):
        for _ in range(max(1, repeat)):
            key.set_joints(sequence[frame]); key.step(); state = key.state()
            rot = C.quat_to_matrix(state["base_quat"]); fallen |= bool(state["base_pos"][2] < .30 or rot[2, 2] < .45)
            q = state["q_hw"][C.MUJOCO_TO_ISAACLAB] - DEFAULT
            errors.append(float(np.abs(q - sequence[frame]).mean()))
            points = __import__("manifold_motion.core.body_envelope", fromlist=["body_points"]).body_points(key.env.model, key.env.data) - state["base_pos"]
            yaw = math.atan2(rot[1, 0], rot[0, 0]); heading = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            max_radius = max(max_radius, float(np.linalg.norm((points @ heading) / np.maximum(semi, 1e-4), axis=1).max()))
            max_drift = max(max_drift, float(np.linalg.norm(state["base_pos"][:2] - origin[:2])))
            if capture:
                trace["q"].append(state["q_hw"][C.MUJOCO_TO_ISAACLAB].copy()); trace["pos"].append(state["base_pos"].copy()); trace["quat"].append(state["base_quat"].copy()); trace["t"].append(key.env.time)
    result = {"tracking_mae_rad": float(np.mean(errors)), "radius": max_radius, "drift_m": max_drift,
              "fallen": bool(fallen), "accepted": bool((not fallen) and max_radius <= 1.0 and max_drift <= .35)}
    return result, {k: np.asarray(v) for k, v in trace.items()} if capture else None


def physical(args: argparse.Namespace) -> int:
    import torch
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device(args.device); model, ck = _load_model(args.checkpoint, device); data = _load(args.catalog)
    corridor = data["corridor"].astype(np.float32); target = data["target_exec"][..., :29].astype(np.float32) - DEFAULT[None, None]
    x = (corridor - ck["geom_mean"][None, None]) / ck["geom_std"][None, None]
    ids = np.flatnonzero(data["split"] == args.eval_split)
    selected = []
    for f in sorted(set(int(v) for v in data["primitive"][ids])):
        group = ids[data["primitive"][ids] == f]
        chosen = np.linspace(0, len(group) - 1, min(args.per_family, len(group))).astype(int)
        selected.extend(int(value) for value in group[chosen])
    selected = selected[:args.max_rows]
    pred, logits = _predict(model, x[selected], device)
    pred = pred * ck["target_std"][None, None] + ck["target_mean"][None, None]
    rows = []; media = args.out / "media"; media.mkdir(parents=True, exist_ok=True)
    render_traces = None
    if args.gif_count:
        from manifold_motion.stage1.catalog_report import render_traces as render_traces_impl
        render_traces = render_traces_impl
    for j, row in enumerate(selected):
        capture = j < args.gif_count
        p, p_trace = _rollout(pred[j], data["corridor"][row, len(data["corridor"][row]) // 2, 3:6], args.repeat, capture=capture)
        o, o_trace = _rollout(target[row], data["corridor"][row, len(data["corridor"][row]) // 2, 3:6], args.repeat, capture=capture)
        family = str(data["primitive_names"][data["primitive"][row]])
        p.update({"row": int(row), "family": family, "predicted_family": str(data["primitive_names"][int(logits[j].argmax())]), "oracle": o})
        rows.append(p)
        if j < args.gif_count:
            path = media / f"temporal_{row:06d}_{family}.gif"
            assert render_traces is not None
            render_traces([o_trace, p_trace], ["recorded target → SONIC", "M_e(t) temporal model → SONIC"],
                          [{"pose_success": o["accepted"], "radius": o["radius"], "target_mae": o["tracking_mae_rad"], "drift_m": o["drift_m"], "fallen": o["fallen"]},
                           {"pose_success": p["accepted"], "radius": p["radius"], "target_mae": p["tracking_mae_rad"], "drift_m": p["drift_m"], "fallen": p["fallen"]}],
                          path, f"Stage 1 temporal primitive / {family} / row {row}")
    summary = {"schema": "manifold-motion.stage1.temporal-physical.v1", "rows": rows,
               "aggregate": {"predicted_acceptance": float(np.mean([r["accepted"] for r in rows])) if rows else None,
                              "oracle_acceptance": float(np.mean([r["oracle"]["accepted"] for r in rows])) if rows else None,
                              "predicted_tracking_mae_rad": float(np.mean([r["tracking_mae_rad"] for r in rows])) if rows else None},
               "frozen_sonic": True, "repeat": args.repeat}
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "physical.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False)); return 0


def report(args: argparse.Namespace) -> int:
    summary = json.loads((args.out / "summary.json").read_text()); physical_path = args.out / "physical.json"
    lines = ["# Stage 1：时序原语实验", "", "本实验冻结 SONIC，只训练上层时序原语头。网络只读取 `M_e(t)` 的 36 帧、7 维椭球序列，不读取 primitive 标签、自身未来流形或目标动作。", "", "## 结果", "", "| split | rows | joint MAE (rad) | velocity MAE (rad) | family top-1 |", "|---|---:|---:|---:|---:|"]
    for k, m in summary["metrics"].items(): lines.append(f"| {k} | {m['rows']} | {m['joint_mae_rad']:.4f} | {m['velocity_mae_rad']:.4f} | {m['family_top1']:.1%} |" )
    b = summary["global_mean_baseline"]; lines += ["", f"全局时序均值基线 eval MAE = **{b['joint_mae_rad']:.4f} rad**；模型 eval MAE = **{summary['metrics']['eval']['joint_mae_rad']:.4f} rad**。", "", "## 物理回放", ""]
    if physical_path.is_file():
        p = json.loads(physical_path.read_text()); lines += [f"预测序列在 {len(p['rows'])} 个按动作族首窗选择的 MuJoCo 回放上通过率 **{p['aggregate']['predicted_acceptance']:.1%}**；记录目标的 oracle SONIC 通过率 **{p['aggregate']['oracle_acceptance']:.1%}**。这是短时 executor gate，不是完整导航成功率。", "", "| family | temporal model GIF |", "|---|---|"]
        for r in p["rows"][:args.gif_count]:
            name = f"temporal_{r['row']:06d}_{r['family']}.gif"; lines.append(f"| {r['family']} | [![](media/{name})](media/{name}) |" )
    lines += ["", "## 边界", "", "模型预测 29 个关节序列；根位置/根旋转仍由后续 Stage 2 动态模型负责。确认集由 actor hash 在训练完成前固定划分，不能与已查看的 development test 混用。SONIC ONNX 权重没有被修改；本实验的适配对象是上层原语模型。"]
    (args.out / "README.md").write_text("\n".join(lines) + "\n"); print(args.out / "README.md"); return 0


def make_confirmation_command(args: argparse.Namespace) -> int:
    manifest = confirmation_split(args.catalog, args.metadata, args.out)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("make-confirmation"); c.add_argument("--catalog", type=Path, required=True); c.add_argument("--metadata", type=Path, required=True); c.add_argument("--out", type=Path, required=True); c.set_defaults(fn=make_confirmation_command)
    t = sub.add_parser("train"); t.add_argument("--catalog", type=Path, required=True); t.add_argument("--out", type=Path, required=True); t.add_argument("--epochs", type=int, default=45); t.add_argument("--batch-size", type=int, default=64); t.add_argument("--hidden", type=int, default=96); t.add_argument("--learning-rate", type=float, default=3e-4); t.add_argument("--velocity-weight", type=float, default=.15); t.add_argument("--class-weight", type=float, default=.6); t.add_argument("--selection-class-weight", type=float, default=.05); t.add_argument("--seed", type=int, default=20260928); t.add_argument("--eval-split", type=int, default=2); t.add_argument("--device", default="cpu"); t.add_argument("--threads", type=int, default=2); t.set_defaults(fn=train)
    q = sub.add_parser("physical"); q.add_argument("--catalog", type=Path, required=True); q.add_argument("--checkpoint", type=Path, required=True); q.add_argument("--out", type=Path, required=True); q.add_argument("--max-rows", type=int, default=90); q.add_argument("--per-family", type=int, default=3); q.add_argument("--gif-count", type=int, default=6); q.add_argument("--repeat", type=int, default=2); q.add_argument("--eval-split", type=int, default=2); q.add_argument("--device", default="cpu"); q.add_argument("--threads", type=int, default=2); q.set_defaults(fn=physical)
    r = sub.add_parser("report"); r.add_argument("--out", type=Path, required=True); r.add_argument("--gif-count", type=int, default=6); r.set_defaults(fn=report)
    args = p.parse_args(); raise SystemExit(args.fn(args))


if __name__ == "__main__": main()
