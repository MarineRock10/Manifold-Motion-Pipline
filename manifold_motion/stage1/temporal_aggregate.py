"""Aggregate the pre-registered multi-seed Stage-1 temporal confirmation runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def aggregate(summary_paths: list[Path], physical_paths: list[Path], out: Path) -> dict:
    summaries = [json.loads(p.read_text()) for p in summary_paths]
    physical = [json.loads(p.read_text()) for p in physical_paths]
    metrics = {}
    for key in ("joint_mae_rad", "velocity_mae_rad", "family_top1"):
        values = np.asarray([s["metrics"]["eval"][key] for s in summaries], dtype=float)
        metrics[key] = {"values": values.tolist(), "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    gate = {}
    for key in ("predicted_acceptance", "oracle_acceptance", "predicted_tracking_mae_rad"):
        values = np.asarray([p["aggregate"][key] for p in physical], dtype=float)
        gate[key] = {"values": values.tolist(), "mean": float(values.mean()),
                     "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    result = {
        "schema": "manifold-motion.stage1.temporal-multiseed.v1",
        "seeds": [int(s["seed"]) for s in summaries],
        "confirmation_rows_per_seed": [int(s["metrics"]["eval"]["rows"]) for s in summaries],
        "physical_rows_per_seed": [len(p["rows"]) for p in physical],
        "metrics": metrics, "physical_gate": gate,
        "protocol": {
            "actor_held_out": True, "per_family_physical_windows": 3,
            "physical_repeat": 2, "sonic_frozen": True,
            "selection": "validation joint MAE + 0.05*(1-family top1)",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    md = ["# Stage 1 时序原语：多 seed 收口", "", "确认集在训练前由 actor hash 固定；三种子不改变确认行。物理集按每个动作族均匀取 3 个窗口，共 42 个窗口/seed。", "", "| metric | seed values | mean ± SD |", "|---|---|---:|"]
    md.append(f"| joint MAE (rad) ↓ | {', '.join(f'{x:.4f}' for x in metrics['joint_mae_rad']['values'])} | {metrics['joint_mae_rad']['mean']:.4f} ± {metrics['joint_mae_rad']['sd']:.4f} |")
    md.append(f"| velocity MAE (rad) ↓ | {', '.join(f'{x:.4f}' for x in metrics['velocity_mae_rad']['values'])} | {metrics['velocity_mae_rad']['mean']:.4f} ± {metrics['velocity_mae_rad']['sd']:.4f} |")
    md.append(f"| family top-1 ↑ | {', '.join(f'{100*x:.1f}%' for x in metrics['family_top1']['values'])} | {100*metrics['family_top1']['mean']:.1f}% ± {100*metrics['family_top1']['sd']:.1f}% |")
    md += ["", "## MuJoCo / SONIC 物理确认", "", f"| predicted gate ↑ | {', '.join(f'{100*x:.1f}%' for x in gate['predicted_acceptance']['values'])} | {100*gate['predicted_acceptance']['mean']:.1f}% ± {100*gate['predicted_acceptance']['sd']:.1f}% |", f"| recorded-target gate ↑ | {', '.join(f'{100*x:.1f}%' for x in gate['oracle_acceptance']['values'])} | {100*gate['oracle_acceptance']['mean']:.1f}% ± {100*gate['oracle_acceptance']['sd']:.1f}% |", f"| tracking MAE (rad) ↓ | {', '.join(f'{x:.4f}' for x in gate['predicted_tracking_mae_rad']['values'])} | {gate['predicted_tracking_mae_rad']['mean']:.4f} ± {gate['predicted_tracking_mae_rad']['sd']:.4f} |", "", "该物理门控只检查冻结 SONIC 下的短时序列安全、身体包络半径和根漂移，不等价于完整导航成功率。"]
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--summaries", nargs="+", type=Path, required=True); p.add_argument("--physical", nargs="+", type=Path, required=True); p.add_argument("--out", type=Path, required=True); args = p.parse_args(); print(json.dumps(aggregate(args.summaries, args.physical, args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
