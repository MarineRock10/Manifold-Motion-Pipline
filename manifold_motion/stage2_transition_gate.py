"""MuJoCo gate for the synthesized Stage-2 primitive-transition supplement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .seed_replay import ReplayConfig, SeedReplayRunner
from .stage2_validate import validate_trajectory


def run(transitions: Path, out: Path, per_transition: int = 2) -> dict:
    with np.load(transitions, allow_pickle=False) as archive:
        targets = np.asarray(archive["target_ref"])
        names = np.asarray(archive["transition"]).astype(str)
        split = np.asarray(archive["split"])
    runner = SeedReplayRunner(C.FLAT_SCENE)
    rows = []
    for name in sorted(set(names.tolist())):
        indices = np.flatnonzero(names == name)
        selected = indices[np.linspace(0, len(indices) - 1,
                                      min(per_transition, len(indices)), dtype=int)]
        for index in selected:
            _, summary = validate_trajectory(
                targets[int(index)], source=transitions, source_hz=30.0,
                config=ReplayConfig(), stratum=f"transition_{name}", runner=runner,
            )
            rows.append({
                "index": int(index), "transition": name, "split": int(split[index]),
                "accepted": bool(summary["accepted"]),
                "failed_checks": summary["failed_checks"],
                "track_err_mean_rad": summary.get("track_err_mean_rad"),
                "base_z_min_m": summary.get("base_z_min_m"),
                "exec_path_m": summary.get("exec_path_m"),
                "obstacle_contact_ticks": summary.get("obstacle_contact_ticks"),
            })
    report = {
        "accepted": bool(rows) and any(row["accepted"] for row in rows),
        "transitions": str(transitions), "tested": len(rows),
        "accepted_count": int(sum(row["accepted"] for row in rows)),
        "acceptance_by_transition": {
            name: {
                "tested": sum(row["transition"] == name for row in rows),
                "accepted": sum(row["transition"] == name and row["accepted"] for row in rows),
            } for name in sorted(set(names.tolist()))
        },
        "rows": rows,
        "gate": "same SeedReplayRunner/ReplayConfig as offline primitive candidate screening",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-transition", type=int, default=2)
    args = parser.parse_args()
    report = run(args.transitions, args.out, args.per_transition)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
