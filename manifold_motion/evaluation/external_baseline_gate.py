"""Fail-closed registry for external-paper adapters.

The repository must not copy success numbers reported on Digit/H1 or in another simulator into
the G1 table.  This small command makes that boundary executable: an adapter is claimable only
when it declares the same robot, controller interface, scene/seed pairs, and hard safety gate.
Until an implementation is installed, the command emits an explicit ``not_ready`` report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REGISTRY: tuple[dict[str, Any], ...] = (
    {"id": "dynamic-subgoal-pursuit", "source": "https://arxiv.org/abs/2506.02206",
     "adapter": "Digit/MPC -> G1/SONIC executor", "status": "not_ready"},
    {"id": "p3o-safe-locomotion", "source": "https://arxiv.org/abs/2508.07611",
     "adapter": "raw spatio-temporal LiDAR -> simulated radar/voxel observation", "status": "not_ready"},
    {"id": "guidewalk", "source": "https://arxiv.org/abs/2606.10449",
     "adapter": "guidance policy -> G1 environment-manifold interface", "status": "not_ready"},
    {"id": "polygonal-semantic-mapping", "source": "https://arxiv.org/abs/2411.01919",
     "adapter": "mapping/planner component only", "status": "not_ready"},
    {"id": "whole-body-imitation-centroidal-mpc", "source": "https://arxiv.org/abs/2508.00362",
     "adapter": "motion reference -> frozen G1 safety executor", "status": "not_ready"},
    {"id": "optimization-embedded-trajectory", "source": "https://arxiv.org/pdf/2609.09158",
     "adapter": "embedded optimizer -> G1 reference/projection interface", "status": "not_ready"},
)


def build_report(out: Path) -> dict[str, Any]:
    report = {
        "schema": "manifold-motion.external-baseline-gate.v1",
        "claimable": False,
        "reason": "no external adapter has yet passed the identical G1/MuJoCo/SONIC gate",
        "required_protocol": {
            "robot": "Unitree G1 29-DoF",
            "controller": "same frozen SONIC checkpoint and action rate",
            "scenes": "same scenario/seed pairs as configs/cvpr_simulation_protocol.json",
            "safety": "same contact, self-manifold, tracking, keyframe and roll gates",
            "training_budget": "reported separately; no external data leakage",
        },
        "adapters": list(REGISTRY),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
