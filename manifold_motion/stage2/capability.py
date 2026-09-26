"""Explicit capability boundary for primitives under the frozen SONIC controller.

The dynamic generator must not route an unsupported morphology just because its token exists in
the SEED taxonomy.  This manifest is intentionally conservative and records the evidence class
for each primitive so a future fine-tuned SONIC checkpoint can replace entries without changing
the Stage-2 data contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CAPABILITIES: dict[str, dict[str, Any]] = {
    "all_fours": {
        "primitive_id": 0, "status": "unsupported",
        "reason": "not a stable locomotion primitive in the frozen controller",
    },
    "crawl": {
        "primitive_id": 1, "status": "unsupported",
        "reason": "87/87 representative SEED crawl replays failed fall/roll gates; requires SONIC fine-tuning",
    },
    "crouch": {
        "primitive_id": 2, "status": "supported",
        "evidence": "independent actor split and low-ceiling continuous execution",
    },
    "low_transition": {
        "primitive_id": 3, "status": "supported",
        "evidence": "34/46 actor-isolated SONIC replays and continuous transition gate",
    },
    "walk_lateral_reverse": {
        "primitive_id": 4, "status": "supported",
        "evidence": "strict side-on 1.05 m corridor; 0 side-wall contacts",
    },
    "walk_nominal": {
        "primitive_id": 5, "status": "supported",
        "evidence": "independent normal-walk acceptance and wide-scene execution",
    },
    "walk_turn": {
        "primitive_id": 6, "status": "supported",
        "evidence": "curvature-triggered center-obstacle route execution",
    },
    "jump": {
        "primitive_id": 7, "status": "partial",
        "reason": (
            "20/20 isolated takeoff/landing checks and one phase-matched "
            "jump->low-transition->crouch sequence pass; corridor-conditioned jump routing is not validated"
        ),
    },
}


def supported_ids(include_partial: bool = False) -> tuple[int, ...]:
    statuses = {"supported", "partial"} if include_partial else {"supported"}
    return tuple(sorted(int(row["primitive_id"]) for row in CAPABILITIES.values()
                       if row["status"] in statuses))


def build(args: argparse.Namespace) -> int:
    result = {
        "controller": "frozen GEAR-SONIC ONNX",
        "capabilities": CAPABILITIES,
        "strict_supported_primitive_ids": list(supported_ids(False)),
        "partial_primitive_ids": list(supported_ids(True)),
        "routing_policy": "unsupported primitives are rejected before Flow sampling; partial jump is opt-in only",
        "next_controller_upgrade": {
            "crawl": "fine-tune low-contact SONIC in Isaac Lab and re-export ONNX",
            "jump": "collect corridor-conditioned takeoff/landing windows and replay them through the new controller",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the frozen-SONIC Stage-2 capability manifest")
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/stage2_capability_manifest.json"))
    return build(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
