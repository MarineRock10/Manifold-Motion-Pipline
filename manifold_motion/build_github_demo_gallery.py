"""Build small, deterministic GIF previews for the public GitHub demo gallery.

The acceptance GIFs in ``reports/`` are intentionally high resolution and are
not checked into the repository.  This command samples those already-accepted
clips, preserves their timing, and writes a compact preview plus a manifest.
It never runs a new rollout and therefore cannot turn an unverified clip into
paper evidence by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image


SPECS = (
    {
        "id": "online_closed_loop",
        "title": "Online 3-D SLAM → M_e → primitive → SONIC",
        "source": "reports/manifold_motion/deploy_perception_demo/deploy_sonic_mujoco_comprehensive.gif",
        "purpose": "continuous radar updates, D* Lite route repair and shadow-gated semantic switching",
        "max_width": 520,
        "stride": 4,
    },
    {
        "id": "moving_obstacle_replanning",
        "title": "Appearing / crossing / disappearing obstacle",
        "source": "reports/manifold_motion/incremental_dynamic_benchmark/moving_obstacle_replanning.gif",
        "purpose": "incremental ESDF + D* Lite route changes under dynamic map updates",
        "max_width": 560,
        "stride": 1,
    },
    {
        "id": "compound_long_horizon",
        "title": "Long horizon: walk → side → turn → walk",
        "source": "reports/manifold_motion/stage2_long_sequence_v1/combo/manifold_adaptive.gif",
        "purpose": "one continuous MuJoCo state with multiple manifold-conditioned primitives",
        "max_width": 480,
        "stride": 8,
    },
    {
        "id": "low_clearance_cycle",
        "title": "Repeated low-clearance intervals",
        "source": "reports/manifold_motion/stage2_long_sequence_v1/low_cycle/manifold_adaptive.gif",
        "purpose": "crouch transitions are triggered by measured vertical aperture",
        "max_width": 480,
        "stride": 10,
    },
    {
        "id": "wide_vs_low_counterfactual",
        "title": "Counterfactual: same route, different aperture",
        "source": "reports/manifold_motion/stage2_manifold_counterfactual_v1/counterfactual_wide_vs_low_compact.gif",
        "purpose": "paired wide/low manifold evidence without a segment-label shortcut",
        "max_width": 480,
        "stride": 6,
    },
    {
        "id": "side_passage",
        "title": "Offset obstacle: side-on passage",
        "source": "reports/manifold_motion/stage2_diverse_demo_v2/block_left/manifold_adaptive.gif",
        "purpose": "lateral gait selection from the local safe corridor",
        "max_width": 480,
        "stride": 6,
    },
    {
        "id": "manifold_behavior_matrix",
        "title": "Primitive matrix: wide / low / narrow / side",
        "source": "reports/manifold_motion/stage2_manifold_behavior_v1/manifold_behavior.gif",
        "purpose": "paired environment-manifold routing and same-reference counterfactual evidence",
        "max_width": 620,
        "stride": 2,
    },
    {
        "id": "flow_route_avoidance",
        "title": "Flow candidates on a blocked route",
        "source": "reports/manifold_motion/stage2_flow_route_avoidance_v1/flow_route_avoidance.gif",
        "purpose": "multiple conditional motion candidates screened against the MuJoCo route",
        "max_width": 520,
        "stride": 5,
    },
    {
        "id": "crouch_transition",
        "title": "Transition → crouch without state reset",
        "source": "reports/manifold_motion/stage2_sequence_transition_crouch_v2/sequence.gif",
        "purpose": "longer temporal transition with bridge frames and continuous MuJoCo state",
        "max_width": 520,
        "stride": 5,
    },
    {
        "id": "jump_and_land",
        "title": "Generated jump and verified landing",
        "source": "reports/manifold_motion/stage2_jump_high_actor_test_scan_v1/index_0/physical_validation/jump.gif",
        "purpose": "distinct high-amplitude primitive with airborne and landing checks",
        "max_width": 520,
        "stride": 2,
    },
    {
        "id": "long_side_gait",
        "title": "Long narrow passage: side gait",
        "source": "reports/manifold_motion/stage2_long_sequence_gallery_v1/narrow_long/manifold_adaptive.gif",
        "purpose": "extended side-on execution through a narrow environment manifold",
        "max_width": 480,
        "stride": 8,
    },
    {
        "id": "long_low_gait",
        "title": "Long low corridor: crouch gait",
        "source": "reports/manifold_motion/stage2_long_sequence_gallery_v1/low_long/manifold_adaptive.gif",
        "purpose": "long-horizon low-clearance execution with measured self-manifold gating",
        "max_width": 480,
        "stride": 9,
    },
    {
        "id": "extended_chicane",
        "title": "Long chicane: nominal walk with alternating turns",
        "source": "reports/manifold_motion/stage2_extended_gallery_v2/chicane_turns/manifold_adaptive.gif",
        "purpose": "17-keyframe sequence where single-sided obstacles keep nominal walking and route curvature triggers nine turns",
        "max_width": 480,
        "stride": 10,
    },
    {
        "id": "extended_low_side_turn",
        "title": "Compound task: crouch → side gait → turn → recovery",
        "source": "reports/manifold_motion/stage2_extended_gallery_v2/low_side_turn/manifold_adaptive.gif",
        "purpose": "18-keyframe no-reset task with low-transition helpers, crouch, lateral gait and curved recovery",
        "max_width": 480,
        "stride": 10,
    },
    {
        "id": "extended_gate_cycle",
        "title": "Repeated gates: crouch ↔ walk → side → crouch",
        "source": "reports/manifold_motion/stage2_extended_gallery_v2/gate_cycle/manifold_adaptive.gif",
        "purpose": "three low gates separated by recovery and a bilateral lateral-aperture gate",
        "max_width": 480,
        "stride": 10,
    },
    {
        "id": "extended_slalom",
        "title": "Long slalom: latched turns and side gait",
        "source": "reports/manifold_motion/stage2_extended_gallery_v2/slalom_turns/manifold_adaptive.gif",
        "purpose": "18-keyframe alternating route with nine curvature turns, three bilateral side intervals and no turn/side chatter",
        "max_width": 480,
        "stride": 10,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preview(source: Path, destination: Path, max_width: int, stride: int) -> dict:
    image = Image.open(source)
    frame_count = getattr(image, "n_frames", 1)
    stride = max(stride, math.ceil(frame_count / 120))
    frames: list[Image.Image] = []
    durations: list[int] = []
    for begin in range(0, frame_count, stride):
        image.seek(begin)
        frame = image.convert("RGB")
        if frame.width > max_width:
            height = round(frame.height * max_width / frame.width)
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            frame = frame.resize((max_width, height), resampling)
        # A local adaptive palette keeps the dark MuJoCo panels readable while
        # keeping the checked-in previews small enough for a normal clone.
        quantize = getattr(Image, "Quantize", None)
        median_cut = quantize.MEDIANCUT if quantize is not None else Image.MEDIANCUT
        frame = frame.quantize(colors=64, method=median_cut)
        frames.append(frame)
        duration = 0
        for offset in range(stride):
            index = min(begin + offset, frame_count - 1)
            image.seek(index)
            duration += int(image.info.get("duration", 50))
        durations.append(max(20, duration))
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return {
        "source": str(source),
        "frames_source": frame_count,
        "frames_preview": len(frames),
        "stride": stride,
        "size": [frames[0].width, frames[0].height],
        "sha256": _sha256(destination),
        "bytes": destination.stat().st_size,
    }


def build(repo: Path, output: Path) -> dict:
    records = []
    for spec in SPECS:
        source = repo / spec["source"]
        if not source.is_file():
            raise FileNotFoundError(
                f"accepted source GIF is missing: {source}; run its documented demo first"
            )
        destination = output / f"{spec['id']}.gif"
        record = {**spec, "preview": str(destination.relative_to(repo))}
        record.update(_preview(source, destination, spec["max_width"], spec["stride"]))
        records.append(record)
    manifest = {
        "schema": "manifold-motion.github-demo-gallery.v1",
        "generated_from": "accepted reports only; no rollout is performed",
        "items": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "demo_gallery" / "media",
    )
    args = parser.parse_args()
    print(json.dumps(build(args.repo.resolve(), args.out.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
