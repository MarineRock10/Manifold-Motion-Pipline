"""Collect accepted long-horizon Stage-2 runs into one reproducible gallery."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SCENARIOS = (
    ("wide_long", "wide corridor -> nominal walk",
     "reports/manifold_motion/stage2_manifold_counterfactual_v1/wide"),
    ("low_long", "low ceiling -> crouch",
     "reports/manifold_motion/stage2_manifold_counterfactual_v1/low"),
    ("narrow_long", "narrow passage -> side gait",
     "reports/manifold_motion/stage2_manifold_counterfactual_v1/narrow"),
    ("center_online", "center block -> online radar SLAM + turn/side",
     "reports/manifold_motion/deploy_online_perception_v1"),
    ("long_low_cycle", "low -> walk -> low without reset",
     "reports/manifold_motion/stage2_long_sequence_v1/low_cycle"),
    ("long_combo", "low -> side gait without reset",
     "reports/manifold_motion/stage2_long_sequence_v1/combo"),
)


def build(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, description, source_text in SCENARIOS:
        source = Path(source_text)
        report_path = source / "report.json"
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text())
        execution = report.get("execution", {})
        if not report.get("accepted", False):
            raise RuntimeError(f"source run is not accepted: {report_path}")
        destination = root / name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, destination / "report.json")
        gif = source / "manifold_adaptive.gif"
        if gif.is_file():
            shutil.copy2(gif, destination / "manifold_adaptive.gif")
        online_gif = source / "deploy_mujoco_slam_online_synced.gif"
        online_archive = source / "online_perception.npz"
        if online_gif.is_file():
            shutil.copy2(online_gif, destination / online_gif.name)
        if online_archive.is_file():
            shutil.copy2(online_archive, destination / online_archive.name)
        rows.append({
            "name": name,
            "description": description,
            "source": source_text,
            "accepted": bool(report["accepted"]),
            "failed_checks": report.get("failed_checks", []),
            "routed_primitives": [row.get("primitive") for row in report.get("decisions", [])],
            "keyframes_reached": execution.get("keyframes_reached"),
            "keyframes_total": execution.get("keyframes_total_excluding_start"),
            "physics_ticks": execution.get("physics_ticks"),
            "duration_s": execution.get("duration_s"),
            "primitive_switch_count": execution.get("primitive_switch_count"),
            "obstacle_contact_ticks": execution.get("obstacle_contact_ticks"),
            "route_deviation_p95_m": execution.get("route_deviation_p95_m"),
            "exact_clearance_min_m": report.get("robot_self_manifold_safety", {}).get(
                "exact_surface_obstacle_clearance_min_m"),
            "online_perception": report.get("online_perception", {"enabled": False}),
            "gif": str(destination / "manifold_adaptive.gif"),
            "online_gif": (str(destination / online_gif.name) if online_gif.is_file() else None),
        })
    summary = {
        "experiment": "accepted long-horizon manifold-conditioned action gallery",
        "synthetic_geometry_note": "MuJoCo fixture geometry; replace the radar adapter with real point-cloud SLAM for hardware",
        "accepted": bool(rows) and all(row["accepted"] and not row["failed_checks"] for row in rows),
        "scenario_count": len(rows),
        "scenarios": rows,
    }
    (root / "comparison_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    markdown = ["# Stage-2 long-sequence gallery", "", "This gallery contains six accepted, no-reset runs. `center_online` additionally uses live radar -> sliding voxel SLAM -> local 3-D A* during execution.", "", "| scenario | action/condition | keyframes | ticks | switches | contacts | clearance (m) |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        clearance = ("n/a" if row["exact_clearance_min_m"] is None
                     else f"{float(row['exact_clearance_min_m']):.3f}")
        markdown.append(
            f"| `{row['name']}` | {row['description']} | "
            f"{row['keyframes_reached']}/{row['keyframes_total']} | {row['physics_ticks']} | "
            f"{row['primitive_switch_count']} | {row['obstacle_contact_ticks']} | "
            f"{clearance} |"
        )
    markdown += ["", "Rebuild the curated gallery from the accepted source runs:", "", "```bash", "./run_stage2_long_gallery.sh", "```", ""]
    (root / "GALLERY.md").write_text("\n".join(markdown))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="collect accepted long-horizon Stage-2 runs")
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/stage2_long_sequence_gallery_v1"))
    args = parser.parse_args()
    summary = build(args.out)
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
