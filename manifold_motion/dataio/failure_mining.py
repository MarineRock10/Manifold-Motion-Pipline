"""Cluster physical replay failures and make a targeted SEED supplementation list.

Failure mining is intentionally conservative and deterministic.  It groups hard checks emitted by
the frozen SONIC/MuJoCo replay into actionable failure modes, then maps each mode to a small set
of SEED attributes to collect.  The output is a data-acquisition plan; it does not silently start
SONIC fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from manifold_motion.dataio.seed_capability_catalog import FAMILIES, _load_candidates, _rank


CLUSTER_RULES = (
    ("jump_phase", ("jump_no_sustained_takeoff", "jump_insufficient_lift", "jump_no_verified_landing")),
    ("nonfoot_contact", ("nonfoot_floor_contact",)),
    ("tracking_instability", ("fall_or_extreme_roll", "roll_limit", "tracking_error", "leg_tracking_error")),
    ("tracking", ("tracking_error", "leg_tracking_error")),
)

TARGETS = {
    "jump_phase": {
        "priority": "P0",
        "families": "side_hop,jump_forward,broad_jump,high_jump,box_jump",
        "attributes": "phase-aligned takeoff;landing;pelvis-height;foot-contact;root-velocity",
        "action": "collect short takeoff and landing clips with explicit contact labels",
    },
    "nonfoot_contact": {
        "priority": "P0",
        "families": "crouch_walk,crouch_transition,all_fours,step_down_box",
        "attributes": "low-clearance;hands-off-floor;foot-contact;body-envelope",
        "action": "collect low-profile transitions with verified hand and torso clearance",
    },
    "tracking_instability": {
        "priority": "P0",
        "families": "crouch_transition,ladder,kneel,all_fours,roll_recovery",
        "attributes": "slow-transition;root-orientation;leg-tracking;handoff-keyframes",
        "action": "collect slower segmented clips with 4-8 keyframes and neutral handoffs",
    },
    "tracking": {
        "priority": "P1",
        "families": "walk_lateral,walk_curve,turn_in_place,side_hop",
        "attributes": "root-yaw;planar-velocity;foot-contact;turn-radius",
        "action": "collect root-position and heading-aligned clips, not isolated pose snapshots",
    },
    "other": {
        "priority": "P2",
        "families": "all_supported",
        "attributes": "root-pose;contacts;failure-phase",
        "action": "inspect manually before adding data",
    },
}

DEFAULT_SUPPLEMENT_FAMILIES = (
    "crouch_walk", "crouch_transition", "side_hop", "jump_forward", "broad_jump",
    "high_jump", "box_jump", "all_fours", "ladder", "roll_recovery",
)


def _cluster(failed_checks: list[str]) -> str:
    checks = set(failed_checks)
    for name, triggers in CLUSTER_RULES:
        if checks.intersection(triggers):
            return name
    return "other"


def _rows(replay_roots: list[Path], manifest: Path | None) -> list[dict[str, Any]]:
    manifest_rows: dict[str, dict[str, str]] = {}
    if manifest and manifest.is_file():
        with manifest.open(newline="") as handle:
            manifest_rows = {row["motion_id"]: row for row in csv.DictReader(handle)}
    unique: dict[str, dict[str, Any]] = {}
    for root in replay_roots:
        for path in sorted((root / "summaries").glob("*.json")):
            summary = json.loads(path.read_text())
            if summary.get("accepted", False):
                continue
            motion_id = str(summary.get("motion_id", path.stem))
            checks = sorted(set(str(value) for value in summary.get("failed_checks", [])))
            row = {
                "motion_id": motion_id,
                "source_replay_root": str(root),
                "stratum": summary.get("stratum", manifest_rows.get(motion_id, {}).get("pilot_stratum", "")),
                "skill_family": manifest_rows.get(motion_id, {}).get("skill_family", ""),
                "family_id": manifest_rows.get(motion_id, {}).get("family_id", ""),
                "failed_checks": checks,
                "cluster": _cluster(checks),
                "track_err_mean_rad": summary.get("track_err_mean_rad"),
                "track_err_legs_rad": summary.get("track_err_legs_rad"),
                "fell": bool(summary.get("fell", False)),
            }
            # If the same clip occurs in pilot and full replay, retain one row and all provenance.
            if motion_id in unique:
                unique[motion_id]["source_replay_root"] += ";" + str(root)
            else:
                unique[motion_id] = row
    return list(unique.values())


def cluster(replay_roots: list[Path], out: Path, manifest: Path | None = None) -> dict[str, Any]:
    rows = _rows(replay_roots, manifest)
    counts = Counter(row["cluster"] for row in rows)
    recommendations = []
    for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        target = TARGETS[name]
        recommendations.append({"cluster": name, "failures": int(count), **target})
    supplement = []
    for recommendation in recommendations:
        supplement.append({"priority": recommendation["priority"], "cluster": recommendation["cluster"],
                           "suggested_families": recommendation["families"],
                           "required_attributes": recommendation["attributes"],
                           "collection_action": recommendation["action"],
                           "source_failure_count": recommendation["failures"]})
    out.mkdir(parents=True, exist_ok=True)
    (out / "failure_clusters.json").write_text(json.dumps({
        "schema": "manifold-motion.failure-clusters.v1",
        "replay_roots": [str(value) for value in replay_roots],
        "failure_count": len(rows), "cluster_counts": dict(counts),
        "rows": rows, "recommendations": recommendations,
        "fine_tuning_policy": "supplement targeted SEED clips first; do not retrain frozen SONIC from all data",
    }, indent=2) + "\n")
    with (out / "targeted_seed_supplement.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(supplement[0]) if supplement else
                                ["priority", "cluster", "suggested_families", "required_attributes",
                                 "collection_action", "source_failure_count"])
        writer.writeheader(); writer.writerows(supplement)
    report = {"failure_count": len(rows), "cluster_counts": dict(counts),
              "clusters": recommendations, "targeted_manifest": "targeted_seed_supplement.csv"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def targeted_manifest(metadata: Path, exclude_manifests: list[Path], output: Path,
                      families: list[str], per_family: int, seed: int,
                      max_frames: int = 1800) -> dict[str, Any]:
    """Select unseen actor-diverse clips for the failure clusters, without extracting the archive."""
    known = {family.name: family for family in FAMILIES}
    unknown = sorted(set(families) - set(known))
    if unknown:
        raise ValueError(f"unknown supplement families: {unknown}")
    excluded_paths: set[str] = set()
    for manifest in exclude_manifests:
        with manifest.open(newline="", encoding="utf-8") as handle:
            excluded_paths.update(row["move_g1_path"] for row in csv.DictReader(handle))
    candidates = _load_candidates(metadata, max_frames=max_frames, include_mirrored=False)
    fields = ["motion_id", "family_id", "skill_family", "pilot_stratum", "move_g1_path",
              "move_name", "actor_uid", "selection_role", "environment_trigger", "body_mask", "risk"]
    rows: list[dict[str, str]] = []
    availability: dict[str, int] = {}
    for name in families:
        family = known[name]
        pool = [row for row in candidates[name] if row["move_g1_path"] not in excluded_paths]
        availability[name] = len(pool)
        ordered = sorted(pool, key=lambda row: (
            {"test": 0, "validation": 1, "train": 2}.get(row["actor_split"], 3),
            _rank(seed, f"supplement:{name}", row["move_g1_path"]),
        ))
        chosen = []
        actors: set[str] = set()
        # Prefer a new actor for every row, then fill from remaining unique motion paths.
        for unique_actor_pass in (True, False):
            for row in ordered:
                if row in chosen or (unique_actor_pass and row.get("actor_uid", "") in actors):
                    continue
                chosen.append(row); actors.add(row.get("actor_uid", ""))
                if len(chosen) >= per_family:
                    break
            if len(chosen) >= per_family:
                break
        split_ordinals: defaultdict[str, int] = defaultdict(int)
        for row in chosen:
            split = row["actor_split"]; ordinal = split_ordinals[split]; split_ordinals[split] += 1
            rows.append({
                "motion_id": f"supp_{family.family_id:02d}_{name}_{split}_{ordinal:02d}",
                "family_id": str(family.family_id), "skill_family": name,
                "pilot_stratum": family.gate_profile, "move_g1_path": row["move_g1_path"],
                "move_name": row["move_name"], "actor_uid": row.get("actor_uid", ""),
                "selection_role": f"failure_directed_{split}",
                "environment_trigger": family.environment_trigger, "body_mask": family.body_mask,
                "risk": family.risk,
            })
            excluded_paths.add(row["move_g1_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    report = {"schema": "manifold-motion.targeted-seed-supplement.v1", "metadata": str(metadata),
              "excluded_manifests": [str(value) for value in exclude_manifests],
              "families": families, "per_family": per_family, "rows": len(rows),
              "available_unseen": availability, "output": str(output),
              "admission_rule": "only frozen SONIC/MuJoCo accepted rows may enter adapter training"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("cluster")
    command.add_argument("--replay-root", type=Path, nargs="+", required=True)
    command.add_argument("--manifest", type=Path, default=None)
    command.add_argument("--out", type=Path, default=Path("reports/manifold_motion/failure_clusters_v1"))
    supplement = sub.add_parser("supplement")
    supplement.add_argument("--metadata", type=Path, required=True)
    supplement.add_argument("--exclude-manifest", type=Path, nargs="+", required=True)
    supplement.add_argument("--out", type=Path, required=True)
    supplement.add_argument("--families", nargs="+", default=list(DEFAULT_SUPPLEMENT_FAMILIES))
    supplement.add_argument("--per-family", type=int, default=3)
    supplement.add_argument("--seed", type=int, default=20260927)
    supplement.add_argument("--max-frames", type=int, default=1800)
    args = parser.parse_args()
    if args.command == "cluster":
        report = cluster(args.replay_root, args.out, args.manifest)
    else:
        report = targeted_manifest(args.metadata, args.exclude_manifest, args.out, args.families,
                                   args.per_family, args.seed, args.max_frames)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
