"""Build and audit a bounded, actor-disjoint SEED capability catalogue.

Semantic filenames only retrieve candidates.  A skill enters the usable library only after its
selected G1 CSV is replayed by :mod:`manifold_motion.dataio.seed_replay` and passes the frozen
SONIC/MuJoCo gates.  The catalogue deliberately covers navigation, constrained locomotion,
recovery and interaction fragments so Stage 2 can learn composition rather than four labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SkillFamily:
    family_id: int
    name: str
    pattern: str
    gate_profile: str
    environment_trigger: str
    body_mask: str
    risk: str = "medium"
    exclude: str = ""


FAMILIES: tuple[SkillFamily, ...] = (
    SkillFamily(0, "walk_forward", r"(?:neutral_walk|walk_ff|walk_forward|loop_forward_walk)", "walk_nominal", "open corridor / nominal progress", "legs+pelvis", "low", r"(?:crouch|jog|crate|handbag)"),
    SkillFamily(1, "jog_forward", r"(?:jog_ff|jog_forward)", "walk_nominal", "time-critical open corridor", "legs+pelvis", "medium"),
    SkillFamily(2, "hands_back_walk", r"walk_hands_on_back", "walk_nominal", "arm-constrained passage / reduced arm sweep", "legs+pelvis", "medium"),
    SkillFamily(3, "walk_lateral", r"(?:walk_sideway|sideway_walk|lateral_speed_step)", "walk_lateral_reverse", "narrow or lateral aperture", "pelvis+legs+arms", "low"),
    SkillFamily(4, "walk_curve", r"(?:walk_arc|arc_walk)", "walk_turn", "curved route", "pelvis+legs", "low"),
    SkillFamily(5, "turn_in_place", r"(?:idle_turn|step_rotate_idle|neutral_turning_around)", "walk_turn", "heading discontinuity", "pelvis+legs", "low"),
    SkillFamily(6, "crouch_walk", r"crouch_ff_loop", "crouch", "long low ceiling", "trunk+hips+knees", "medium"),
    SkillFamily(7, "crouch_transition", r"(?:idle_crouch_(?:start|stop|to_|right_to_)|crouch_ff_(?:start|stop))", "low_transition", "enter or leave low clearance", "trunk+hips+knees", "medium"),
    SkillFamily(8, "bend_duck_walk", r"(?:bend_down.*walk|duck.*walk|avoid_obstacle_bend)", "crouch", "short overhead obstacle", "trunk+hips", "medium"),
    SkillFamily(9, "dodge_lateral", r"(?:dodge|avoid_car|evad)", "walk_lateral_reverse", "fast crossing obstacle", "trunk+pelvis+legs", "medium"),
    SkillFamily(10, "forward_lunge", r"(?:forward_lunge|lunge_forward)", "low_transition", "push recovery / extended reach", "legs+trunk", "medium"),
    SkillFamily(11, "side_hop", r"(?:lateral_hop|side_hop)", "jump", "small lateral ground hazard", "legs+pelvis", "high"),
    SkillFamily(12, "jump_forward", r"jump_ff", "jump", "short ground gap", "legs+pelvis", "high", r"(?:high_jump|box_jump|broad_jump)"),
    SkillFamily(13, "broad_jump", r"broad_jump", "jump", "wide ground gap", "legs+pelvis+arms", "high"),
    SkillFamily(14, "high_jump", r"high_jump", "jump", "tall ground obstacle", "legs+pelvis+arms", "high"),
    SkillFamily(15, "box_jump", r"box_jump(?!_down)", "jump", "raised platform", "legs+pelvis+arms", "high"),
    SkillFamily(16, "step_up_box", r"(?:come_up_50cm_box|step_up|climb_(?:on_)?box)", "walk_nominal", "step or platform ascent", "swing_leg+pelvis", "high"),
    SkillFamily(17, "step_down_box", r"(?:come_down_50cm_box|box_jump_down|step_down)", "walk_nominal", "step or platform descent", "legs+pelvis", "high"),
    SkillFamily(18, "kneel", r"(?:kneel|kneeling)", "low_transition", "very low stationary clearance", "trunk+hips+knees", "medium"),
    SkillFamily(19, "crawl", r"crawl_ff", "crawl", "long very-low tunnel", "whole_body", "high", r"spider"),
    SkillFamily(20, "all_fours", r"(?:all_fours|on_all_fours)", "all_fours", "four-contact low passage", "whole_body", "high"),
    SkillFamily(21, "spider_crawl", r"spider_crawl", "crawl", "low passage with inverted support", "whole_body", "high"),
    SkillFamily(22, "inchworm", r"inchworm", "crawl", "low passage / recovery", "whole_body", "high"),
    SkillFamily(23, "get_up_recovery", r"(?:stand_up_(?:lying|prone|supine)|crutch_get_up|faint_stand_up|get_up)", "low_transition", "fall recovery", "whole_body", "high"),
    SkillFamily(24, "vault", r"vault_over", "jump", "waist-height obstacle", "whole_body", "high"),
    SkillFamily(25, "door_interaction", r"(?:door_handle.*open|open.*door|door.*open)", "low_transition", "doorway interaction", "arms+trunk+pelvis", "high"),
    SkillFamily(26, "ladder", r"ladder_climbing", "all_fours", "vertical traversal", "whole_body", "high"),
    SkillFamily(27, "button_lever", r"(?:high_button|horizontal_handle.*lever|vertical_lever|push_button)", "low_transition", "reachable environment control", "arm+trunk", "medium"),
    SkillFamily(28, "carry_object", r"(?:carry.*(?:box|object)|(?:box|object).*carry|lift_crate.*walk|handbag_walk)", "walk_nominal", "navigation with changed self-manifold", "arms+trunk+legs", "high"),
    SkillFamily(29, "roll_recovery", r"(?:safety_roll|forward_roll|backward_roll)", "low_transition", "impact dissipation / recovery", "whole_body", "high"),
)


def actor_split(actor_uid: str) -> str:
    bucket = hashlib.sha256(actor_uid.encode("utf-8")).digest()[0] % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def _rank(seed: int, family: str, path: str) -> bytes:
    return hashlib.sha256(f"{seed}:{family}:{path}".encode("utf-8")).digest()


def classify(move_name: str) -> list[SkillFamily]:
    matches = []
    for family in FAMILIES:
        if re.search(family.pattern, move_name, flags=re.IGNORECASE) and not (
            family.exclude and re.search(family.exclude, move_name, flags=re.IGNORECASE)
        ):
            matches.append(family)
    return matches


def _load_candidates(metadata: Path, *, max_frames: int, include_mirrored: bool) -> dict[str, list[dict[str, str]]]:
    result = {family.name: [] for family in FAMILIES}
    with metadata.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for family in FAMILIES:
        for row in rows:
            path = row.get("move_g1_path", "")
            name = row.get("move_name", "")
            if not path or not name:
                continue
            if not include_mirrored and row.get("is_mirror", "").lower() in {"true", "1", "1.0"}:
                continue
            try:
                frames = int(float(row.get("move_duration_frames", "0")))
            except ValueError:
                continue
            if frames < 60 or frames > max_frames:
                continue
            if family not in classify(name):
                continue
            enriched = dict(row)
            enriched["actor_split"] = actor_split(row.get("actor_uid", "unknown"))
            result[family.name].append(enriched)
    return result


def build_catalog(metadata: Path, output_dir: Path, *, train: int, validation: int, test: int,
                  seed: int, max_frames: int, include_mirrored: bool) -> dict[str, object]:
    candidates = _load_candidates(metadata, max_frames=max_frames, include_mirrored=include_mirrored)
    requested = {"train": train, "validation": validation, "test": test}
    rows: list[dict[str, str]] = []
    pilot_rows: list[dict[str, str]] = []
    report_families = []
    fields = ["motion_id", "family_id", "skill_family", "pilot_stratum", "move_g1_path",
              "move_name", "actor_uid", "selection_role", "environment_trigger", "body_mask", "risk"]
    for family in FAMILIES:
        family_rows = candidates[family.name]
        selected_family: list[dict[str, str]] = []
        available = {}
        selected_counts = {}
        for split, count in requested.items():
            unique = {row["move_g1_path"]: row for row in family_rows if row["actor_split"] == split}
            ordered = sorted(unique.values(), key=lambda row: _rank(seed, "walk_backward" if family.family_id == 2 else family.name, row["move_g1_path"]))
            chosen = ordered[:count]
            available[split] = len(ordered)
            selected_counts[split] = len(chosen)
            for ordinal, row in enumerate(chosen):
                selected_family.append({
                    "motion_id": f"cap_{family.family_id:02d}_{family.name}_{split}_{ordinal:02d}",
                    "family_id": str(family.family_id), "skill_family": family.name,
                    "pilot_stratum": family.gate_profile, "move_g1_path": row["move_g1_path"],
                    "move_name": row["move_name"], "actor_uid": row.get("actor_uid", ""),
                    "selection_role": f"actor_disjoint_{split}",
                    "environment_trigger": family.environment_trigger, "body_mask": family.body_mask,
                    "risk": family.risk,
                })
        rows.extend(selected_family)
        pilot = next((row for row in selected_family if row["selection_role"] == "actor_disjoint_test"), None)
        pilot = pilot or next((row for row in selected_family if row["selection_role"] == "actor_disjoint_validation"), None)
        pilot = pilot or (selected_family[0] if selected_family else None)
        if pilot is not None:
            pilot_rows.append(dict(pilot))
        report_families.append({**asdict(family), "available": available, "selected": selected_counts,
                                "pilot_selected": pilot is not None,
                                "actor_count": len({row.get("actor_uid", "") for row in family_rows})})
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "seed_capability_manifest_v1.csv"
    pilot_manifest = output_dir / "seed_capability_pilot_v1.csv"
    for path, content in ((manifest, rows), (pilot_manifest, pilot_rows)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(content)
    report: dict[str, object] = {
        "schema": "manifold-motion.seed-capability-catalog.v1", "metadata": str(metadata),
        "family_count": len(FAMILIES), "families_with_candidates": sum(bool(candidates[x.name]) for x in FAMILIES),
        "families_with_pilot": len(pilot_rows), "selected_rows": len(rows), "pilot_rows": len(pilot_rows),
        "requested_per_split": requested, "seed": seed, "max_frames": max_frames,
        "manifest": str(manifest), "pilot_manifest": str(pilot_manifest), "families": report_families,
        "semantic_labels_are_capabilities": False,
        "admission_rule": "Only SONIC/MuJoCo accepted replays may train the latent prior.",
    }
    (output_dir / "seed_capability_catalog_v1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _manifest_paths(manifest: Path) -> list[str]:
    with manifest.open(newline="", encoding="utf-8") as stream:
        return [row["move_g1_path"] for row in csv.DictReader(stream)]


def extract_selected(archive: Path, manifest: Path, output_dir: Path) -> dict[str, object]:
    """Extract only manifest members using the native tar implementation.

    The archive is gzip-compressed, so the stream still has to be read once, but native tar is
    substantially faster and uses bounded Python memory compared with ``tarfile.getmembers()``.
    """
    wanted = _manifest_paths(manifest)
    if any(not name.startswith("g1/csv/") or ".." in Path(name).parts for name in wanted):
        raise ValueError("manifest contains an unsafe archive member")
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_wanted = [name for name in sorted(set(wanted)) if not (output_dir / name).is_file()]
    if not missing_wanted:
        return {"archive": str(archive), "manifest": str(manifest), "output_dir": str(output_dir),
                "requested": len(wanted), "extracted": 0, "skipped_existing": len(set(wanted))}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as stream:
        for name in missing_wanted:
            stream.write(name + "\n")
        list_path = Path(stream.name)
    try:
        command = ["tar", "-xzf", str(archive), "-C", str(output_dir), "--files-from", str(list_path)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"tar extraction failed: {completed.stderr.strip()}")
    finally:
        list_path.unlink(missing_ok=True)
    missing = [name for name in wanted if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} selected members were not extracted; first={missing[0]}")
    return {"archive": str(archive), "manifest": str(manifest), "output_dir": str(output_dir),
            "requested": len(wanted), "extracted": len(missing_wanted),
            "skipped_existing": len(set(wanted)) - len(missing_wanted)}


def build_confirmation_manifest(full_manifest: Path, pilot_manifest: Path, replay_dir: Path,
                                output: Path, per_family: int) -> dict[str, object]:
    """Expand only pilot-accepted families with actor-diverse confirmation clips."""
    def read_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    full = read_rows(full_manifest)
    pilots = read_rows(pilot_manifest)
    pilot_by_family = {row["skill_family"]: row for row in pilots}
    accepted_families = set()
    for row in pilots:
        summary = replay_dir / "summaries" / f"{row['motion_id']}.json"
        if summary.is_file() and json.loads(summary.read_text(encoding="utf-8")).get("accepted"):
            accepted_families.add(row["skill_family"])
    selected: list[dict[str, str]] = []
    for family in FAMILIES:
        if family.name not in accepted_families:
            continue
        pilot = pilot_by_family[family.name]
        candidates = [row for row in full if row["skill_family"] == family.name]
        # Keep the passing pilot and then prefer new actors and new split roles.
        chosen = [pilot]
        actors = {pilot.get("actor_uid", "")}
        roles = {pilot.get("selection_role", "")}
        ordered = sorted(candidates, key=lambda row: (
            row.get("actor_uid", "") in actors,
            row.get("selection_role", "") in roles,
            row["motion_id"],
        ))
        for row in ordered:
            if row["move_g1_path"] == pilot["move_g1_path"]:
                continue
            if row.get("actor_uid", "") in actors and len({x.get("actor_uid", "") for x in candidates}) > 1:
                continue
            chosen.append(row); actors.add(row.get("actor_uid", "")); roles.add(row.get("selection_role", ""))
            if len(chosen) >= per_family:
                break
        selected.extend(chosen)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(selected[0]) if selected else []
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(selected)
    return {"accepted_pilot_families": len(accepted_families), "rows": len(selected),
            "per_family": per_family, "output": str(output)}


def summarize_replays(manifest: Path, replay_dirs: list[Path], output: Path) -> dict[str, object]:
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_family: dict[str, list[dict[str, object]]] = {family.name: [] for family in FAMILIES}
    for row in rows:
        path = next((root / "summaries" / f"{row['motion_id']}.json" for root in replay_dirs
                     if (root / "summaries" / f"{row['motion_id']}.json").is_file()), None)
        if path is not None:
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["actor_uid"] = row.get("actor_uid", "")
            by_family[row["skill_family"]].append(summary)
    family_reports = []
    for family in FAMILIES:
        values = by_family[family.name]
        accepted = [value for value in values if value.get("accepted")]
        if not values: status = "untested"
        elif len(accepted) >= 2 and len({x.get("actor_uid") for x in accepted}) >= 2: status = "supported"
        elif accepted: status = "partial"
        else: status = "unsupported"
        failures: dict[str, int] = {}
        for value in values:
            for check in value.get("failed_checks", []): failures[check] = failures.get(check, 0) + 1
        family_reports.append({"family_id": family.family_id, "skill_family": family.name,
                               "gate_profile": family.gate_profile, "status": status,
                               "replayed": len(values), "accepted": len(accepted),
                               "acceptance_rate": len(accepted) / len(values) if values else None,
                               "accepted_actors": sorted({x.get("actor_uid", "") for x in accepted}),
                               "failed_checks": failures})
    result: dict[str, object] = {
        "schema": "manifold-motion.seed-physical-capabilities.v1", "controller": "frozen GEAR-SONIC ONNX",
        "manifest": str(manifest), "replay_dirs": [str(path) for path in replay_dirs], "families": family_reports,
        "counts": {status: sum(row["status"] == status for row in family_reports)
                   for status in ("supported", "partial", "unsupported", "untested")},
        "admission_rule": "supported and partial clips remain separated; unsupported clips never train the latent prior",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--metadata", type=Path, required=True)
    build.add_argument("--out", type=Path, default=Path("data/seed_capability_v1"))
    build.add_argument("--train", type=int, default=6); build.add_argument("--validation", type=int, default=2); build.add_argument("--test", type=int, default=2)
    build.add_argument("--seed", type=int, default=20260927); build.add_argument("--max-frames", type=int, default=1800)
    build.add_argument("--include-mirrored", action="store_true")
    extract = sub.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True); extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--out", type=Path, default=Path("data/seed_capability_v1"))
    confirm = sub.add_parser("confirm")
    confirm.add_argument("--manifest", type=Path, required=True); confirm.add_argument("--pilot-manifest", type=Path, required=True)
    confirm.add_argument("--replay-dir", type=Path, required=True); confirm.add_argument("--out", type=Path, required=True)
    confirm.add_argument("--per-family", type=int, default=3)
    summary = sub.add_parser("summarize")
    summary.add_argument("--manifest", type=Path, required=True); summary.add_argument("--replay-dir", type=Path, nargs="+", required=True)
    summary.add_argument("--out", type=Path, default=Path("reports/manifold_motion/seed_capability_v1/capabilities.json"))
    args = parser.parse_args()
    if args.command == "build": result = build_catalog(args.metadata, args.out, train=args.train, validation=args.validation, test=args.test, seed=args.seed, max_frames=args.max_frames, include_mirrored=args.include_mirrored)
    elif args.command == "extract": result = extract_selected(args.archive, args.manifest, args.out)
    elif args.command == "confirm": result = build_confirmation_manifest(args.manifest, args.pilot_manifest, args.replay_dir, args.out, args.per_family)
    else: result = summarize_replays(args.manifest, args.replay_dir, args.out)
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())