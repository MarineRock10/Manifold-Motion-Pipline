"""Select a small actor-disjoint capability slice from the full SEED metadata CSV.

The Hugging Face archive is intentionally large.  This utility chooses only the G1 CSV members
needed to establish whether the frozen SONIC executor can admit a new primitive before any
dynamic model is trained.  It outputs the same manifest contract as ``seed_replay`` and never
silently treats a semantic filename as a verified capability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def _split(actor_uid: str) -> str:
    bucket = hashlib.sha256(actor_uid.encode("utf-8")).digest()[0] % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def _rank(seed: int, path: str) -> bytes:
    return hashlib.sha256(f"{seed}:{path}".encode("utf-8")).digest()


def main() -> int:
    parser = argparse.ArgumentParser(description="make an actor-disjoint SEED metadata capability manifest")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--name-regex", required=True, help="case-insensitive regex on move_name")
    parser.add_argument("--label", required=True, help="Stage-2 primitive label, e.g. jump or crawl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=12)
    parser.add_argument("--validation", type=int, default=4)
    parser.add_argument("--test", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--include-mirrored", action="store_true")
    args = parser.parse_args()
    if min(args.train, args.validation, args.test) < 1:
        parser.error("every requested split must be positive")
    pattern = re.compile(args.name_regex, flags=re.IGNORECASE)
    with args.metadata.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = []
    for row in rows:
        path = row.get("move_g1_path", "")
        name = row.get("move_name", "")
        if not path or not name or not pattern.search(name):
            continue
        if not args.include_mirrored and path.lower().endswith("_m.csv"):
            continue
        candidates.append(row)
    if not candidates:
        raise ValueError("metadata query found no G1 rows")
    fields = ["motion_id", "pilot_stratum", "move_g1_path", "move_name", "actor_uid", "selection_role"]
    desired = {"train": args.train, "validation": args.validation, "test": args.test}
    selected: list[dict[str, str]] = []
    report: dict[str, object] = {"metadata": str(args.metadata), "name_regex": args.name_regex,
                                 "label": args.label, "seed": args.seed, "requested": desired,
                                 "available": {}, "selected": {}}
    for split, count in desired.items():
        unique: dict[str, dict[str, str]] = {}
        for row in sorted((row for row in candidates if _split(row.get("actor_uid", "unknown")) == split),
                          key=lambda item: _rank(args.seed, item["move_g1_path"])):
            unique.setdefault(row["move_g1_path"], row)
        chosen = list(unique.values())[:count]
        if len(chosen) < count:
            raise ValueError(f"{split} has {len(chosen)} candidates, requested {count}")
        report["available"][split] = len(unique)  # type: ignore[index]
        report["selected"][split] = {  # type: ignore[index]
            "rows": len(chosen), "actors": sorted({row.get("actor_uid", "") for row in chosen})}
        for ordinal, row in enumerate(chosen):
            stem = Path(row["move_g1_path"]).stem
            selected.append({"motion_id": f"metadata_{args.label}_{split}_{ordinal:02d}_{stem}",
                             "pilot_stratum": args.label, "move_g1_path": row["move_g1_path"],
                             "move_name": row["move_name"], "actor_uid": row.get("actor_uid", ""),
                             "selection_role": f"actor_disjoint_{split}"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
