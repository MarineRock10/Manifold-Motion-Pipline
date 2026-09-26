"""Make a deterministic actor-disjoint manifest slice from the downloaded SEED pilot.

This is used to enlarge an under-covered primitive without moving an actor between splits.
It selects a bounded number of distinct motion rows *within* the stable actor hash partitions;
the output retains all original manifest fields and is suitable for ``seed_replay`` directly.
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


def _rank(seed: int, row: dict[str, str]) -> bytes:
    return hashlib.sha256(f"{seed}:{row['motion_id']}".encode("utf-8")).digest()


def main() -> int:
    parser = argparse.ArgumentParser(description="select an actor-disjoint SEED primitive manifest slice")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stratum", required=True)
    parser.add_argument("--name-regex", default=None,
                        help="optional case-insensitive regular expression applied to move_name")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--validation", type=int, default=20)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if min(args.train, args.validation, args.test) < 1:
        parser.error("every split count must be positive")
    with args.manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if row.get("pilot_stratum") == args.stratum]
    pattern = re.compile(args.name_regex, flags=re.IGNORECASE) if args.name_regex else None
    if pattern is not None:
        rows = [row for row in rows if pattern.search(row.get("move_name", ""))]
    if not fields or not rows:
        raise ValueError(f"no '{args.stratum}' rows in {args.manifest}")
    desired = {"train": args.train, "validation": args.validation, "test": args.test}
    selected: list[dict[str, str]] = []
    report = {"source": str(args.manifest), "stratum": args.stratum, "name_regex": args.name_regex, "seed": args.seed,
              "requested": desired, "available": {}, "selected": {}}
    for split, count in desired.items():
        candidates = [row for row in rows if _split(row.get("actor_uid", "unknown")) == split]
        # Stable hash ordering prevents path/date prefixes from accidentally choosing only one
        # capture session.  Deduplicate exact CSV paths before slicing mirrored metadata rows.
        unique: dict[str, dict[str, str]] = {}
        for row in sorted(candidates, key=lambda item: _rank(args.seed, item)):
            unique.setdefault(row["move_g1_path"], row)
        chosen = list(unique.values())[:count]
        if len(chosen) < count:
            raise ValueError(f"{split} has only {len(chosen)} available {args.stratum} paths, requested {count}")
        selected.extend(chosen)
        report["available"][split] = len(unique)
        report["selected"][split] = {"rows": len(chosen),
                                      "actors": sorted({row.get("actor_uid", "") for row in chosen})}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    (args.out.with_suffix(".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
