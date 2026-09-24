"""Regression test for the honest 4 x 26 primary structural smoke matrix."""

from __future__ import annotations

import json
from pathlib import Path

from manifold_motion.cvpr_smoke import run_smoke


def test_primary_smoke_has_explicit_104_row_coverage(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "primary_smoke.jsonl"
    report = run_smoke(root / "configs" / "cvpr_simulation_protocol.json", output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert report["accepted"]
    assert report["rows"] == 104
    assert len({row["scenario"] for row in rows}) == 26
    assert {row["method"] for row in rows} == {"B2", "Ours-2", "Ours-3", "Ours-4"}
    assert all(row["evidence_type"] == "structural_preflight_only" for row in rows)
    assert all("method_profile" in row and "scenario_adapter" in row for row in rows)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_primary_smoke_has_explicit_104_row_coverage(Path(directory))
    print("CVPR structural smoke tests passed")
