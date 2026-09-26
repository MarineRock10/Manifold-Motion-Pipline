import json
from pathlib import Path

from manifold_motion.dataio.failure_mining import _cluster, cluster


def test_failure_priority_rules():
    assert _cluster(["jump_insufficient_lift"]) == "jump_phase"
    assert _cluster(["nonfoot_floor_contact"]) == "nonfoot_contact"
    assert _cluster(["tracking_error", "fall_or_extreme_roll"]) == "tracking_instability"


def test_failure_report_is_deterministic(tmp_path: Path):
    root = tmp_path / "replay" / "summaries"
    root.mkdir(parents=True)
    (root / "bad.json").write_text(json.dumps({
        "motion_id": "bad", "accepted": False,
        "failed_checks": ["jump_no_verified_landing"], "stratum": "jump",
    }))
    report = cluster([tmp_path / "replay"], tmp_path / "out")
    assert report["failure_count"] == 1
    assert report["cluster_counts"] == {"jump_phase": 1}
    assert (tmp_path / "out" / "targeted_seed_supplement.csv").exists()
