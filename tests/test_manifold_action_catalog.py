import json

import numpy as np

from manifold_motion.dataio.manifold_action_catalog import build_catalog


def _source(tmp_path):
    windows = tmp_path / "windows.npz"
    metadata = tmp_path / "windows_metadata.json"
    n, horizon = 3, 4
    corridor = np.zeros((n, horizon, 7), dtype=np.float32)
    corridor[:, :, 3:6] = np.asarray([1.0, 0.7, 1.2], dtype=np.float32)
    corridor[1, :, 3:6] *= 0.9
    corridor[2, :, 3:6] *= 1.1
    arrays = {
        "state": np.zeros((n, 69), dtype=np.float32),
        "history": np.zeros((n, 2, 69), dtype=np.float32),
        "primitive": np.asarray([0, 1, 0], dtype=np.int64),
        "primitive_names": np.asarray(["walk_forward", "walk_lateral"]),
        "target_ref": np.zeros((n, horizon, 38), dtype=np.float32),
        "target_exec": np.zeros((n, horizon, 38), dtype=np.float32),
        "split": np.asarray([0, 1, 2], dtype=np.uint8),
        "clip_index": np.asarray([0, 1, 2], dtype=np.int32),
        "source_origin": np.asarray([3, 5, 7], dtype=np.int32),
        "corridor": corridor,
        "sdf": np.zeros((n, 3, 3, 2), dtype=np.float32),
        "self_manifold": np.ones((n, horizon, 3), dtype=np.float32),
    }
    np.savez_compressed(windows, **arrays)
    metadata.write_text(json.dumps({"clips": [
        {"motion_id": "clip_a", "actor_uid": "A", "primitive": "walk_forward"},
        {"motion_id": "clip_b", "actor_uid": "B", "primitive": "walk_lateral"},
        {"motion_id": "clip_c", "actor_uid": "C", "primitive": "walk_forward"},
    ]}))
    return windows, metadata


def test_catalog_preserves_conditions_and_actor_disjoint_split(tmp_path):
    windows, metadata = _source(tmp_path)
    report = build_catalog(windows, metadata, tmp_path / "catalog")
    assert report["rows"] == 3
    assert report["family_count"] == 2
    assert report["split_counts"] == {"test": 1, "train": 1, "validation": 1}
    rows = (tmp_path / "catalog" / "manifold_action_catalog_v1.csv").read_text().splitlines()
    assert "source_motion_id" in rows[0]
    assert "reverse_synthesized_from_accepted_R_exec" in rows[1]
    with np.load(tmp_path / "catalog" / "manifold_action_catalog_v1.npz") as archive:
        assert archive["corridor"].shape == (3, 4, 7)
        assert archive["catalog_id"].tolist() == ["me_action_000000", "me_action_000001", "me_action_000002"]
