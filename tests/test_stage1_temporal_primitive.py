import json

import numpy as np

from manifold_motion.stage1.temporal_primitive import (
    TemporalPrimitiveNet, _metrics, confirmation_split,
)
from manifold_motion.stage1.temporal_aggregate import aggregate


def _tiny(tmp_path):
    n, t = 6, 4
    payload = {
        "corridor": np.zeros((n, t, 7), np.float32),
        "target_exec": np.zeros((n, t, 38), np.float32),
        "primitive": np.array([0, 0, 1, 1, 0, 1], np.int64),
        "primitive_names": np.array(["walk", "crouch"]),
        "split": np.array([0, 0, 0, 1, 2, 0], np.uint8),
        "clip_index": np.array([0, 0, 1, 1, 0, 1], np.int64),
    }
    catalog = tmp_path / "catalog.npz"
    np.savez_compressed(catalog, **payload)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"clips": [{"actor_uid": "A0"}, {"actor_uid": "A1"}]}))
    return catalog, metadata


def test_confirmation_split_is_deterministic(tmp_path):
    catalog, metadata = _tiny(tmp_path)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    confirmation_split(catalog, metadata, first, modulo=2)
    confirmation_split(catalog, metadata, second, modulo=2)
    with np.load(first) as a, np.load(second) as b:
        np.testing.assert_array_equal(a["split"], b["split"])
        assert set(np.unique(a["split"])).issubset({0, 1, 2, 3})
        assert 2 in set(np.unique(a["split"])) and 3 in set(np.unique(a["split"]))


def test_temporal_head_contract_and_metrics():
    import torch

    model = TemporalPrimitiveNet.build(torch, 7, 16, 3)
    sequence, logits = model(torch.zeros((2, 4, 7)))
    assert sequence.shape == (2, 4, 29)
    assert logits.shape == (2, 3)
    metric = _metrics(np.zeros((2, 4, 29), np.float32), np.ones((2, 4, 29), np.float32),
                      np.array([0, 1]), np.array([0, 1]), np.array(["walk", "crouch"]), "test")
    assert metric["joint_mae_rad"] == 1.0
    assert metric["family_top1"] == 1.0


def test_multiseed_aggregate(tmp_path):
    summaries = []
    physical = []
    for seed, mae in ((1, .2), (2, .3)):
        sp = tmp_path / f"s{seed}.json"
        sp.write_text(json.dumps({"seed": seed, "metrics": {"eval": {
            "rows": 5, "joint_mae_rad": mae, "velocity_mae_rad": .1,
            "family_top1": .5}}}))
        pp = tmp_path / f"p{seed}.json"
        pp.write_text(json.dumps({"rows": [{}, {}], "aggregate": {
            "predicted_acceptance": .75, "oracle_acceptance": .5,
            "predicted_tracking_mae_rad": .08}}))
        summaries.append(sp); physical.append(pp)
    result = aggregate(summaries, physical, tmp_path / "aggregate.json")
    assert result["metrics"]["joint_mae_rad"]["mean"] == .25
    assert result["physical_rows_per_seed"] == [2, 2]
