"""Pairing must preserve source time, root coordinates, and policy joint order."""

import numpy as np
import pytest
from PIL import Image

from manifold_motion.core import constants as C
from manifold_motion.dataio.seed_windows import _relative_pose, _resample_record
from manifold_motion.simulation.env import G1FlatEnv
from manifold_motion.visualization.manifold_action_pairs import (
    _gif_durations,
    _load_window,
    _local_to_world,
    _save_gif,
    _select_rows,
    _set_pose,
)


def test_representatives_are_deterministic_one_per_present_family():
    targets = np.zeros((4, 3, 38))
    targets[2, :, 0] = [0, 1, 2]
    targets[3, :, 0] = [0, 2, 4]
    arrays = {"primitive_names": np.array(["walk", "crouch", "absent"]),
              "primitive": np.array([0, 1, 0, 1]), "target_exec": targets}
    assert _select_rows(arrays, max_families=0) == [3, 2]
    assert _select_rows(arrays, max_families=1) == [3]


def test_origin_rotation_maps_local_centres_to_world():
    quat = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    world = _local_to_world(np.array([[1, 0, 0], [0, 1, 1]]), np.array([4, 5, 6]), quat)
    np.testing.assert_allclose(world, [[4, 6, 6], [3, 5, 7]], atol=1e-7)


def _fixture(tmp_path):
    count = 14
    t = np.arange(count) / 30
    q = np.outer(t, np.linspace(-1, 1, 29))
    pos = np.column_stack([t, t**2, 0.8 + 0.01 * t])
    quat = np.tile([np.cos(0.3), 0, 0, np.sin(0.3)], (count, 1))
    source = {"t": t, "q_ref": q, "dq_ref": q * 0, "root_ref_pos_seed_m": pos,
              "root_ref_quat": quat, "q_exec": q, "dq_exec": q * 0, "base_pos": pos,
              "base_quat": quat, "base_lin_vel": pos * 0, "base_ang_vel": pos * 0,
              "action": q * 0, "source_frame": np.arange(count),
              "foot_contact": np.ones((count, 2)), "hand_contact": np.zeros((count, 2)),
              "nonfoot_floor_contact": np.zeros(count)}
    (tmp_path / "records").mkdir()
    np.savez(tmp_path / "records" / "clip_a.npz", **source)
    resampled = _resample_record(source, 30)
    origin = 3
    future = slice(origin + 1, origin + 5)
    relative_pos, relative_rot = _relative_pose(resampled["base_pos"], resampled["base_quat"], origin, future)
    target = np.concatenate([resampled["q_exec"][future], relative_pos, relative_rot], axis=1)
    corridor = np.concatenate([relative_pos, np.ones((4, 3)), np.zeros((4, 1))], axis=1)
    arrays = {"primitive_names": np.array(["walk"]), "primitive": np.array([0]),
              "clip_index": np.array([0]), "source_origin": np.array([origin]),
              "target_exec": target[None].astype(np.float32), "corridor": corridor[None],
              "self_manifold": np.ones((1, 4, 3)) * 0.5, "split": np.array([0]),
              "catalog_id": np.array(["me_action_000042"])}
    metadata = {"clips": [{"motion_id": "clip_a", "actor_uid": "A"}], "config": {"output_hz": 30}}
    return arrays, metadata, resampled, future


def test_source_frames_and_all_38_target_features_match(tmp_path):
    arrays, metadata, record, future = _fixture(tmp_path)
    sample = _load_window(0, arrays, metadata, tmp_path)
    assert sample["catalog_id"] == "me_action_000042"
    assert sample["max_target_error"] < 1e-6
    np.testing.assert_allclose(sample["q"], record["q_exec"][future])
    np.testing.assert_allclose(sample["env_centres"], record["base_pos"][future], atol=1e-7)
    np.testing.assert_allclose(sample["source_time"], np.arange(1, 5) / 30)


@pytest.mark.parametrize("feature", [0, 29, 32])
def test_wrong_joint_position_or_rotation_is_rejected(tmp_path, feature):
    arrays, metadata, _, _ = _fixture(tmp_path)
    arrays["target_exec"][0, 2, feature] += 0.1
    with pytest.raises(ValueError, match="mismatch"):
        _load_window(0, arrays, metadata, tmp_path)


def test_missing_source_record_is_not_replaced_by_synthetic_pose(tmp_path):
    arrays, metadata, _, _ = _fixture(tmp_path)
    metadata["clips"][0]["motion_id"] = "not_present"
    with pytest.raises(FileNotFoundError):
        _load_window(0, arrays, metadata, tmp_path)


def test_action_replay_uses_absolute_joints_without_adding_default_pose():
    env = G1FlatEnv()
    q = C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB] + 0.05
    pos = np.array([1.2, -0.4, 0.78])
    quat = np.array([1, 0, 0, 0])
    _set_pose(env, q, pos, quat)
    np.testing.assert_allclose(env.data.qpos[env.body_qadr], C.DEFAULT_ANGLES + 0.05)
    np.testing.assert_allclose(env.data.qpos[:3], pos)


def test_separate_and_combined_gifs_have_identical_exact_timeline(tmp_path):
    left = [Image.new("RGB", (32, 24), (i * 5, 50, 100)) for i in range(36)]
    right = [Image.new("RGB", (32, 24), (100, i * 5, 40)) for i in range(36)]
    joined = []
    for a, b in zip(left, right):
        frame = Image.new("RGB", (64, 24))
        frame.paste(a, (0, 0))
        frame.paste(b, (32, 0))
        joined.append(frame)
    timelines = []
    for label, frames in (("manifold", left), ("action", right), ("pair", joined)):
        path = tmp_path / f"{label}.gif"
        _save_gif(frames, path, 15)
        with Image.open(path) as gif:
            assert gif.n_frames == 36
            assert gif.info["loop"] == 0
            durations = []
            for i in range(gif.n_frames):
                gif.seek(i)
                durations.append(gif.info["duration"])
            timelines.append(durations)
    assert timelines[0] == timelines[1] == timelines[2] == _gif_durations(36, 15)
    assert sum(timelines[0]) == 2400
