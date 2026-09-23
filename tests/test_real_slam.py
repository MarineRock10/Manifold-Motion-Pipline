"""Timestamp/extrinsic regression tests for the hardware SLAM ingress."""

from __future__ import annotations

import numpy as np

from manifold_motion.real_slam import SlamSyncConfig, TimeSynchronizedSlamAdapter


def test_interpolation_and_extrinsic() -> None:
    base_T_sensor = np.eye(4)
    base_T_sensor[:3, 3] = [0.2, 0.0, 0.1]
    adapter = TimeSynchronizedSlamAdapter(base_T_sensor, SlamSyncConfig(max_pose_gap_s=0.2))
    adapter.push_pose(1.0, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), 0.01)
    adapter.push_pose(1.1, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), 0.02)
    scan, report = adapter.ingest_pointcloud(np.array([[1.0, 0.0, 0.0]]), 1.05)
    assert np.allclose(scan.origins_world[0], [0.7, 0.0, 0.1], atol=1e-6)
    assert np.allclose(scan.points_world[0], [1.7, 0.0, 0.1], atol=1e-6)
    assert np.isclose(report["interpolation_weight"], 0.5)


def test_stale_scan_is_rejected() -> None:
    adapter = TimeSynchronizedSlamAdapter(np.eye(4), SlamSyncConfig(max_extrapolation_s=0.01))
    adapter.push_pose(1.0, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    try:
        adapter.ingest_pointcloud(np.zeros((1, 3)), 1.2)
    except RuntimeError as error:
        assert "exceeds pose buffer" in str(error)
    else:
        raise AssertionError("stale point cloud was accepted")
    assert adapter.summary()["rejected_scans"] == 1


if __name__ == "__main__":
    test_interpolation_and_extrinsic()
    test_stale_scan_is_rejected()
    print("real SLAM adapter tests passed")
