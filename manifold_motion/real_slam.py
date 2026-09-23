"""Hardware-facing SLAM pose, timestamp and radar-extrinsic adapter.

This module has no ROS dependency so its frame/time contract can be tested in CI and reused by
ROS2, LCM or a vendor SDK.  A transport callback pushes timestamped SLAM poses and point clouds;
the adapter interpolates ``world_T_base``, applies calibrated ``base_T_sensor`` and emits the
same world-frame :class:`RadarScan` consumed by the probabilistic voxel mapper.
"""

from __future__ import annotations

import bisect
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deploy_perception import RadarScan


def _quat_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _slerp(first: np.ndarray, second: np.ndarray, weight: float) -> np.ndarray:
    q0 = np.asarray(first, dtype=np.float64)
    q1 = np.asarray(second, dtype=np.float64)
    q0 /= max(float(np.linalg.norm(q0)), 1.0e-12)
    q1 /= max(float(np.linalg.norm(q1)), 1.0e-12)
    dot = float(q0 @ q1)
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        value = q0 + weight * (q1 - q0)
        return value / max(float(np.linalg.norm(value)), 1.0e-12)
    angle = float(np.arccos(dot))
    return ((np.sin((1.0 - weight) * angle) * q0
             + np.sin(weight * angle) * q1) / np.sin(angle))


@dataclass(frozen=True)
class TimedPose:
    timestamp_s: float
    position_world_m: np.ndarray
    quat_world_base_wxyz: np.ndarray
    position_covariance_trace: float | None = None


@dataclass(frozen=True)
class SlamSyncConfig:
    max_pose_gap_s: float = 0.10
    max_extrapolation_s: float = 0.025
    sensor_time_offset_s: float = 0.0
    max_position_covariance_trace: float = 0.20
    pose_buffer_size: int = 512
    world_frame: str = "map"
    base_frame: str = "base_link"
    sensor_frame: str = "radar_link"

    def validate(self) -> None:
        if min(self.max_pose_gap_s, self.max_extrapolation_s,
               self.max_position_covariance_trace, self.pose_buffer_size) <= 0:
            raise ValueError("SLAM synchronization limits must be positive")


class TimeSynchronizedSlamAdapter:
    """Interpolate SLAM odometry and apply a calibrated radar/LiDAR extrinsic."""

    def __init__(self, base_T_sensor: np.ndarray, config: SlamSyncConfig = SlamSyncConfig()):
        config.validate()
        transform = np.asarray(base_T_sensor, dtype=np.float64)
        if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError("base_T_sensor must be a homogeneous 4x4 transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.999:
            raise ValueError("base_T_sensor rotation must be right-handed orthonormal")
        self.base_T_sensor = transform.copy()
        self.config = config
        self.poses: deque[TimedPose] = deque(maxlen=config.pose_buffer_size)
        self.accepted_scans = 0
        self.rejected_scans = 0
        self.last_report: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, path: Path) -> "TimeSynchronizedSlamAdapter":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.asarray(payload["base_T_sensor"], dtype=np.float64),
                   SlamSyncConfig(**payload.get("sync", {})))

    def push_pose(self, timestamp_s: float, position_world_m: np.ndarray,
                  quat_world_base_wxyz: np.ndarray,
                  position_covariance_trace: float | None = None) -> None:
        timestamp = float(timestamp_s)
        position = np.asarray(position_world_m, dtype=np.float64).reshape(3)
        quat = np.asarray(quat_world_base_wxyz, dtype=np.float64).reshape(4)
        if not np.isfinite(timestamp) or not np.isfinite(position).all() or not np.isfinite(quat).all():
            raise ValueError("SLAM pose must be finite")
        quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
        covariance = None if position_covariance_trace is None else float(position_covariance_trace)
        pose = TimedPose(timestamp, position, quat, covariance)
        if self.poses and timestamp < self.poses[-1].timestamp_s:
            values = list(self.poses)
            index = bisect.bisect_left([item.timestamp_s for item in values], timestamp)
            values.insert(index, pose)
            self.poses = deque(values[-self.config.pose_buffer_size:],
                               maxlen=self.config.pose_buffer_size)
        else:
            self.poses.append(pose)

    def _pose_at(self, timestamp_s: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if not self.poses:
            raise RuntimeError("no SLAM pose is buffered")
        values = list(self.poses)
        stamps = [item.timestamp_s for item in values]
        index = bisect.bisect_left(stamps, timestamp_s)
        if index == 0:
            first = second = values[0]
            extrapolation = first.timestamp_s - timestamp_s
            if extrapolation > self.config.max_extrapolation_s:
                raise RuntimeError(f"point cloud precedes pose buffer by {extrapolation:.6f}s")
            weight = 0.0
        elif index == len(values):
            first = second = values[-1]
            extrapolation = timestamp_s - first.timestamp_s
            if extrapolation > self.config.max_extrapolation_s:
                raise RuntimeError(f"point cloud exceeds pose buffer by {extrapolation:.6f}s")
            weight = 0.0
        else:
            first, second = values[index - 1], values[index]
            gap = second.timestamp_s - first.timestamp_s
            if gap > self.config.max_pose_gap_s:
                raise RuntimeError(f"SLAM pose interpolation gap {gap:.6f}s exceeds limit")
            weight = ((timestamp_s - first.timestamp_s) / gap) if gap > 1.0e-12 else 0.0
            extrapolation = 0.0
        covariance = max(value for value in (
            first.position_covariance_trace, second.position_covariance_trace) if value is not None
        ) if any(value is not None for value in (
            first.position_covariance_trace, second.position_covariance_trace)) else None
        if covariance is not None and covariance > self.config.max_position_covariance_trace:
            raise RuntimeError(f"SLAM covariance trace {covariance:.6f} exceeds limit")
        position = (1.0 - weight) * first.position_world_m + weight * second.position_world_m
        quat = _slerp(first.quat_world_base_wxyz, second.quat_world_base_wxyz, weight)
        return position, quat, {
            "pose_t0_s": first.timestamp_s, "pose_t1_s": second.timestamp_s,
            "interpolation_weight": float(weight), "extrapolation_s": float(extrapolation),
            "position_covariance_trace": covariance,
        }

    def ingest_pointcloud(self, points_sensor_m: np.ndarray, timestamp_s: float,
                          *, receive_timestamp_s: float | None = None,
                          labels: list[str] | None = None) -> tuple[RadarScan, dict[str, Any]]:
        points = np.asarray(points_sensor_m, dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(points).all():
            raise ValueError("point cloud contains non-finite coordinates")
        corrected_timestamp = float(timestamp_s) + self.config.sensor_time_offset_s
        try:
            base_position, base_quat, sync = self._pose_at(corrected_timestamp)
        except RuntimeError:
            self.rejected_scans += 1
            raise
        world_R_base = _quat_matrix(base_quat)
        base_R_sensor = self.base_T_sensor[:3, :3]
        base_t_sensor = self.base_T_sensor[:3, 3]
        points_base = points @ base_R_sensor.T + base_t_sensor[None, :]
        points_world = points_base @ world_R_base.T + base_position[None, :]
        sensor_origin_world = world_R_base @ base_t_sensor + base_position
        origins = np.repeat(sensor_origin_world[None, :], len(points_world), axis=0)
        latency = (None if receive_timestamp_s is None else
                   float(receive_timestamp_s) - float(timestamp_s))
        report = {
            "accepted": True, "raw_sensor_timestamp_s": float(timestamp_s),
            "corrected_sensor_timestamp_s": corrected_timestamp,
            "receive_latency_s": latency, "point_count": int(len(points_world)),
            "frames": {"world": self.config.world_frame, "base": self.config.base_frame,
                       "sensor": self.config.sensor_frame},
            "extrinsic_translation_base_m": base_t_sensor.astype(float).tolist(),
            **sync,
        }
        self.accepted_scans += 1
        self.last_report = report
        scan = RadarScan(
            points_world=points_world.astype(np.float32),
            origins_world=origins.astype(np.float32),
            geom_names=(labels if labels is not None else ["real_return"] * len(points_world)),
            timestamp=corrected_timestamp,
        )
        return scan, report

    def summary(self) -> dict[str, Any]:
        return {
            "accepted_scans": self.accepted_scans, "rejected_scans": self.rejected_scans,
            "buffered_poses": len(self.poses), "sync_config": asdict(self.config),
            "base_T_sensor": self.base_T_sensor.astype(float).tolist(),
            "last_report": self.last_report,
            "transport_contract": (
                "push SLAM world_T_base poses and sensor-frame clouds with source timestamps; "
                "output RadarScan is world-frame and directly consumable by update_radar"
            ),
        }


@dataclass(frozen=True)
class Ros2TopicContract:
    """Topic names/types expected by a thin ROS2 transport wrapper."""

    pose_topic: str = "/slam/odometry"
    pose_type: str = "nav_msgs/msg/Odometry"
    pointcloud_topic: str = "/radar/points"
    pointcloud_type: str = "sensor_msgs/msg/PointCloud2"
    qos: str = "sensor_data"
    timestamp_source: str = "message_header"
