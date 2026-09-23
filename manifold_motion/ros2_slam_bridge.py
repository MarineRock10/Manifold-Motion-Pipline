"""Optional ROS2 transport for :mod:`manifold_motion.real_slam`.

ROS2 is intentionally an optional runtime dependency. Importing this module in the existing
MuJoCo-only environment is safe; launching the node gives the real SLAM/radar path a strict
timestamp/frame/extrinsic contract and forwards accepted scans to a caller callback.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from .real_slam import Ros2TopicContract, TimeSynchronizedSlamAdapter


def _stamp(message) -> float:
    return float(message.header.stamp.sec) + 1.0e-9 * float(message.header.stamp.nanosec)


class Ros2SlamIngress:
    """ROS2 node wiring Odometry + PointCloud2 into a world-frame RadarScan callback."""

    def __init__(self, adapter: TimeSynchronizedSlamAdapter,
                 on_scan: Callable, topics: Ros2TopicContract = Ros2TopicContract()):
        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import PointCloud2
            from rclpy.node import Node
            from sensor_msgs_py import point_cloud2
        except ImportError as error:  # pragma: no cover - exercised only on ROS2 hosts
            raise RuntimeError(
                "ROS2 ingress requires rclpy, nav_msgs, sensor_msgs and sensor_msgs_py"
            ) from error
        self.rclpy = rclpy
        self.point_cloud2 = point_cloud2
        self.adapter = adapter
        self.on_scan = on_scan
        self.node = Node("manifold_motion_slam_ingress")
        self.node.create_subscription(Odometry, topics.pose_topic, self._pose_callback, 20)
        self.node.create_subscription(PointCloud2, topics.pointcloud_topic,
                                       self._cloud_callback, 10)
        self.last_error: str | None = None
        self.accepted_scans = 0

    def _pose_callback(self, message) -> None:  # pragma: no cover - ROS2 transport
        pose = message.pose.pose
        covariance = float(message.pose.covariance[0] + message.pose.covariance[7]
                           + message.pose.covariance[14])
        self.adapter.push_pose(
            _stamp(message),
            np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64),
            np.array([pose.orientation.w, pose.orientation.x,
                      pose.orientation.y, pose.orientation.z], dtype=np.float64),
            covariance,
        )

    def _cloud_callback(self, message) -> None:  # pragma: no cover - ROS2 transport
        try:
            rows = self.point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True)
            points = np.asarray(list(rows), dtype=np.float64).reshape(-1, 3)
            scan, _ = self.adapter.ingest_pointcloud(
                points, _stamp(message), receive_timestamp_s=self.node.get_clock().now().nanoseconds * 1e-9)
            self.on_scan(scan)
            self.accepted_scans += 1
        except (RuntimeError, ValueError) as error:
            self.last_error = f"{type(error).__name__}: {error}"

    def spin(self) -> None:  # pragma: no cover - ROS2 transport
        self.rclpy.spin(self.node)


def main() -> int:  # pragma: no cover - ROS2 transport
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pose-topic", default="/slam/odometry")
    parser.add_argument("--pointcloud-topic", default="/radar/points")
    args = parser.parse_args()
    adapter = TimeSynchronizedSlamAdapter.from_json(args.config)
    try:
        import rclpy
    except ImportError as error:
        raise SystemExit("ROS2 is not installed in this WSL environment") from error
    rclpy.init()
    ingress = Ros2SlamIngress(
        adapter, lambda scan: None,
        Ros2TopicContract(pose_topic=args.pose_topic, pointcloud_topic=args.pointcloud_topic),
    )
    ingress.spin()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
