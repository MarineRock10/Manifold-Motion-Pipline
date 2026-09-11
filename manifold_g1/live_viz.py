"""Live 3D training dashboard.

Top row (both MuJoCo renders, 3D):
  left   -- the manifold: corridor + translucent ellipsoid chain, robot hidden
  right  -- the G1 training inside the manifold, tracking camera on the pelvis
Bottom row: metric curves updated as training runs
  episode return | rolling success rate | |vx| command | manifold compliance radius

Enable from the training script:

    python3 -m manifold_g1.train --viz                     # live window
    python3 -m manifold_g1.train --viz-video reports/...mp4  # headless recording
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .manifold import ManifoldSpec


def _rolling(values: np.ndarray | list, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return np.array([])
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


class LiveDashboard:
    def __init__(self, env, spec: ManifoldSpec, show: bool = False,
                 video_path: Path | None = None, min_interval: float = 0.2,
                 fps: float = 2.0, video_scale: int = 2):
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mujoco

        self.plt = plt
        self.env = env
        self.mj_env = env.env
        self.spec = spec
        self.show = show
        self.video_path = Path(video_path) if video_path else None
        self.min_interval = min_interval
        self.video_interval = 1.0 / max(fps, 0.1)
        self.video_scale = max(1, video_scale)

        self.fig = plt.figure(figsize=(15.0, 8.6))
        grid = self.fig.add_gridspec(2, 10, height_ratios=[2.3, 1.0], hspace=0.30, wspace=0.45)
        self.ax_manifold = self.fig.add_subplot(grid[0, 0:5])
        self.ax_scene = self.fig.add_subplot(grid[0, 5:10])
        self.ax_return = self.fig.add_subplot(grid[1, 0:2])
        self.ax_success = self.fig.add_subplot(grid[1, 2:4])
        self.ax_cmd = self.fig.add_subplot(grid[1, 4:6])
        self.ax_compliance = self.fig.add_subplot(grid[1, 6:8])
        self.ax_height = self.fig.add_subplot(grid[1, 8:10])
        self.ax_manifold.set_title("Manifold: ellipsoid chain (robot hidden)")
        self.ax_scene.set_title("G1 training inside the manifold")
        for axis in (self.ax_manifold, self.ax_scene):
            axis.axis("off")

        self.renderer = mujoco.Renderer(self.mj_env.model, height=360, width=560)
        self._blank = np.zeros((360, 560, 3), dtype=np.uint8)
        self._img_manifold = self.ax_manifold.imshow(self._blank)
        self._img_scene = self.ax_scene.imshow(self._blank)

        self.option_manifold = mujoco.MjvOption()
        self.option_manifold.geomgroup[:] = 0
        self.option_manifold.geomgroup[1] = 1          # manifold geoms only (robot is group 0)

        self.cam_manifold = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam_manifold)
        self.cam_manifold.lookat[:] = (0.0, 0.0, 0.5 * spec.height_start)
        self.cam_manifold.distance = max(4.0, 0.85 * spec.length)
        self.cam_manifold.azimuth = 160.0
        self.cam_manifold.elevation = -12.0

        self.cam_scene = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam_scene)
        self.cam_scene.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam_scene.trackbodyid = mujoco.mj_name2id(self.mj_env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.cam_scene.distance = 2.8
        self.cam_scene.azimuth = 145.0
        self.cam_scene.elevation = -12.0

        self.episode_returns: list[float] = []
        self.success_flags: list[float] = []
        self.cmd_vx: list[float] = []
        self.compliance: list[float] = []
        self.height_goal: list[float] = []
        self.height_pelvis: list[float] = []
        self.height_success: list[bool] = []
        self.step = 0
        self.episodes = 0
        self.last_update = 0.0
        self.last_video = 0.0
        self._writer = None

    # -- data in ------------------------------------------------------------
    def log_step(self, command: np.ndarray, compliance: float) -> None:
        self.cmd_vx.append(abs(float(command[0])))
        self.compliance.append(float(compliance))
        self.step += 1

    def log_episode(self, outcome: str, episode_return: float, height_goal: float | None = None,
                    pelvis_z: float | None = None) -> None:
        self.success_flags.append(1.0 if outcome == "success" else 0.0)
        self.episode_returns.append(float(episode_return))
        self.episodes += 1
        if height_goal is not None and pelvis_z is not None:
            self.height_goal.append(float(height_goal))
            self.height_pelvis.append(float(pelvis_z))
            self.height_success.append(outcome == "success")

    # -- rendering ----------------------------------------------------------
    def maybe_update(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_update < self.min_interval:
            return
        self.last_update = now
        self._render_views()
        self._draw_curves()
        self.fig.canvas.draw()
        if self.show:
            self.plt.pause(0.001)
        if self.video_path is not None and now - self.last_video >= self.video_interval:
            self.last_video = now
            self._write_frame()

    def _render_views(self) -> None:
        data = self.mj_env.data
        self.renderer.update_scene(data, camera=self.cam_manifold, scene_option=self.option_manifold)
        self._img_manifold.set_data(self.renderer.render())
        self.renderer.update_scene(data, camera=self.cam_scene)
        self._img_scene.set_data(self.renderer.render())

    def _draw_curves(self) -> None:
        self._curve(self.ax_return, "Episode return", self.episode_returns, window=10, ylabel="return")
        self._curve(self.ax_success, "Success rate (rolling 20)", self.success_flags, window=20, ylabel="rate")
        self.ax_success.set_ylim(-0.05, 1.05)
        self._curve(self.ax_cmd, "|vx| command (rolling 100)", self.cmd_vx, window=100, ylabel="m/s")
        self._curve(self.ax_compliance, "Manifold ellipse radius (rolling 100)", self.compliance,
                    window=100, ylabel="r", baseline=1.0)
        self._draw_height_panel()
        self.fig.suptitle(
            f"steps {self.step}   episodes {self.episodes}   "
            f"success(last 20) {np.mean(self.success_flags[-20:]):.2f}" if self.success_flags
            else f"steps {self.step}   episodes {self.episodes}",
            fontsize=11,
        )

    def _draw_height_panel(self) -> None:
        """L3 evidence: achieved pelvis height versus the manifold's goal-end ceiling height."""
        axis = self.ax_height
        axis.clear()
        if self.height_goal:
            goals = np.asarray(self.height_goal)
            pelvis = np.asarray(self.height_pelvis)
            success = np.asarray(self.height_success)
            axis.scatter(goals[success], pelvis[success], s=14, color="#2ca02c", label="success")
            axis.scatter(goals[~success], pelvis[~success], s=14, color="#d62728", label="failed")
            if success.sum() >= 5:
                order = np.argsort(goals[success])
                axis.plot(goals[success][order], pelvis[success][order], color="#2ca02c", lw=1, alpha=0.5)
        axis.set_title("Pelvis height vs manifold ceiling (goal end)", fontsize=9)
        axis.set_xlabel("ceiling height [m]", fontsize=8)
        axis.set_ylabel("pelvis z [m]", fontsize=8)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=7)
        axis.legend(fontsize=6)

    def _curve(self, axis, title: str, values: list[float], window: int, ylabel: str,
               baseline: float | None = None) -> None:
        axis.clear()
        if values:
            axis.plot(values, lw=0.8, alpha=0.45, color="#1f77b4")
            smooth = _rolling(values, window)
            if smooth.size:
                axis.plot(np.arange(window - 1, len(values)), smooth, lw=2.0, color="#d62728")
        if baseline is not None:
            axis.axhline(baseline, color="#888888", lw=1, ls="--")
        axis.set_title(title, fontsize=9)
        axis.set_ylabel(ylabel, fontsize=8)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=7)

    def _write_frame(self) -> None:
        import cv2
        frame = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]
        if self.video_scale > 1:
            frame = cv2.resize(frame, None, fx=1.0 / self.video_scale, fy=1.0 / self.video_scale)
        if self._writer is None:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(str(self.video_path),
                                           cv2.VideoWriter_fourcc(*"mp4v"),
                                           1.0 / self.video_interval, (width, height))
        self._writer.write(np.ascontiguousarray(frame[:, :, ::-1]))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            print(f"[viz] wrote {self.video_path}")
        if self.show:
            self.plt.close(self.fig)
