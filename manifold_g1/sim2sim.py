"""Sim2sim visualization: run a trained policy in the native MuJoCo viewer.

Training is headless; this is the visualization path. The native viewer gives interactive
3D (drag to orbit, scroll to zoom), scene shadows/reflections, the soft ellipsoid manifold
(translucent, non-colliding), a pelvis trajectory trail, live episode curves drawn by
MuJoCo itself (no matplotlib), and keyboard control.

    python3 -m manifold_g1.sim2sim --resume reports/manifold_g1/ppo_soft/policy.pt
    python3 -m manifold_g1.sim2sim --resume ... --no-viewer --video out.mp4   # headless

Keys: R reset | [ / ] lower / raise the far-ellipsoid half-height | P pause | T toggle trail
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from . import constants as C
from .manifold import EllipsoidManifold
from .ppo import PPO
from .task_env import GoalReachEnv, TaskConfig

TRAIL_LENGTH = 200


class EpisodeFigure:
    """A MuJoCo figure showing one value per episode."""

    def __init__(self, title: str, color=(0.35, 0.6, 1.0)):
        self.fig = mujoco.MjvFigure()
        self.fig.title = title
        self.fig.flg_legend = 0
        self.fig.linewidth = 1.6
        self.fig.linergb[0] = color
        self.series: list[float] = []

    def add(self, value: float) -> None:
        self.series.append(float(value))
        if len(self.series) > 2000:
            self.series = self.series[-2000:]

    def update(self) -> None:
        count = len(self.series)
        self.fig.linepnt[0] = count
        if count:
            self.fig.linedata[0, 0:2 * count:2] = np.arange(count)
            self.fig.linedata[0, 1:2 * count:2] = np.asarray(self.series)
            lo, hi = min(self.series), max(self.series)
            pad = 0.1 * max(hi - lo, 1e-6)
            self.fig.range[1] = (lo - pad, hi + pad)
        self.fig.range[0] = (0, max(count - 1, 1))


def _viewer_figures(return_fig: EpisodeFigure, height_fig: EpisodeFigure):
    return [
        (mujoco.MjrRect(10, 10, 320, 150), return_fig.fig),
        (mujoco.MjrRect(10, 170, 320, 150), height_fig.fig),
    ]


def _update_trail(viewer, pelvis_trail: deque, color=(1.0, 0.55, 0.1, 0.8)) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0
    points = list(pelvis_trail)
    for i in range(len(points) - 1):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012,
            np.asarray(points[i], dtype=np.float64), np.asarray(points[i + 1], dtype=np.float64),
        )
        geom.rgba[:] = color
        scene.ngeom += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize a trained policy in MuJoCo (sim2sim)")
    parser.add_argument("--resume", type=Path, default=Path("reports/manifold_g1/ppo_soft/policy.pt"))
    parser.add_argument("--geo-policy", type=Path, default=None,
                        help="pure-geometry policy (task-space obs); executed here by SONIC")
    parser.add_argument("--manifold", choices=("single", "tunnel"), default="single")
    parser.add_argument("--episodes", type=int, default=0, help="0 = run until the window is closed")
    parser.add_argument("--semi-x", type=float, default=2.6, help="half-length of a single ellipsoid")
    parser.add_argument("--length", type=float, default=4.5)
    parser.add_argument("--semi-y", type=float, default=1.1)
    parser.add_argument("--entry-height", type=float, default=1.3)
    parser.add_argument("--tunnel-height", type=float, default=0.75)
    parser.add_argument("--primitives", type=int, default=3)
    parser.add_argument("--tilt-deg", type=float, default=0.0)
    parser.add_argument("--random-tunnels", default=None,
                        help="sample the far half-height per episode, e.g. 0.65,1.1")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    parser.add_argument("--no-viewer", action="store_true", help="headless run (use with --video)")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument("--view-every", type=int, default=2, help="sync the viewer every N control ticks")
    parser.add_argument("--trail", action="store_true", default=True)
    parser.add_argument("--no-trail", dest="trail", action="store_false")
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--fast", dest="realtime", action="store_false", help="run as fast as possible")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    tunnel_range = tuple(float(v) for v in args.random_tunnels.split(",")) if args.random_tunnels else None

    def tunnel_height() -> float:
        return float(rng.uniform(*tunnel_range)) if tunnel_range else args.tunnel_height

    def manifold(h: float) -> EllipsoidManifold:
        if args.manifold == "single":
            return EllipsoidManifold.single(semi_x=args.semi_x, semi_y=args.semi_y,
                                            semi_z=h, tilt_deg=args.tilt_deg)
        return EllipsoidManifold.tunnel(length=args.length, semi_y=args.semi_y,
                                        entry_semi_z=args.entry_height, tunnel_semi_z=h,
                                        count=args.primitives, tilt_deg=args.tilt_deg)

    env = GoalReachEnv(manifold(tunnel_height()), TaskConfig())
    obs, _ = env.reset(tunnel_semi_z=tunnel_height())
    policy_path = args.geo_policy or args.resume
    probe = env.reduced_obs() if args.geo_policy is not None else obs
    ppo = PPO(int(probe.shape[0]), env.action_dim)
    ppo.load(policy_path)
    print(f"loaded {policy_path}  obs_dim={probe.shape[0]}  action_dim={env.action_dim}"
          f"{'  (pure-geometry policy executed by SONIC)' if args.geo_policy is not None else ''}")

    state = {"reset": False, "pause": False, "trail": args.trail, "tunnel_delta": 0.0}

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if key == "r":
            state["reset"] = True
        elif key == "p":
            state["pause"] = not state["pause"]
        elif key == "t":
            state["trail"] = not state["trail"]
        elif key == "[":
            state["tunnel_delta"] = -0.05
        elif key == "]":
            state["tunnel_delta"] = +0.05

    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(env.env.model, env.env.data, key_callback=key_callback,
                                              show_left_ui=True, show_right_ui=False)
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.pelvis_id
        viewer.cam.distance = 3.4
        viewer.cam.azimuth = 150.0
        viewer.cam.elevation = -14.0

    renderer = None
    writer = None
    if args.video is not None:
        renderer = mujoco.Renderer(env.env.model, height=720, width=1280)

    return_fig = EpisodeFigure("episode return", color=(0.35, 0.6, 1.0))
    height_fig = EpisodeFigure("pelvis z (blue) vs tunnel half-height (orange)", color=(0.35, 0.6, 1.0))
    height_fig.fig.linergb[1] = (1.0, 0.6, 0.2)
    height_fig_second: list[float] = []
    pelvis_trail: deque = deque(maxlen=TRAIL_LENGTH)

    tick_index = 0
    video_due = 0.0

    def on_tick() -> None:
        nonlocal tick_index, video_due
        tick_index += 1
        pelvis_trail.append(env.env.data.xpos[env.pelvis_id].copy())
        if viewer is not None:
            if state["trail"]:
                _update_trail(viewer, pelvis_trail)
            else:
                viewer.user_scn.ngeom = 0
            if tick_index % max(1, args.view_every) == 0:
                viewer.sync()
        if renderer is not None and tick_index >= video_due:
            video_due = tick_index + max(1, int(round(50.0 / max(args.video_fps, 1.0))))
            renderer.update_scene(env.env.data, camera=viewer.cam if viewer is not None else -1)
            _write_video_frame(renderer.render())

    def _write_video_frame(frame: np.ndarray) -> None:
        nonlocal writer
        import cv2
        if writer is None:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.video_fps, (frame.shape[1], frame.shape[0]))
        writer.write(np.ascontiguousarray(frame[:, :, ::-1]))

    episodes_done = 0
    outcomes: list[str] = []
    wall_clock = time.perf_counter()
    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            if state["pause"]:
                if viewer is not None:
                    viewer.sync()
                time.sleep(0.02)
                continue
            if state["tunnel_delta"]:
                height = float(np.clip(env.manifold.primitives[-1].semi[2] + state["tunnel_delta"], 0.5, 1.4))
                state["tunnel_delta"] = 0.0
                env.set_manifold(manifold(height))
                state["reset"] = True
            if state["reset"]:
                state["reset"] = False
                pelvis_trail.clear()
                obs, _ = env.reset(tunnel_semi_z=tunnel_height())
                continue

            policy_obs = env.reduced_obs() if args.geo_policy is not None else obs
            action, _, _ = ppo.act(policy_obs, deterministic=not args.stochastic)
            obs, _, done, truncated, info = env.step(action, tick_callback=on_tick)
            if viewer is not None:
                viewer.set_texts([(None, None,
                                   f"episode {episodes_done}  ({info['outcome']})  return {info['episode_return']:+.1f}",
                                   f"pelvis z {info['base_z']:.2f}  tunnel {info['tunnel_semi_z']:.2f}  "
                                   f"r {info['manifold_radius']:.2f}  spine {info['spine_alignment']:+.2f}  "
                                   f"dist {info['distance']:.2f}")])
                return_fig.update()
                height_fig.update()
                viewer.set_figures(_viewer_figures(return_fig, height_fig))
            if done or truncated:
                episodes_done += 1
                outcomes.append(info["outcome"])
                return_fig.add(info["episode_return"])
                height_fig.add(info["base_z"])
                height_fig_second.append(info["tunnel_semi_z"])
                height_fig.fig.linepnt[1] = len(height_fig_second)
                n2 = len(height_fig_second)
                height_fig.fig.linedata[1, 0:2 * n2:2] = np.arange(n2)
                height_fig.fig.linedata[1, 1:2 * n2:2] = np.asarray(height_fig_second)
                print(f"episode {episodes_done:3d}  {info['outcome']:>15s}  steps {info['episode_step']:3d}  "
                      f"return {info['episode_return']:+7.2f}  pelvis_z {info['base_z']:.3f}  "
                      f"tunnel {info['tunnel_semi_z']:.2f}  r_max {info['manifold_radius_max']:.2f}",
                      flush=True)
                if args.episodes and episodes_done >= args.episodes:
                    break
                pelvis_trail.clear()
                obs, _ = env.reset(tunnel_semi_z=tunnel_height())

            if args.realtime:
                wall_clock += C.CONTROL_DT * 5
                delay = wall_clock - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    wall_clock = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
            print(f"[video] wrote {args.video}")
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()

    print(json.dumps({"episodes": episodes_done, "outcomes": outcomes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
