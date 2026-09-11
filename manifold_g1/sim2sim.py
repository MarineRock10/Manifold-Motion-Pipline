"""Sim2sim visualization: run a trained policy in the native MuJoCo viewer.

Training is headless; this is the visualization path. The native viewer gives interactive
3D (drag to orbit, scroll to zoom), scene shadows/reflections, the ellipsoid manifold, a
pelvis trajectory trail, live episode curves drawn by MuJoCo itself (no matplotlib), and
keyboard control.

    python3 -m manifold_g1.sim2sim --resume reports/manifold_g1/ppo_l3/policy.pt
    python3 -m manifold_g1.sim2sim --resume ... --no-viewer --video out.mp4   # headless

Keys: R reset | [ / ] lower / raise the goal-end ceiling | P pause | T toggle trail
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
from .manifold import ManifoldSpec, apply_profile
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
        self.baseline: float | None = None

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
            lo = min(min(self.series), self.baseline if self.baseline is not None else min(self.series))
            hi = max(max(self.series), self.baseline if self.baseline is not None else max(self.series))
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
    parser.add_argument("--resume", type=Path, default=Path("reports/manifold_g1/ppo_l3/policy.pt"))
    parser.add_argument("--episodes", type=int, default=0, help="0 = run until the window is closed")
    parser.add_argument("--length", type=float, default=6.0)
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--height-start", type=float, default=1.5)
    parser.add_argument("--height-goal", type=float, default=1.0)
    parser.add_argument("--random-heights", default=None,
                        help="sample the goal-end ceiling per episode, e.g. 0.95,1.25")
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
    height_range = tuple(float(v) for v in args.random_heights.split(",")) if args.random_heights else None

    def height_goal() -> float:
        return float(rng.uniform(*height_range)) if height_range else args.height_goal

    spec = ManifoldSpec(length=args.length, width=args.width,
                        height_start=args.height_start, height_goal=args.height_goal)
    env = GoalReachEnv(spec, TaskConfig())
    obs, _ = env.reset(height_start=args.height_start, height_goal=args.height_goal)
    ppo = PPO(int(obs.shape[0]), env.action_dim)
    ppo.load(args.resume)
    print(f"loaded {args.resume}  obs_dim={obs.shape[0]}  action_dim={env.action_dim}")

    state = {"reset": False, "pause": False, "trail": args.trail, "height_delta": 0.0}

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if key == "r":
            state["reset"] = True
        elif key == "p":
            state["pause"] = not state["pause"]
        elif key == "t":
            state["trail"] = not state["trail"]
        elif key == "[":
            state["height_delta"] = -0.05
        elif key == "]":
            state["height_delta"] = +0.05

    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(env.env.model, env.env.data, key_callback=key_callback,
                                              show_left_ui=True, show_right_ui=False)
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        pelvis_id = mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id
        viewer.cam.distance = 3.4
        viewer.cam.azimuth = 150.0
        viewer.cam.elevation = -14.0
        viewer.set_figures(_viewer_figures(EpisodeFigure(""), EpisodeFigure("")))

    renderer = None
    writer = None
    if args.video is not None:
        renderer = mujoco.Renderer(env.env.model, height=720, width=1280)

    return_fig = EpisodeFigure("episode return", color=(0.35, 0.6, 1.0))
    height_fig = EpisodeFigure("pelvis z (blue) vs ceiling (orange)", color=(0.35, 0.6, 1.0))
    height_fig.fig.linergb[1] = (1.0, 0.6, 0.2)
    height_fig.fig.linepnt[1] = 0
    height_fig_second: list[float] = []
    pelvis_trail: deque = deque(maxlen=TRAIL_LENGTH)

    tick_index = 0
    video_due = 0.0

    def on_tick() -> None:
        nonlocal tick_index, video_due
        tick_index += 1
        pelvis = env.env.data.xpos[mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
        pelvis_trail.append(pelvis.copy())
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
            frame = renderer.render()
            _write_video_frame(frame)

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
    obs, info = env.reset(height_start=args.height_start, height_goal=args.height_goal)
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
            if state["height_delta"]:
                spec.height_goal = float(np.clip(spec.height_goal + state["height_delta"], 0.85, spec.height_start))
                state["height_delta"] = 0.0
                apply_profile(env.env.model, spec, env.env.data)
                state["reset"] = True
            if state["reset"]:
                state["reset"] = False
                pelvis_trail.clear()
                obs, info = env.reset(height_start=args.height_start, height_goal=height_goal())
                continue

            action, _, _ = ppo.act(obs, deterministic=not args.stochastic)
            obs, _, done, truncated, info = env.step(action, tick_callback=on_tick)
            if viewer is not None:
                viewer.set_texts([(None, None,
                                   f"episode {episodes_done}  ({info['outcome']})  return {info['episode_return']:+.1f}",
                                   f"pelvis z {info['base_z']:.2f} m   ceiling {info['local_ceiling']:.2f} m   "
                                   f"dist {info['distance']:.2f} m")])
                return_fig.update()
                height_fig.update()
                viewer.set_figures(_viewer_figures(return_fig, height_fig))
            if done or truncated:
                episodes_done += 1
                outcomes.append(info["outcome"])
                return_fig.add(info["episode_return"])
                height_fig.add(info["base_z"])
                height_fig_second.append(info["local_ceiling"])
                height_fig.fig.linepnt[1] = len(height_fig_second)
                n2 = len(height_fig_second)
                height_fig.fig.linedata[1, 0:2 * n2:2] = np.arange(n2)
                height_fig.fig.linedata[1, 1:2 * n2:2] = np.asarray(height_fig_second)
                print(f"episode {episodes_done:3d}  {info['outcome']:>13s}  "
                      f"return {info['episode_return']:+7.2f}  pelvis_z {info['base_z']:.3f}  "
                      f"ceiling {info['local_ceiling']:.2f}", flush=True)
                if args.episodes and episodes_done >= args.episodes:
                    break
                pelvis_trail.clear()
                obs, info = env.reset(height_start=args.height_start, height_goal=height_goal())

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

    summary = {"episodes": episodes_done, "outcomes": outcomes}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
