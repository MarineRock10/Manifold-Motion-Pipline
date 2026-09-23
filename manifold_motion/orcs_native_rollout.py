"""Headless native ORCS checkpoint rollout for a portable A/B video artifact.

Run this from an ORCS Python 3.11 environment.  It reuses ORCS's own task
registration, checkpoint loader, observation groups and policy runner, but
replaces the interactive viewer loop with a bounded rollout so a remote WSL
session cannot wait forever for a GUI window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mediapy as media
import numpy as np
import torch


def _focus_offscreen_camera(render_env) -> None:
    """Center the camera on G1 instead of the terrain-world origin."""
    renderer = getattr(render_env, "_offline_renderer", None)
    scene = getattr(render_env, "scene", None)
    if renderer is None or scene is None or not getattr(scene, "entities", None):
        return
    try:
        import mujoco

        camera = renderer._cam
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE.value
        camera.trackbodyid = -1
        camera.distance = 3.2
        camera.elevation = -12.0
        camera.azimuth = 135.0
    except (AttributeError, StopIteration, TypeError, ValueError):
        # Rendering is still useful with mjlab's task camera if a future ORCS
        # release changes the private scene/renderer shape.
        return


def _update_camera_lookat(render_env) -> None:
    renderer = getattr(render_env, "_offline_renderer", None)
    sim = getattr(render_env, "sim", None)
    if renderer is None or sim is None:
        return
    try:
        root = sim.data.qpos[0, :3].detach().cpu().numpy()
        renderer._cam.lookat[:] = root
    except (AttributeError, IndexError, TypeError, ValueError):
        return


def _headless_run(viewer) -> None:
    env = viewer.env
    # ``run_play`` exposes an RslRlVecEnvWrapper, which intentionally omits
    # ``render``.  Its ``unwrapped`` ManagerBasedRlEnv retains the renderer
    # used by mjlab's VideoRecorder.
    render_env = getattr(env, "unwrapped", env)
    _focus_offscreen_camera(render_env)
    observations = env.get_observations()
    frames: list[np.ndarray] = []
    steps = int(getattr(viewer, "_orcs_rollout_steps", 200))
    for _ in range(steps):
        with torch.inference_mode():
            actions = viewer.policy(observations)
        step_result = env.step(actions)
        # mjlab's VecEnv wrapper follows the classic four-return API, while
        # newer Gymnasium wrappers return five values.  Keep the rollout
        # portable across both without changing ORCS itself.
        if len(step_result) == 5:
            observations, _, terminated, truncated, _ = step_result
        elif len(step_result) == 4:
            observations, _, done, _ = step_result
            terminated = done
            truncated = torch.zeros_like(done, dtype=torch.bool)
        else:
            raise RuntimeError(f"unexpected env.step return length: {len(step_result)}")
        _update_camera_lookat(render_env)
        frame = render_env.render()
        if frame is not None:
            frame = np.asarray(frame)
            if frame.ndim == 4:
                frame = frame[0]
            frames.append(frame.astype(np.uint8, copy=False))
        if bool(terminated[0]) or bool(truncated[0]):
            observations = env.get_observations()
    if not frames:
        raise RuntimeError("ORCS environment returned no RGB frames; use --video support")
    output = Path(viewer._orcs_rollout_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(output), frames, fps=50)
    print(f"ORCS native rollout: {output} ({len(frames)} frames)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Orcs-PerLoco-Grail-AdaptSonic")
    parser.add_argument("--agent", choices=("release", "initial"), default="release")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    import mjlab  # noqa: F401  # registration must happen before ORCS task import
    from mjlab.scripts import play

    # ORCS widens PlayConfig/run_play at import time. The release path resolves
    # and verifies the official checkpoint; initial constructs the same frozen
    # base but leaves the LoRA residual at zero.
    original_native = play.NativeMujocoViewer.run
    original_viser = play.ViserPlayViewer.run

    def run(self):
        self._orcs_rollout_steps = args.steps
        self._orcs_rollout_output = str(args.out)
        _headless_run(self)

    play.NativeMujocoViewer.run = run
    play.ViserPlayViewer.run = run
    try:
        cfg = play.PlayConfig(
            agent=args.agent,
            device=args.device,
            viewer="native",
            video=True,
            video_length=args.steps,
            num_envs=1,
        )
        play.run_play(args.task, cfg)
    finally:
        play.NativeMujocoViewer.run = original_native
        play.ViserPlayViewer.run = original_viser
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
