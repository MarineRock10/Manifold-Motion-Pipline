"""Render an evaluated Stage-2 sample to MP4 without relying on a desktop OpenGL surface.

This is the reliable fallback for Windows Remote Desktop/WSLg sessions that show the native
MuJoCo window as a white ``COPY MODE`` surface.  It uses MuJoCo's EGL renderer and OpenCV's
software video writer, so the resulting MP4 can be opened on Windows even when interactive
GLFW composition is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import constants as C
from .corridor import _resample_corridor
from .env import G1FlatEnv


def _local_to_world(points: np.ndarray, origin_pos: np.ndarray, origin_quat: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ C.quat_to_matrix(origin_quat).T + origin_pos


def _add_sphere(scene: mujoco.MjvScene, position: np.ndarray, radius: float,
                color: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0.0, 0.0]), position,
                        np.eye(3).flatten(), np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def _add_ellipsoid(scene: mujoco.MjvScene, center: np.ndarray, semi: np.ndarray,
                   yaw: float, color: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                         [np.sin(yaw), np.cos(yaw), 0.0],
                         [0.0, 0.0, 1.0]])
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        np.asarray(semi, dtype=np.float64), center, rotation.flatten(),
                        np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def render(args: argparse.Namespace) -> int:
    with np.load(args.sample) as archive:
        sample = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.executed) as archive:
        execution = {key: np.asarray(archive[key]) for key in archive.files}
    for key in ("q_exec", "base_pos", "base_quat"):
        if key not in execution:
            raise ValueError(f"executed file is missing {key}")

    env = G1FlatEnv(args.scene or C.FLAT_SCENE)
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    camera.distance, camera.azimuth, camera.elevation = args.distance, 125.0, -12.0
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)

    n = len(execution["q_exec"])
    origin_pos, origin_quat = execution["base_pos"][0], execution["base_quat"][0]
    generated = sample.get("generated_ref")
    expected = sample.get("expected_ref", generated)
    model_path = _local_to_world(generated[:, 29:32], origin_pos, origin_quat) if generated is not None else None
    expected_path = _local_to_world(expected[:, 29:32], origin_pos, origin_quat) if expected is not None else None
    executed_path = execution["base_pos"]
    corridor = sample.get("condition_corridor")
    corridor_50 = _resample_corridor(corridor, n) if corridor is not None else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    try:
        for tick in range(n):
            env.data.qpos[:3] = execution["base_pos"][tick]
            env.data.qpos[3:7] = execution["base_quat"][tick]
            env.data.qpos[env.body_qadr] = execution["q_exec"][tick][C.ISAACLAB_TO_MUJOCO]
            env.data.qpos[env.hand_qadr] = 0.0
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, camera=camera, scene_option=option)
            scene = renderer.scene
            if model_path is not None:
                for point in model_path[::2]: _add_sphere(scene, point, 0.014, (0.2, 0.75, 1.0, 0.60))
            if expected_path is not None:
                for point in expected_path[::2]: _add_sphere(scene, point, 0.012, (0.25, 1.0, 0.35, 0.45))
            for point in executed_path[max(0, tick - 250):tick:4]:
                _add_sphere(scene, point, 0.016, (1.0, 0.55, 0.10, 0.75))
            if corridor_50 is not None:
                element = corridor_50[tick]
                center = _local_to_world(element[None, :3], origin_pos, origin_quat)[0]
                _add_ellipsoid(scene, center, element[3:6], float(element[6]), (0.75, 0.25, 1.0, 0.16))
            frame = Image.fromarray(np.asarray(renderer.render()), mode="RGB")
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, args.width, 48), fill=(15, 15, 15))
            draw.text((20, 14), f"Stage-2 execution | tick {tick}/{n-1} | orange=exec cyan=model green=reference",
                      fill=(230, 230, 230))
            frames.append(frame)
    finally:
        renderer.close()
    if not frames:
        raise RuntimeError("renderer produced no frames")
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(round(1000 / args.fps)), loop=0)
    print(json.dumps({"out": str(args.out), "frames": n, "fps": args.fps,
                      "duration_s": round(n / args.fps, 3)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an evaluated Stage-2 sample to animated GIF")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--executed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help=".gif output path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--distance", type=float, default=3.0)
    parser.add_argument("--scene", type=Path, default=None,
                        help="optional physical MuJoCo scene used during validation")
    args = parser.parse_args()
    if min(args.width, args.height, args.fps) <= 0:
        parser.error("width, height, and fps must be positive")
    return render(args)


if __name__ == "__main__":
    raise SystemExit(main())
