"""Render an evidence-first Stage-2 before/after GIF.

Unlike the generic replay viewer, this dashboard deliberately makes the learned difference
visible: each panel contains the same held-out SEED route (green), its generated reference
(cyan), and what SONIC/MuJoCo actually executed (orange).  The lower plan view is in the root
frame, so it remains readable even while the camera tracks the robot.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from . import constants as C
from .env import G1FlatEnv


@dataclass
class Case:
    label: str
    sample: dict[str, np.ndarray]
    execution: dict[str, np.ndarray]
    summary: dict[str, object]
    generated_xy: np.ndarray
    seed_xy: np.ndarray
    executed_xy: np.ndarray


def _load_case(label: str, sample_path: Path, executed_path: Path, summary_path: Path) -> Case:
    with np.load(sample_path) as archive:
        sample = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(executed_path) as archive:
        execution = {key: np.asarray(archive[key]) for key in archive.files}
    summary = json.loads(summary_path.read_text())
    # ``stage2_select`` stores gate metrics under ``selected`` while the standalone validator
    # stores them at the root.  Accept both report forms so the dashboard can show a genuinely
    # selected stochastic candidate rather than falling back to candidate zero.
    if isinstance(summary.get("selected"), dict):
        summary = summary["selected"]
    for key in ("generated_ref", "expected_ref"):
        if key not in sample or sample[key].shape[-1] < 32:
            raise ValueError(f"{sample_path} lacks a valid {key}")
    origin_pos, origin_quat = execution["base_pos"][0], execution["base_quat"][0]
    rotation = C.quat_to_matrix(origin_quat)
    executed_local = (execution["base_pos"] - origin_pos) @ rotation
    return Case(label, sample, execution, summary,
                sample["generated_ref"][:, 29:31], sample["expected_ref"][:, 29:31],
                executed_local[:, :2])


def _add_sphere(scene: mujoco.MjvScene, position: np.ndarray, radius: float,
                color: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0.0, 0.0]), position,
                        np.eye(3).flatten(), np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def _bounds(cases: list[Case]) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate([np.concatenate((case.generated_xy, case.seed_xy, case.executed_xy), axis=0)
                             for case in cases], axis=0)
    lower, upper = points.min(axis=0), points.max(axis=0)
    span = np.maximum(upper - lower, 0.25)
    return lower - 0.16 * span, upper + 0.16 * span


def _draw_plan(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], case: Case,
               lower: np.ndarray, upper: np.ndarray) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=10, fill=(22, 28, 38), outline=(70, 80, 96), width=2)
    draw.text((left + 10, top + 7), "Root-frame route: green=SEED  cyan=reference  orange=executed",
              fill=(225, 230, 235))
    map_top = top + 29
    width, height = right - left - 18, bottom - map_top - 9

    def project(points: np.ndarray) -> list[tuple[int, int]]:
        normalized = (points - lower) / np.maximum(upper - lower, 1e-6)
        return [(int(left + 9 + value[0] * width), int(map_top + height - value[1] * height))
                for value in normalized]

    for path, color, line_width in ((case.seed_xy, (60, 235, 90), 4),
                                    (case.generated_xy, (45, 190, 255), 4),
                                    (case.executed_xy, (255, 155, 38), 5)):
        pixels = project(path)
        if len(pixels) > 1:
            draw.line(pixels, fill=color, width=line_width, joint="curve")
            end = pixels[-1]
            draw.ellipse((end[0] - 5, end[1] - 5, end[0] + 5, end[1] + 5), fill=color)
    start = project(np.zeros((1, 2)))[0]
    draw.ellipse((start[0] - 5, start[1] - 5, start[0] + 5, start[1] + 5), fill=(245, 245, 245))


def _render_robot(renderer: mujoco.Renderer, camera: mujoco.MjvCamera, option: mujoco.MjvOption,
                  env: G1FlatEnv, case: Case, tick: int) -> Image.Image:
    execution = case.execution
    env.data.qpos[:3] = execution["base_pos"][tick]
    env.data.qpos[3:7] = execution["base_quat"][tick]
    env.data.qpos[env.body_qadr] = execution["q_exec"][tick][C.ISAACLAB_TO_MUJOCO]
    env.data.qpos[env.hand_qadr] = 0.0
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    renderer.update_scene(env.data, camera=camera, scene_option=option)
    scene = renderer.scene
    for point in execution["base_pos"][max(0, tick - 250):tick:3]:
        _add_sphere(scene, point, 0.018, (1.0, 0.55, 0.10, 0.85))
    return Image.fromarray(np.asarray(renderer.render()), mode="RGB")


def render(args: argparse.Namespace) -> int:
    cases = [
        _load_case(args.early_label, args.early_sample, args.early_executed, args.early_summary),
        _load_case(args.residual_label, args.residual_sample, args.residual_executed, args.residual_summary),
    ]
    lower, upper = _bounds(cases)
    scene_height, header_height, plan_height = args.scene_height, 82, 178
    panel_height = header_height + scene_height + plan_height
    frame_count = min(len(case.execution["q_exec"]) for case in cases)
    env = G1FlatEnv()
    renderer = mujoco.Renderer(env.model, height=scene_height, width=args.panel_width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    camera.distance, camera.azimuth, camera.elevation = 2.7, 125.0, -12.0
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    frames = []
    try:
        for tick in range(frame_count):
            canvas = Image.new("RGB", (args.panel_width * 2, panel_height), (12, 16, 24))
            for index, case in enumerate(cases):
                x = index * args.panel_width
                panel = Image.new("RGB", (args.panel_width, panel_height), (12, 16, 24))
                robot = _render_robot(renderer, camera, option, env, case, tick)
                panel.paste(robot, (0, header_height))
                draw = ImageDraw.Draw(panel)
                accepted = bool(case.summary.get("accepted", False))
                status, color = ("PASS: selector found executable candidate", (65, 235, 100)) if accepted else \
                    ("REJECTED: hard gate failed", (255, 85, 85))
                progress = 100 * float(case.summary.get("motion_progress_ratio", 0.0))
                tracking = float(case.summary.get("track_err_mean_rad", float("nan")))
                corridor = float(case.summary.get("corridor_radius_max", float("nan")))
                draw.rectangle((0, 0, args.panel_width, header_height), fill=(18, 22, 31))
                draw.text((12, 9), case.label, fill=(245, 245, 245))
                draw.text((12, 30), status, fill=color)
                draw.text((12, 51), f"progress {progress:.1f}%  |  tracking {tracking:.3f} rad  |  corridor {corridor:.3f}",
                          fill=(232, 232, 232))
                _draw_plan(draw, (9, header_height + scene_height + 5, args.panel_width - 9, panel_height - 7),
                           case, lower, upper)
                canvas.paste(panel, (x, 0))
            frames.append(canvas)
    finally:
        renderer.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=int(round(1000 / args.fps)), loop=0)
    print(json.dumps({"out": str(args.out), "frames": frame_count, "fps": args.fps,
                      "same_source_index": int(cases[0].sample["source_index"]) == int(cases[1].sample["source_index"])}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="render a Stage-2 effect dashboard GIF")
    for name in ("early", "residual"):
        parser.add_argument(f"--{name}-sample", type=Path, required=True)
        parser.add_argument(f"--{name}-executed", type=Path, required=True)
        parser.add_argument(f"--{name}-summary", type=Path, required=True)
    parser.add_argument("--early-label", default="Early latent Flow")
    parser.add_argument("--residual-label", default="Residual Flow + selector")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--scene-height", type=int, default=300)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    return render(args)


if __name__ == "__main__":
    raise SystemExit(main())
