"""Render a before/after Stage-2 comparison GIF for the same held-out window."""

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


def _load_case(label: str, sample_path: Path, execution_path: Path, summary_path: Path | None) -> Case:
    with np.load(sample_path) as archive:
        sample = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(execution_path) as archive:
        execution = {key: np.asarray(archive[key]) for key in archive.files}
    summary = json.loads(summary_path.read_text()) if summary_path and summary_path.is_file() else {}
    if isinstance(summary.get("selected"), dict):
        summary = summary["selected"]
    # A semantic-route report carries router evidence at the top level and the
    # SONIC hard-gate metrics under ``execution``.  The renderer's status line
    # must use the latter; otherwise a routed case looks like zero-progress even
    # when its actual execution passed.
    if isinstance(summary.get("execution"), dict):
        execution_summary = dict(summary["execution"])
        for key in ("routed_primitive", "routed_primitive_id", "router_confidence", "source_primitive"):
            if key in summary:
                execution_summary[key] = summary[key]
        summary = execution_summary
    return Case(label, sample, execution, summary)


def _add_sphere(scene: mujoco.MjvScene, position: np.ndarray, radius: float,
                color: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0.0, 0.0]), position,
                        np.eye(3).flatten(), np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def _render_case(renderer: mujoco.Renderer, camera: mujoco.MjvCamera, option: mujoco.MjvOption,
                 env: G1FlatEnv, case: Case, tick: int) -> Image.Image:
    execution = case.execution
    env.data.qpos[:3] = execution["base_pos"][tick]
    env.data.qpos[3:7] = execution["base_quat"][tick]
    env.data.qpos[env.body_qadr] = execution["q_exec"][tick][C.ISAACLAB_TO_MUJOCO]
    env.data.qpos[env.hand_qadr] = 0.0
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    camera.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    renderer.update_scene(env.data, camera=camera, scene_option=option)
    scene = renderer.scene
    for point in execution["base_pos"][max(0, tick - 250):tick:4]:
        _add_sphere(scene, point, 0.014, (1.0, 0.55, 0.10, 0.75))
    frame = Image.fromarray(np.asarray(renderer.render()), mode="RGB")
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, frame.width, 74), fill=(10, 10, 10))
    summary = case.summary
    accepted = bool(summary.get("accepted", False))
    status = "PASS" if accepted else "diagnostic"
    progress = float(summary.get("motion_progress_ratio", 0.0)) * 100.0
    corridor = float(summary.get("corridor_radius_max", float("nan")))
    tracking = float(summary.get("track_err_mean_rad", float("nan")))
    draw.text((10, 9), case.label, fill=(245, 245, 245))
    draw.text((10, 30), f"{status}  progress {progress:.1f}%", fill=(245, 245, 245))
    draw.text((10, 50), f"track {tracking:.3f} rad  corridor {corridor:.3f}", fill=(245, 245, 245))
    return frame


def render(args: argparse.Namespace) -> int:
    cases = [
        _load_case(args.oracle_label, args.oracle_sample, args.oracle_executed, args.oracle_summary),
        _load_case(args.early_label, args.early_sample, args.early_executed, args.early_summary),
        _load_case(args.trained_label, args.trained_sample, args.trained_executed, args.trained_summary),
    ]
    frame_count = min(len(case.execution["q_exec"]) for case in cases)
    env = G1FlatEnv()
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.distance, camera.azimuth, camera.elevation = 2.5, 125.0, -12.0
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    frames = []
    for tick in range(frame_count):
        panels = [_render_case(renderer, camera, option, env, case, tick) for case in cases]
        canvas = Image.new("RGB", (args.width * 3, args.height), (18, 18, 18))
        for index, panel in enumerate(panels):
            canvas.paste(panel, (index * args.width, 0))
        frames.append(canvas)
    renderer.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(round(1000 / args.fps)), loop=0)
    print(json.dumps({"out": str(args.out), "frames": frame_count, "fps": args.fps,
                      "duration_s": round(frame_count / args.fps, 3),
                      "cases": [case.label for case in cases]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a Stage-2 before/after comparison GIF")
    for prefix, title in (("oracle", "SEED reference"), ("early", "early Flow"), ("trained", "trained model")):
        parser.add_argument(f"--{prefix}-sample", type=Path, required=True, help=f"{title} sample.npz")
        parser.add_argument(f"--{prefix}-executed", type=Path, required=True, help=f"{title} executed.npz")
        parser.add_argument(f"--{prefix}-summary", type=Path, required=True, help=f"{title} summary.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--oracle-label", default="SEED reference", help="label for the first panel")
    parser.add_argument("--early-label", default="Early Flow (10/20 ep)", help="label for the second panel")
    parser.add_argument("--trained-label", default="Trained walk model",
                        help="label for the third panel")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    return render(args)


if __name__ == "__main__":
    raise SystemExit(main())
