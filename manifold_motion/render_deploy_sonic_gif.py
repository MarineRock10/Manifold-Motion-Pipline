"""Render deploy perception + main-branch SONIC execution in one headless GIF.

The deploy branch supplies only the P1 perception condition: a 3-D probabilistic
grid, A* route and ellipsoid corridor.  The downstream chain remains the main
branch semantic primitive router, Stage-2 generator, hard gate, frozen SONIC
controller and MuJoCo.  The actual robot panel is *not* kinematic: it replays
the selected reference and renders the logged ``q_exec``/``base_pos`` state.
This makes a failed route-following model visible instead of hiding it with a
manually translated root pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from . import constants as C
from .corridor import _resample_corridor
from .env import G1FlatEnv
from .seed_replay import ReplayConfig, SeedReplayRunner
from .stage2_validate import validate_trajectory
from .stage2_render import _add_ellipsoid, _add_sphere


BG = (24, 30, 40)
TEXT = (232, 238, 245)
GRID = (76, 91, 108)
PLANNED = (255, 216, 62)
EXEC = (255, 112, 52)
SCOUT = (60, 190, 230)
ROBOT = (65, 235, 115)
GOAL = (80, 135, 255)
TUBE = (50, 238, 214)
OCCUPIED = (191, 58, 58)
UNKNOWN = (101, 111, 124)


def _text(draw: ImageDraw.ImageDraw, value: str, xy: tuple[int, int], fill=TEXT) -> None:
    draw.text(xy, value, fill=fill)


def _load_condition(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _load_or_execute(args: argparse.Namespace, sample: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    """Run the same SONIC/MuJoCo replay as main's validator unless explicitly reused."""
    if args.reuse_executed and args.executed is not None and args.executed.exists():
        with np.load(args.executed) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        summary = {}
        if args.summary is not None and args.summary.exists():
            summary = json.loads(args.summary.read_text())
        summary.setdefault("source", "reused_stage2_physical_execution")
        return data, summary

    trajectory = sample.get("generated_ref")
    if trajectory is None:
        candidates = sample.get("generated_ref_candidates")
        if candidates is None:
            raise ValueError("sample has neither generated_ref nor generated_ref_candidates")
        trajectory = candidates[args.candidate_index]
    corridor = sample.get("condition_corridor")
    stratum = str(np.asarray(sample.get("primitive_name", "generated")))
    runner = SeedReplayRunner(args.scene)
    data, summary = validate_trajectory(
        np.asarray(trajectory), source=args.sample, source_hz=args.source_hz,
        config=ReplayConfig(), corridor=corridor,
        max_corridor_radius=args.max_corridor_radius,
        stratum=stratum, runner=runner,
    )
    summary.update({"source": "stage2_physical_execution", "sample": str(args.sample),
                    "scene": str(args.scene), "candidate_index": args.candidate_index})
    if args.executed is not None:
        args.executed.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.executed, **data)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    return data, summary


def _grid_image(condition: dict[str, np.ndarray], execution: dict[str, np.ndarray], tick: int,
                width: int, height: int) -> Image.Image:
    """Render the final deploy voxel probability map with planned and executed paths."""
    panel = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(panel)
    _text(draw, "deploy 3-D probabilistic grid / route", (12, 10))
    probability = np.asarray(condition["map_probability"], dtype=np.float32)
    origin = np.asarray(condition.get("map_origin_world_xyz", [-3.76, -6.16, -0.32]), dtype=np.float64)
    resolution = float(np.asarray(condition.get("map_resolution_m", 0.08)).reshape(-1)[0])
    # Use a body-height projection; retaining all occupied voxels in the height band makes
    # the obstacle and narrow corridor visible in a compact 2-D panel.
    base_z = float(execution["base_pos"][tick, 2])
    z0 = max(0, int(np.floor((base_z - 0.75 - origin[2]) / resolution)))
    z1 = min(probability.shape[0], int(np.ceil((base_z + 0.90 - origin[2]) / resolution)))
    z1 = max(z0 + 1, z1)
    projected = np.max(probability[z0:z1], axis=0)
    observed = np.any(np.abs(probability[z0:z1] - 0.5) > 0.08, axis=0)
    intensity = np.clip((1.0 - projected) * 210.0 + 35.0, 0, 255).astype(np.uint8)
    rgb = np.stack([intensity, intensity, intensity], axis=2)
    rgb[~observed] = UNKNOWN
    rgb[projected > 0.68] = OCCUPIED
    map_image = Image.fromarray(np.flipud(rgb), mode="RGB").resize(
        (width - 24, height - 52), getattr(Image, "Resampling", Image).NEAREST)
    panel.paste(map_image, (12, 40))

    def pixel(point: np.ndarray) -> tuple[int, int]:
        cell = (np.asarray(point, dtype=np.float64)[:2] - origin[:2]) / resolution
        x = 12 + (cell[0] + 0.5) / max(probability.shape[2], 1) * (width - 24)
        y = 40 + (probability.shape[1] - cell[1] - 0.5) / max(probability.shape[1], 1) * (height - 52)
        return int(round(x)), int(round(y))

    route = np.asarray(condition.get("route_world_xyz", []), dtype=np.float64)
    executed = np.asarray(execution["base_pos"][:tick + 1], dtype=np.float64)
    scout = np.asarray(condition.get("scout_trace_world_xy", []), dtype=np.float64)
    if len(scout) > 1:
        draw.line([pixel(np.r_[p, base_z]) for p in scout], fill=SCOUT, width=2)
    if len(route) > 1:
        draw.line([pixel(p) for p in route], fill=PLANNED, width=4)
    if len(executed) > 1:
        draw.line([pixel(p) for p in executed], fill=EXEC, width=4)
    goal = np.asarray(condition.get("goal_world_xy", route[-1, :2] if len(route) else [0, 0]))
    if len(route):
        start = pixel(route[0])
        draw.ellipse((start[0] - 5, start[1] - 5, start[0] + 5, start[1] + 5), fill=PLANNED)
    current = pixel(executed[-1])
    draw.ellipse((current[0] - 6, current[1] - 6, current[0] + 6, current[1] + 6), fill=ROBOT)
    target = pixel(np.r_[goal[:2], base_z])
    draw.ellipse((target[0] - 6, target[1] - 6, target[0] + 6, target[1] + 6), fill=GOAL)
    _text(draw, "yellow=deploy A*   orange=SONIC/MuJoCo   green=actual root", (12, height - 21))
    return panel


def _slice_image(condition: dict[str, np.ndarray], width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(panel)
    _text(draw, "3-D voxel slices (deploy map)", (12, 10))
    probability = np.asarray(condition["map_probability"], dtype=np.float32)
    indices = np.unique([0, probability.shape[0] // 2, probability.shape[0] - 1]).astype(int)
    tile_w = (width - 48) // len(indices)
    tile_h = height - 78
    for order, index in enumerate(indices):
        p = probability[index]
        observed = np.abs(p - 0.5) > 0.08
        intensity = np.clip((1.0 - p) * 210.0 + 35.0, 0, 255).astype(np.uint8)
        rgb = np.stack([intensity, intensity, intensity], axis=2)
        rgb[~observed] = UNKNOWN
        rgb[p > 0.68] = OCCUPIED
        tile = Image.fromarray(np.flipud(rgb), mode="RGB").resize(
            (tile_w, tile_h), getattr(Image, "Resampling", Image).NEAREST)
        left = 12 + order * (tile_w + 12)
        panel.paste(tile, (left, 42))
        z = float(np.asarray(condition.get("map_origin_world_xyz", [0, 0, -0.32]))[2]) + (index + 0.5) * 0.08
        _text(draw, f"z={z:.2f} m", (left + 4, height - 27))
    return panel


def _side_image(condition: dict[str, np.ndarray], execution: dict[str, np.ndarray], corridor: np.ndarray,
                tick: int, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(panel)
    _text(draw, "planned corridor / actual x-z", (12, 10))
    route = np.asarray(condition.get("route_world_xyz", []), dtype=np.float64)
    executed = np.asarray(execution["base_pos"][:tick + 1], dtype=np.float64)
    radar = np.asarray(condition.get("radar_points_world", []), dtype=np.float64).reshape(-1, 3)
    chunks = [x[:, [0, 2]] for x in (route, executed, radar) if len(x)]
    points = np.concatenate(chunks, axis=0) if chunks else np.zeros((1, 2))
    xmin, xmax = float(points[:, 0].min() - .35), float(points[:, 0].max() + .35)
    zmin, zmax = 0.0, float(max(2.3, points[:, 1].max() + .35))

    def pixel(x: float, z: float) -> tuple[int, int]:
        return (int(26 + (x - xmin) / max(xmax - xmin, 1e-8) * (width - 42)),
                int(height - 30 - (z - zmin) / max(zmax - zmin, 1e-8) * (height - 70)))

    draw.line([pixel(xmin, 0), pixel(xmax, 0)], fill=GRID, width=2)
    if len(route) > 1:
        draw.line([pixel(float(x), float(z)) for x, z in route[:, [0, 2]]], fill=PLANNED, width=4)
    if len(executed) > 1:
        draw.line([pixel(float(x), float(z)) for x, z in executed[:, [0, 2]]], fill=EXEC, width=4)
    if len(radar):
        for x, z in radar[::max(1, len(radar) // 100), [0, 2]]:
            px, py = pixel(float(x), float(z))
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(255, 100, 80))
    if len(executed):
        px, py = pixel(float(executed[-1, 0]), float(executed[-1, 2]))
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=ROBOT)
    # Corridor centres are expressed relative to deploy root; route_world_xyz is the
    # corresponding global path, so map each condition element to the closest route sample.
    for index in range(0, len(corridor), max(1, len(corridor) // 8)):
        if not len(route):
            break
        route_index = min(int(round(index * (len(route) - 1) / max(len(corridor) - 1, 1))), len(route) - 1)
        x, z = float(route[route_index, 0]), float(route[route_index, 2])
        cx, cy = pixel(x, z)
        sx = abs(pixel(x + float(corridor[index, 3]), z)[0] - cx)
        sz = abs(pixel(x, z + float(corridor[index, 5]))[1] - cy)
        draw.ellipse((cx - sx, cy - sz, cx + sx, cy + sz), outline=TUBE, width=2)
    _text(draw, "yellow=planned   orange=physical execution", (12, height - 21))
    return panel


def _mujoco_image(env: G1FlatEnv, renderer: mujoco.Renderer, camera: mujoco.MjvCamera,
                  option: mujoco.MjvOption, execution: dict[str, np.ndarray], corridor: np.ndarray,
                  tick: int, width: int, height: int) -> Image.Image:
    q_exec = np.asarray(execution["q_exec"][tick], dtype=np.float64)
    base_pos = np.asarray(execution["base_pos"][tick], dtype=np.float64)
    base_quat = np.asarray(execution["base_quat"][tick], dtype=np.float64)
    base_quat /= max(float(np.linalg.norm(base_quat)), 1e-8)
    env.data.qpos[:3] = base_pos
    env.data.qpos[3:7] = base_quat
    env.data.qpos[env.body_qadr] = q_exec[C.ISAACLAB_TO_MUJOCO]
    env.data.qpos[env.hand_qadr] = 0.0
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    camera.lookat[:] = base_pos
    renderer.update_scene(env.data, camera=camera, scene_option=option)
    scene = renderer.scene
    for point in np.asarray(execution["base_pos"][:tick + 1])[::max(1, tick // 50 + 1)]:
        _add_sphere(scene, point, 0.017, (1.0, 0.45, 0.08, 0.75))
    if corridor is not None and len(corridor):
        element = corridor[min(tick, len(corridor) - 1)]
        # This is the current body's own safety envelope.  It is deliberately centred on
        # the physical root, so a bad deploy route cannot make the rendered robot fly away.
        _add_ellipsoid(scene, base_pos, np.asarray(element[3:6]), float(element[6]), (0.15, 0.95, 0.85, 0.18))
    frame = Image.fromarray(np.asarray(renderer.render()), mode="RGB")
    frame = frame.resize((width, height - 38), getattr(Image, "Resampling", Image).LANCZOS)
    panel = Image.new("RGB", (width, height), BG)
    panel.paste(frame, (0, 38))
    draw = ImageDraw.Draw(panel)
    _text(draw, f"MuJoCo physical SONIC execution  tick={tick + 1}/{len(execution['q_exec'])}", (12, 12))
    return panel


def render(args: argparse.Namespace) -> Path:
    sample = _load_condition(args.sample)
    condition = _load_condition(args.condition)
    execution, summary = _load_or_execute(args, sample)
    for key in ("q_exec", "base_pos", "base_quat"):
        if key not in execution:
            raise ValueError(f"execution is missing {key}")
    corridor = sample.get("condition_corridor", condition.get("corridor"))
    if corridor is not None:
        corridor = _resample_corridor(np.asarray(corridor, dtype=np.float64), len(execution["q_exec"]))
    env = G1FlatEnv(args.scene)
    renderer = mujoco.Renderer(env.model, height=args.height - 38, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    camera.distance, camera.azimuth, camera.elevation = args.distance, 125.0, -14.0
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)

    n = len(execution["q_exec"])
    step = max(1, int(round(float(args.control_hz) / float(args.fps))))
    ticks = list(range(0, n, step))
    if ticks[-1] != n - 1:
        ticks.append(n - 1)
    frames: list[Image.Image] = []
    panel_w, panel_h = args.width, args.height
    try:
        for tick in ticks:
            robot = _mujoco_image(env, renderer, camera, option, execution, corridor, tick, panel_w, panel_h)
            grid = _grid_image(condition, execution, tick, panel_w, panel_h)
            slices = _slice_image(condition, panel_w, panel_h)
            side = _side_image(condition, execution, corridor if corridor is not None else np.zeros((0, 7)), tick,
                               panel_w, panel_h)
            frame = Image.new("RGB", (panel_w * 2, panel_h * 2), BG)
            frame.paste(robot, (0, 0))
            frame.paste(grid, (panel_w, 0))
            frame.paste(slices, (0, panel_h))
            frame.paste(side, (panel_w, panel_h))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, panel_w * 2, 28), fill=(13, 18, 26))
            accepted = bool(summary.get("accepted", False))
            status = "PASS" if accepted else "DIAGNOSTIC: route gate not passed"
            draw.text((12, 7), f"deploy P1 condition -> main router -> Stage-2 -> SONIC -> MuJoCo | {status} | t={tick / args.control_hz:.2f}s",
                       fill=TEXT)
            frames.append(frame)
    finally:
        renderer.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(round(1000 / args.fps)),
                    loop=0, optimize=False)
    report = {"out": str(args.out), "frames": len(frames), "fps": args.fps,
              "physical_ticks": n, "frame_step": step, "summary": summary}
    print(json.dumps(report, indent=2))
    return args.out


def main() -> int:
    parser = argparse.ArgumentParser(description="Render deploy perception plus physical SONIC/MuJoCo execution")
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--executed", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--reuse-executed", action="store_true")
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--source-hz", type=float, default=30.0)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--max-corridor-radius", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=520)
    parser.add_argument("--height", type=int, default=390)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--distance", type=float, default=2.7)
    args = parser.parse_args()
    if min(args.width, args.height, args.fps, args.control_hz) <= 0:
        parser.error("width, height, fps and control-hz must be positive")
    render(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
