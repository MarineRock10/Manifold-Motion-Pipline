"""Compose the accepted MuJoCo rollout with its synchronized P1 SLAM view.

The deploy perception demo builds the map before the continuous Stage-2 rollout.  This
renderer keeps that contract explicit: the first few frames replay the five radar updates,
then every MuJoCo frame is paired with the same rollout tick on the final probability grid.
The robot marker, executed trace, active primitive, and measured self-manifold are therefore
time-synchronized without claiming that the frozen P1 map is being updated during Stage 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .deploy_perception import (
    ProbabilisticSlidingVoxelGrid,
    RadarScan,
    SlidingGridConfig,
)


BACKGROUND = (15, 20, 29)
PANEL = (22, 29, 40)
TEXT = (230, 237, 245)
MUTED = (157, 171, 188)
UNKNOWN = (70, 78, 91)
FREE = (37, 79, 102)
OCCUPIED = (205, 61, 54)
ROUTE = (255, 214, 55)
TRACE = (255, 142, 33)
ROBOT = (47, 225, 121)
SCOUT = (79, 188, 237)
RADAR = (255, 106, 85)
CORRIDOR = (45, 236, 210)
SELF_MANIFOLD = (214, 82, 255)
GOAL = (91, 139, 255)


def _font(size: int = 14):
    # The default font is portable across the WSL/Windows headless environments used here.
    try:
        from PIL import ImageFont
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        from PIL import ImageFont
        return ImageFont.load_default()


def _line_points(points: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float],
                 width: int, height: int) -> list[tuple[int, int]]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        return []
    x0, x1 = xlim
    y0, y1 = ylim
    result = []
    for x, y in values[:, :2]:
        px = int(round((x - x0) / max(x1 - x0, 1e-6) * (width - 1)))
        py = int(round((y1 - y) / max(y1 - y0, 1e-6) * (height - 1)))
        result.append((px, py))
    return result


def _ellipse_points(center: np.ndarray, semi_x: float, semi_y: float, yaw: float,
                    xlim: tuple[float, float], ylim: tuple[float, float],
                    width: int, height: int, count: int = 32) -> list[tuple[int, int]]:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    local = np.column_stack([semi_x * np.cos(angles), semi_y * np.sin(angles)])
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    rotated = local @ np.array([[c, s], [-s, c]], dtype=np.float64)
    world = rotated + np.asarray(center, dtype=np.float64)[None, :2]
    return _line_points(world, xlim, ylim, width, height)


def _limits(route: np.ndarray, trace: np.ndarray, radar: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    clouds = [np.asarray(route)[:, :2], np.asarray(trace)[:, :2]]
    if len(radar):
        clouds.append(np.asarray(radar)[:, :2])
    cloud = np.concatenate(clouds, axis=0)
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    # Preserve a useful aspect ratio for the long corridor while keeping the detour visible.
    xlim = (float(lo[0] - 0.65), float(hi[0] + 0.65))
    half_y = max(float(hi[1] - lo[1]) * 0.5 + 0.65, 1.55)
    centre_y = float((hi[1] + lo[1]) * 0.5)
    return xlim, (centre_y - half_y, centre_y + half_y)


def _crop_cells(origin_xyz: np.ndarray, resolution: float, shape_zyx: tuple[int, int, int],
                xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = int(shape_zyx[2]), int(shape_zyx[1])
    xs = origin_xyz[0] + (np.arange(nx, dtype=np.float64) + 0.5) * resolution
    ys = origin_xyz[1] + (np.arange(ny, dtype=np.float64) + 0.5) * resolution
    ix = np.flatnonzero((xs >= xlim[0]) & (xs <= xlim[1]))
    iy = np.flatnonzero((ys >= ylim[0]) & (ys <= ylim[1]))
    if not len(ix) or not len(iy):
        raise ValueError("SLAM grid does not overlap the requested route view")
    return ix, iy


def _map_image(probability: np.ndarray, origin_xyz: np.ndarray, config: SlidingGridConfig,
               xlim: tuple[float, float], ylim: tuple[float, float], width: int, height: int,
               z_min: float = 0.02, z_max: float = 1.65) -> Image.Image:
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 3:
        raise ValueError("probability must be [z,y,x]")
    ix, iy = _crop_cells(origin_xyz, config.resolution_m, probability.shape, xlim, ylim)
    z_centres = origin_xyz[2] + (np.arange(probability.shape[0], dtype=np.float64) + 0.5) * config.resolution_m
    iz = np.flatnonzero((z_centres >= z_min) & (z_centres <= z_max))
    if not len(iz):
        iz = np.arange(probability.shape[0])
    volume = probability[np.ix_(iz, iy, ix)]
    occupied = np.max(volume, axis=0) >= config.occupied_probability
    free = np.min(volume, axis=0) < (config.prior_probability - config.unknown_band)
    known = np.any(np.abs(volume - config.prior_probability) > config.unknown_band, axis=0)
    intensity = np.clip(75.0 + (1.0 - np.min(volume, axis=0)) * 105.0, 0.0, 255.0).astype(np.uint8)
    rgb = np.empty((len(iy), len(ix), 3), dtype=np.uint8)
    rgb[:] = np.asarray(UNKNOWN, dtype=np.uint8)
    rgb[known & free] = np.stack([intensity[known & free] * 0.25,
                                  intensity[known & free] * 0.70,
                                  intensity[known & free]], axis=1).astype(np.uint8)
    rgb[occupied] = np.asarray(OCCUPIED, dtype=np.uint8)
    image = Image.fromarray(np.flipud(rgb), mode="RGB")
    return image.resize((width, height), getattr(Image, "Resampling", Image).NEAREST)


def _slice_image(probability: np.ndarray, origin_xyz: np.ndarray, config: SlidingGridConfig,
                 z_index: int, xlim: tuple[float, float], ylim: tuple[float, float],
                 width: int, height: int) -> Image.Image:
    probability = np.asarray(probability, dtype=np.float32)
    z_index = int(np.clip(z_index, 0, probability.shape[0] - 1))
    ix, iy = _crop_cells(origin_xyz, config.resolution_m, probability.shape, xlim, ylim)
    plane = probability[z_index][np.ix_(iy, ix)]
    occupied = plane >= config.occupied_probability
    known = np.abs(plane - config.prior_probability) > config.unknown_band
    rgb = np.empty((*plane.shape, 3), dtype=np.uint8)
    rgb[:] = np.asarray(UNKNOWN, dtype=np.uint8)
    rgb[known] = np.asarray(FREE, dtype=np.uint8)
    rgb[occupied] = np.asarray(OCCUPIED, dtype=np.uint8)
    return Image.fromarray(np.flipud(rgb), mode="RGB").resize(
        (width, height), getattr(Image, "Resampling", Image).NEAREST)


def _draw_map_panel(panel: Image.Image, probability: np.ndarray, origin_xyz: np.ndarray,
                    config: SlidingGridConfig, xlim: tuple[float, float], ylim: tuple[float, float],
                    route: np.ndarray, environment_corridor: np.ndarray, scout_trace: np.ndarray,
                    radar_points: np.ndarray, executed: dict[str, np.ndarray], tick: int,
                    update_label: str, accepted: bool, map_updates: int) -> None:
    width, height = panel.size
    draw = ImageDraw.Draw(panel)
    title = _font(16)
    body = _font(12)
    small = _font(10)
    draw.rectangle((0, 0, width, height), fill=PANEL)
    draw.text((12, 8), "Synchronized P1 SLAM / Stage-2 state", fill=TEXT, font=title)
    draw.text((12, 29), update_label, fill=(108, 224, 255), font=body)
    map_box = (10, 52, width - 10, min(535, height - 150))
    map_x, map_y = map_box[0], map_box[1]
    map_w, map_h = map_box[2] - map_box[0], map_box[3] - map_box[1]
    base = _map_image(probability, origin_xyz, config, xlim, ylim, map_w, map_h)
    panel.paste(base, (map_x, map_y))

    def pts(values: np.ndarray) -> list[tuple[int, int]]:
        raw = _line_points(values, xlim, ylim, map_w, map_h)
        return [(map_x + x, map_y + y) for x, y in raw]

    route_points = pts(route)
    if len(route_points) > 1:
        draw.line(route_points, fill=ROUTE, width=3, joint="curve")
    scout_points = pts(scout_trace)
    if len(scout_points) > 1:
        draw.line(scout_points, fill=SCOUT, width=2, joint="curve")
    actual = np.asarray(executed["base_pos"][:tick + 1, :2])
    actual_points = pts(actual)
    if len(actual_points) > 1:
        draw.line(actual_points, fill=TRACE, width=4, joint="curve")

    for point in np.asarray(radar_points).reshape(-1, 3)[::max(1, len(radar_points) // 80 or 1)]:
        p = pts(np.asarray(point[None, :2]))
        if p:
            x, y = p[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=RADAR)

    # Draw a sparse world-frame M_e tube to make the planner input visible.
    corridor = np.asarray(environment_corridor)
    for index in range(0, len(corridor), max(1, len(corridor) // 12)):
        row = corridor[index]
        poly = _ellipse_points(row[:2], float(row[3]), float(row[4]), float(row[6]),
                               xlim, ylim, map_w, map_h, count=24)
        poly = [(map_x + x, map_y + y) for x, y in poly]
        if len(poly) > 2:
            draw.line(poly + [poly[0]], fill=CORRIDOR, width=1)

    root = np.asarray(executed["base_pos"][tick])
    safe = np.asarray(executed["robot_manifold_safe"][tick])
    self_poly = _ellipse_points(safe[:2], float(safe[3]), float(safe[4]), float(safe[6]),
                                xlim, ylim, map_w, map_h, count=32)
    self_poly = [(map_x + x, map_y + y) for x, y in self_poly]
    if len(self_poly) > 2:
        draw.line(self_poly + [self_poly[0]], fill=SELF_MANIFOLD, width=3)
    root_point = pts(root[None, :2])[0]
    draw.ellipse((root_point[0] - 6, root_point[1] - 6,
                  root_point[0] + 6, root_point[1] + 6), fill=ROBOT, outline=(0, 0, 0))
    goal = np.asarray(route[-1, :2])
    goal_point = pts(goal[None, :])[0]
    draw.ellipse((goal_point[0] - 5, goal_point[1] - 5,
                  goal_point[0] + 5, goal_point[1] + 5), fill=GOAL, outline=(0, 0, 0))
    draw.rectangle(map_box, outline=(92, 109, 130), width=1)
    draw.text((map_x + 8, map_y + 8),
              "red=occupied  yellow=A*  orange=executed  magenta=M_r  cyan=M_e",
              fill=TEXT, font=small)

    # Three z slices are deliberately kept in the synchronized panel; this exposes that the
    # planner input is a 3-D probability volume rather than a single 2-D obstacle image.
    slice_top = map_box[3] + 10
    slice_height = min(120, height - slice_top - 48)
    slice_width = (width - 34) // 3
    z_centres = origin_xyz[2] + (np.arange(probability.shape[0]) + 0.5) * config.resolution_m
    for order, z_value in enumerate((0.12, 0.78, 1.42)):
        z_index = int(np.argmin(np.abs(z_centres - z_value)))
        tile = _slice_image(probability, origin_xyz, config, z_index, xlim, ylim,
                            slice_width, slice_height)
        left = 10 + order * (slice_width + 7)
        panel.paste(tile, (left, slice_top))
        draw.text((left + 4, slice_top + slice_height + 3),
                  f"z={z_centres[z_index]:.2f}m", fill=MUTED, font=small)
    active = str(np.asarray(executed["active_primitive_name"])[tick])
    clearance = float(np.asarray(executed["self_manifold_obstacle_clearance_m"])[tick])
    contact = bool(np.asarray(executed["obstacle_contact"])[tick])
    status = "PASS" if accepted and not contact else "FAIL"
    draw.text((12, height - 31),
              f"tick={tick:03d}  primitive={active}  clearance={clearance:.3f}m  "
              f"contact={contact}  hard-gate={status}  map-updates={map_updates}",
              fill=(119, 243, 151) if status == "PASS" else (255, 111, 100), font=body)


def _replay_slam_updates(condition: dict[str, np.ndarray], radar: dict[str, np.ndarray],
                         config: SlidingGridConfig
                         ) -> list[tuple[np.ndarray, np.ndarray, int, np.ndarray]]:
    root = np.asarray(condition["root_pos_world"], dtype=np.float64)
    scout = np.asarray(condition["scout_trace_world_xy"], dtype=np.float64)
    points = np.asarray(radar["points_world"], dtype=np.float32)
    origins = np.asarray(radar["origins_world"], dtype=np.float32)
    grid = ProbabilisticSlidingVoxelGrid(config, root)
    snapshots: list[tuple[np.ndarray, np.ndarray, int, np.ndarray]] = []
    accumulated: list[np.ndarray] = []
    unique_origins = np.unique(np.round(origins, 5), axis=0)
    for order, xy in enumerate(scout):
        if len(unique_origins):
            source = unique_origins[int(np.argmin(np.linalg.norm(unique_origins[:, :2] - xy[None, :], axis=1)))]
            mask = np.all(np.isclose(origins, source[None, :], atol=2e-4), axis=1)
        else:
            mask = np.zeros(len(points), dtype=bool)
        scan = RadarScan(points[mask], origins[mask], [""] * int(np.sum(mask)), float(order))
        grid.update_radar(np.array([xy[0], xy[1], root[2]], dtype=np.float64), scan)
        accumulated.append(points[mask])
        visible = np.concatenate(accumulated, axis=0) if accumulated else np.empty((0, 3), np.float32)
        snapshots.append((grid.probability().copy(), grid.origin_world_xyz.copy(),
                          int(grid.update_count), visible.copy()))
    return snapshots


def render(args: argparse.Namespace) -> Path:
    with np.load(args.condition) as archive:
        condition = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.slam_grid) as archive:
        slam = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.radar_returns) as archive:
        radar = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.executed) as archive:
        executed = {key: np.asarray(archive[key]) for key in archive.files}
    if not {"base_pos", "active_primitive_name", "robot_manifold_safe"}.issubset(executed):
        raise ValueError("executed.npz is missing synchronized pose/primitive/manifold fields")

    source = Image.open(args.mujoco_gif)
    source_frames: list[Image.Image] = []
    try:
        for index in range(source.n_frames):
            source.seek(index)
            source_frames.append(source.convert("RGB").copy())
    finally:
        source.close()
    if not source_frames:
        raise ValueError("MuJoCo GIF contains no frames")
    width, height = source_frames[0].size
    config = SlidingGridConfig()
    route = np.asarray(condition["route_world_xyz"], dtype=np.float64)
    environment_corridor = np.asarray(condition.get("environment_corridor", route), dtype=np.float64)
    # The continuous runner stores the full world-frame corridor separately from condition.npz.
    if environment_corridor.ndim != 2 or environment_corridor.shape[1] < 7:
        if args.segment_conditions is None:
            raise ValueError("environment corridor is unavailable; pass --segment-conditions")
        with np.load(args.segment_conditions) as archive:
            environment_corridor = np.asarray(archive["environment_corridor"], dtype=np.float64)
    route_xy = route[:, :2]
    trace = np.asarray(condition.get("scout_trace_world_xy", route_xy), dtype=np.float64)
    radar_points = np.asarray(radar["points_world"], dtype=np.float64)
    xlim, ylim = _limits(route_xy, np.asarray(executed["base_pos"])[:, :2], radar_points)
    snapshots = _replay_slam_updates(condition, radar, config)
    if not snapshots:
        snapshots = [(np.asarray(slam["probability"]), np.asarray(slam["origin_world_xyz"]),
                      int(np.asarray(slam.get("update_count", 0)).item()), radar_points)]
    final_probability = np.asarray(slam["probability"])
    final_origin = np.asarray(slam["origin_world_xyz"], dtype=np.float64)
    accepted = True
    if args.report is not None and args.report.exists():
        report = json.loads(args.report.read_text(encoding="utf-8"))
        accepted = bool(report.get("accepted", False))

    right_width = args.right_width
    canvas_size = (width + right_width, height)
    # Exact Stage-2 render sampling is 50 Hz control / requested 20 Hz GIF.  Infer that map so
    # the right-side tick corresponds to the same actual execution state as the left frame.
    n_exec = len(executed["base_pos"])
    if len(source_frames) == 1:
        source_ticks = [0]
    else:
        source_ticks = np.rint(np.linspace(0, n_exec - 1, len(source_frames))).astype(int).tolist()
    frames: list[Image.Image] = []
    warmup_hold = max(1, int(args.warmup_hold))
    first_frame = source_frames[0].resize((width, height), getattr(Image, "Resampling", Image).LANCZOS)
    for snap_index, (probability, origin, count, visible_radar) in enumerate(snapshots):
        for _ in range(warmup_hold):
            right = Image.new("RGB", (right_width, height), PANEL)
            dummy_tick = source_ticks[0]
            _draw_map_panel(right, probability, origin, config, xlim, ylim, route_xy,
                             environment_corridor, trace, visible_radar, executed, dummy_tick,
                             f"P1 radar update {snap_index + 1}/{len(snapshots)}  (map construction)",
                             accepted, count)
            frame = Image.new("RGB", canvas_size, BACKGROUND)
            frame.paste(first_frame, (0, 0))
            frame.paste(right, (width, 0))
            frames.append(frame)

    for source_frame, tick in zip(source_frames, source_ticks):
        right = Image.new("RGB", (right_width, height), PANEL)
        _draw_map_panel(right, final_probability, final_origin, config, xlim, ylim, route_xy,
                         environment_corridor, trace, radar_points, executed, int(tick),
                         "Stage-2 rollout  (P1 map frozen after 5 updates)", accepted,
                         int(np.asarray(slam.get("update_count", 5)).item()))
        frame = Image.new("RGB", canvas_size, BACKGROUND)
        frame.paste(source_frame.resize((width, height), getattr(Image, "Resampling", Image).LANCZOS), (0, 0))
        frame.paste(right, (width, 0))
        frames.append(frame)

    if not frames:
        raise RuntimeError("synchronized renderer produced no frames")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    palette = getattr(getattr(Image, "Palette", Image), "ADAPTIVE", getattr(Image, "ADAPTIVE", 1))
    indexed = [frame.convert("P", palette=palette, colors=int(args.colors), dither=0) for frame in frames]
    indexed[0].save(args.out, save_all=True, append_images=indexed[1:],
                    duration=int(round(1000.0 / args.fps)), loop=0, optimize=False)
    print(json.dumps({"out": str(args.out), "frames": len(indexed), "fps": args.fps,
                      "warmup_frames": len(snapshots) * warmup_hold,
                      "execution_frames": len(source_frames), "accepted": accepted}, indent=2))
    return args.out


def main() -> int:
    parser = argparse.ArgumentParser(description="compose synchronized MuJoCo and P1 SLAM GIF")
    parser.add_argument("--mujoco-gif", type=Path, required=True)
    parser.add_argument("--executed", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--slam-grid", type=Path, required=True)
    parser.add_argument("--radar-returns", type=Path, required=True)
    parser.add_argument("--segment-conditions", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--warmup-hold", type=int, default=5)
    parser.add_argument("--right-width", type=int, default=520)
    parser.add_argument("--colors", type=int, default=160)
    args = parser.parse_args()
    if min(args.fps, args.warmup_hold, args.right_width, args.colors) <= 0:
        parser.error("fps/warmup-hold/right-width/colors must be positive")
    render(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
