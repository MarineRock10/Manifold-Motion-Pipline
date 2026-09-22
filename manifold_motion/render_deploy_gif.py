"""Render the deploy perception loop as a compact, headless comprehensive GIF."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import constants as C
from .deploy_perception import (
    ProbabilisticSlidingVoxelGrid,
    RadarConfig,
    SimulatedRadar,
    SlidingGridConfig,
    safe_corridor_from_grid,
    voxel_astar,
)


WIDTH, HEIGHT = 420, 420
BACKGROUND = (25, 31, 40)
TEXT = (225, 232, 240)
GRID = (65, 77, 91)
ROUTE = (255, 215, 55)
SCOUT = (50, 190, 230)
ROBOT = (45, 220, 110)
GOAL = (65, 125, 250)
ELLIPSE = (40, 235, 220)
RETURN = (255, 100, 80)


def _label(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], fill=TEXT) -> None:
    draw.text(xy, text, fill=fill)


def _world_to_pixel(grid: ProbabilisticSlidingVoxelGrid, point_xy: np.ndarray,
                    width: int, height: int, margin: int = 12) -> tuple[int, int]:
    cell = grid.world_to_global_cell_xy(np.asarray(point_xy, dtype=np.float64)[None, :])[0]
    local = cell - grid.origin_cell_xyz[:2]
    sx = (width - 2 * margin) / float(grid.shape_zyx[2])
    sy = (height - 2 * margin) / float(grid.shape_zyx[1])
    return (int(margin + (local[0] + 0.5) * sx),
            int(height - margin - (local[1] + 0.5) * sy))


def _ellipse_polygon(center: tuple[int, int], semi: tuple[float, float], yaw: float,
                    grid: ProbabilisticSlidingVoxelGrid, width: int, height: int,
                    margin: int = 12) -> list[tuple[int, int]]:
    sx = (width - 2 * margin) / float(grid.shape_zyx[2]) / grid.config.resolution_m
    sy = (height - 2 * margin) / float(grid.shape_zyx[1]) / grid.config.resolution_m
    points = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
        local = np.array([semi[0] * math.cos(angle), semi[1] * math.sin(angle)])
        rotated = np.array([[math.cos(yaw), -math.sin(yaw)],
                            [math.sin(yaw), math.cos(yaw)]]) @ local
        points.append((int(center[0] + rotated[0] * sx),
                      int(center[1] - rotated[1] * sy)) )
    return points


def _corridor_world(corridor: np.ndarray, root: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
    rotation = C.quat_to_matrix(quat)
    forward = C.quat_rotate(quat, np.array([1.0, 0.0, 0.0]))
    root_yaw = math.atan2(float(forward[1]), float(forward[0]))
    centres = np.column_stack([
        root[:2] + corridor[:, :2] @ rotation[:2, :2].T,
        root[2] + corridor[:, 2],
    ])
    return centres, corridor[:, 6] + root_yaw


def _add_sphere(scene, position: np.ndarray, radius: float,
                color: tuple[float, float, float, float]) -> None:
    """Add a marker to a MuJoCo offscreen scene without requiring a GUI context."""
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0.0, 0.0]), position,
                        np.eye(3).flatten(), np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def _add_ellipsoid(scene, center: np.ndarray, semi: np.ndarray, yaw: float,
                   color: tuple[float, float, float, float]) -> None:
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                         [np.sin(yaw), np.cos(yaw), 0.0],
                         [0.0, 0.0, 1.0]])
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        np.asarray(semi, dtype=np.float64), center, rotation.flatten(),
                        np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def _robot_panel(renderer, env, root: np.ndarray, quat: np.ndarray, route: np.ndarray,
                 corridor: np.ndarray, corridor_root: np.ndarray, corridor_quat: np.ndarray,
                 trace: np.ndarray, goal: np.ndarray, tick: int) -> Image.Image:
    """Render the actual G1 MuJoCo model at the current perception pose."""
    import mujoco

    panel = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    _label(draw, "MuJoCo G1 / simulated perception pose", (12, 10))
    env.data.qpos[:3] = root
    env.data.qpos[3:7] = quat / max(float(np.linalg.norm(quat)), 1e-8)
    env.data.qpos[env.body_qadr] = C.DEFAULT_ANGLES
    env.data.qpos[env.hand_qadr] = 0.0
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    camera = renderer._deploy_camera
    camera.lookat[:] = root
    renderer.update_scene(env.data, camera=camera, scene_option=renderer._deploy_option)
    scene = renderer.scene
    for point in route[::max(1, len(route) // 35)]:
        _add_sphere(scene, point, 0.018, (0.1, 0.85, 1.0, 0.70))
    for point in trace[-80:]:
        _add_sphere(scene, np.array([point[0], point[1], root[2]]), 0.022, (1.0, 0.55, 0.1, 0.80))
    centres_world, yaws_world = _corridor_world(corridor, corridor_root, corridor_quat)
    active_index = int(np.argmin(np.linalg.norm(centres_world - root[None, :], axis=1)))
    _add_ellipsoid(scene, centres_world[active_index], corridor[active_index, 3:6],
                   float(yaws_world[active_index]), (0.1, 1.0, 0.8, 0.16))
    _add_sphere(scene, np.array([goal[0], goal[1], root[2]]), 0.045, (0.2, 0.35, 1.0, 0.9))
    rendered = Image.fromarray(np.asarray(renderer.render()), mode="RGB")
    rendered = rendered.resize((WIDTH, HEIGHT - 42), getattr(Image, "Resampling", Image).LANCZOS)
    panel.paste(rendered, (0, 42))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, WIDTH, 38), fill=BACKGROUND)
    _label(draw, f"kinematic MuJoCo pose {tick}", (12, 12))
    return panel


def _top_panel(grid: ProbabilisticSlidingVoxelGrid, route: np.ndarray,
               trace: np.ndarray, radar_points: np.ndarray, corridor: np.ndarray,
               root: np.ndarray, quat: np.ndarray, goal: np.ndarray,
               robot_pose: np.ndarray | None = None) -> Image.Image:
    width, height = WIDTH, HEIGHT
    panel = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    z_min = root[2] + grid.config.body_z_min_offset_m
    z_max = root[2] + grid.config.body_z_max_offset_m
    probability = grid.probability_xy_projection(z_min, z_max)
    observed = ~grid.unknown_xy_projection(z_min, z_max)
    intensity = np.clip((1.0 - probability) * 210.0 + 35.0, 0, 255).astype(np.uint8)
    rgb = np.stack([intensity, intensity, intensity], axis=2)
    rgb[~observed] = np.array([105, 112, 122], dtype=np.uint8)
    rgb[grid.occupancy_xy_projection(z_min, z_max)] = np.array([188, 58, 55], dtype=np.uint8)
    map_image = Image.fromarray(np.flipud(rgb), mode="RGB").resize((width - 24, height - 50),
        getattr(Image, "Resampling", Image).NEAREST)
    panel.paste(map_image, (12, 38))
    draw = ImageDraw.Draw(panel)
    _label(draw, "3-D voxel map / body-height projection", (12, 10))

    def pixel(point: np.ndarray) -> tuple[int, int]:
        x, y = _world_to_pixel(grid, point, width, height - 50, margin=12)
        return x, y + 38

    for first, second in zip(trace[:-1], trace[1:]):
        draw.line([pixel(first), pixel(second)], fill=SCOUT, width=3)
    for first, second in zip(route[:-1, :2], route[1:, :2]):
        draw.line([pixel(first), pixel(second)], fill=ROUTE, width=4)
    centres_world, yaws_world = _corridor_world(corridor, root, quat)
    for index in range(0, len(centres_world), max(1, len(centres_world) // 8)):
        centre = pixel(centres_world[index, :2])
        polygon = _ellipse_polygon(centre, corridor[index, 3:5], float(yaws_world[index]),
                                   grid, width, height - 50)
        polygon = [(x, y + 38) for x, y in polygon]
        draw.line(polygon + [polygon[0]], fill=ELLIPSE, width=2)
    for point in np.asarray(radar_points).reshape(-1, 3)[::max(1, len(radar_points) // 80 or 1)]:
        x, y = pixel(point[:2])
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=RETURN)
    rx, ry = pixel((root if robot_pose is None else robot_pose)[:2])
    gx, gy = pixel(goal)
    draw.ellipse((rx - 6, ry - 6, rx + 6, ry + 6), fill=ROBOT, outline=(0, 0, 0))
    draw.ellipse((gx - 6, gy - 6, gx + 6, gy + 6), fill=GOAL, outline=(0, 0, 0))
    return panel


def _slice_panel(grid: ProbabilisticSlidingVoxelGrid) -> Image.Image:
    panel = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    _label(draw, "3-D voxel slices", (12, 10))
    probability = grid.probability()
    occupied = grid.occupancy_mask()
    unknown = grid.unknown_mask()
    indices = np.unique(np.asarray([0, grid.shape_zyx[0] // 2, grid.shape_zyx[0] - 1]))
    tile_width = (WIDTH - 48) // len(indices)
    tile_height = HEIGHT - 78
    for order, z_index in enumerate(indices):
        intensity = np.clip((1.0 - probability[z_index]) * 210.0 + 35.0, 0, 255).astype(np.uint8)
        rgb = np.stack([intensity, intensity, intensity], axis=2)
        rgb[unknown[z_index]] = np.array([105, 112, 122], dtype=np.uint8)
        rgb[occupied[z_index]] = np.array([188, 58, 55], dtype=np.uint8)
        tile = Image.fromarray(np.flipud(rgb), mode="RGB").resize((tile_width, tile_height),
            getattr(Image, "Resampling", Image).NEAREST)
        left = 12 + order * (tile_width + 12)
        panel.paste(tile, (left, 42))
        draw.text((left + 4, HEIGHT - 27),
                  f"z={grid.origin_world_xyz[2] + (int(z_index) + .5) * grid.config.resolution_m:.2f} m",
                  fill=TEXT)
    return panel


def _side_panel(route: np.ndarray, corridor: np.ndarray, radar_points: np.ndarray,
                root: np.ndarray, goal: np.ndarray, robot_pose: np.ndarray | None = None) -> Image.Image:
    panel = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    _label(draw, "3-D ellipsoid corridor / x-z view", (12, 10))
    all_x = np.concatenate([route[:, 0], np.asarray(radar_points).reshape(-1, 3)[:, 0], [root[0], goal[0]]])
    all_z = np.concatenate([route[:, 2], np.asarray(radar_points).reshape(-1, 3)[:, 2], [root[2], root[2]]])
    xmin, xmax = float(all_x.min() - .35), float(all_x.max() + .35)
    zmin, zmax = 0.0, float(max(2.2, all_z.max() + .35))

    def pixel(point_x: float, point_z: float) -> tuple[int, int]:
        return (int(26 + (point_x - xmin) / max(xmax - xmin, 1e-6) * (WIDTH - 42)),
                int(HEIGHT - 28 - (point_z - zmin) / max(zmax - zmin, 1e-6) * (HEIGHT - 70)))

    draw.line([pixel(xmin, 0.0), pixel(xmax, 0.0)], fill=GRID, width=2)
    for point in np.asarray(radar_points).reshape(-1, 3)[::max(1, len(radar_points) // 100 or 1)]:
        x, y = pixel(float(point[0]), float(point[2]))
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=RETURN)
    route_points = [pixel(float(x), float(z)) for x, z in route[:, [0, 2]]]
    draw.line(route_points, fill=ROUTE, width=4)
    # z-axis ellipsoid projection: sx and sz are the horizontal/vertical semiaxes.
    for index in range(0, len(corridor), max(1, len(corridor) // 8)):
        route_index = min(int(round(index * (len(route) - 1) / max(len(corridor) - 1, 1))), len(route) - 1)
        x, z = float(route[route_index, 0]), float(route[route_index, 2])
        cx, cy = pixel(x, z)
        sx = abs(pixel(x + float(corridor[index, 3]), z)[0] - cx)
        sz = abs(pixel(x, z + float(corridor[index, 5]))[1] - cy)
        draw.ellipse((cx - sx, cy - sz, cx + sx, cy + sz), outline=ELLIPSE, width=2)
    marker = root if robot_pose is None else robot_pose
    rx, ry = pixel(float(marker[0]), float(marker[2]))
    gx, gy = pixel(float(goal[0]), float(marker[2]))
    draw.ellipse((rx - 6, ry - 6, rx + 6, ry + 6), fill=ROBOT)
    draw.ellipse((gx - 6, gy - 6, gx + 6, gy + 6), fill=GOAL)
    return panel


def render(scene: Path, out: Path, *, goal: np.ndarray, seed: int = 20260922,
           pose_count: int = 5, smooth_substeps: int = 5) -> Path:
    import mujoco

    radar = SimulatedRadar(scene, RadarConfig(seed=seed))
    config = SlidingGridConfig()
    root = np.array([0.0, 0.0, 0.78], dtype=np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    goal_xyz = np.array([goal[0], goal[1], root[2]], dtype=np.float64)
    grid = ProbabilisticSlidingVoxelGrid(config, root)
    trace = [root[:2].copy()]
    frames: list[Image.Image] = []
    route = np.column_stack([np.linspace(0, goal[0], 2), np.zeros(2), np.full(2, root[2])])
    env = __import__("manifold_motion.env", fromlist=["G1FlatEnv"]).G1FlatEnv(scene)
    renderer = mujoco.Renderer(env.model, height=HEIGHT - 42, width=WIDTH)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    camera.distance, camera.azimuth, camera.elevation = 4.6, 125.0, -18.0
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    renderer._deploy_camera = camera
    renderer._deploy_option = option
    for timestamp in range(max(2, pose_count)):
        scan = radar.scan(root, quat, timestamp=timestamp * 0.1)
        grid.update_radar(root, scan)
        route = voxel_astar(grid, root, goal_xyz, body_radius_m=0.40,
                            body_half_height_m=0.78, clearance_m=0.10, allow_unknown=True)
        corridor, _, report = safe_corridor_from_grid(grid, route, root, quat)
        current_root = root.copy()
        next_root = current_root.copy()
        if timestamp + 1 < max(2, pose_count) and len(route) > 1:
            step_lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
            next_index = min(int(np.searchsorted(np.cumsum(step_lengths), .45, side="left") + 1), len(route) - 1)
            next_root = route[next_index].astype(np.float64)
        for substep in range(max(1, smooth_substeps)):
            alpha = (substep + 1) / max(1, smooth_substeps)
            robot_pose = current_root * (1.0 - alpha) + next_root * alpha
            robot_panel = _robot_panel(renderer, env, robot_pose, quat, route, corridor,
                                       current_root, quat, np.asarray(trace), goal, timestamp)
            top = _top_panel(grid, route, np.asarray(trace), scan.points_world,
                             corridor, current_root, quat, goal, robot_pose=robot_pose)
            slices = _slice_panel(grid)
            side = _side_panel(route, corridor, scan.points_world, current_root, goal,
                               robot_pose=robot_pose)
            frame = Image.new("RGB", (WIDTH * 2, HEIGHT * 2), BACKGROUND)
            frame.paste(robot_panel, (0, 0))
            frame.paste(top, (WIDTH, 0))
            frame.paste(slices, (0, HEIGHT))
            frame.paste(side, (WIDTH, HEIGHT))
            header = ImageDraw.Draw(frame)
            header.rectangle((0, 0, WIDTH * 2, 28), fill=(15, 19, 27))
            header.text((12, 7), f"radar/SLAM update {timestamp + 1}/{max(2, pose_count)}  "
                        f"returns={len(scan.points_world)}  "
                        f"3D A* z=[{route[:,2].min():.2f},{route[:,2].max():.2f}]m  "
                        f"ellipsoid sz min={corridor[:,5].min():.2f}m", fill=TEXT)
            frames.append(frame)
        root = next_root
        trace.append(root[:2].copy())
    renderer.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="render integrated 3-D deploy perception GIF")
    parser.add_argument("--scene", type=Path, default=Path("data/g1_flat/scene_long_avoidance.xml"))
    parser.add_argument("--goal", type=float, nargs=2, default=(3.6, 0.0))
    parser.add_argument("--out", type=Path,
                        default=Path("reports/manifold_motion/deploy_perception_comprehensive.gif"))
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--pose-count", type=int, default=5)
    args = parser.parse_args()
    path = render(args.scene, args.out, goal=np.asarray(args.goal, dtype=np.float64),
                  seed=args.seed, pose_count=args.pose_count)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
