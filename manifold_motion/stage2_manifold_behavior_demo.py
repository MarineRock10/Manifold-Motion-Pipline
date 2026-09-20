"""Build a visible, physics-gated ``environment manifold -> motion`` demonstration.

The demo deliberately uses three small-amplitude primitives that the frozen SONIC executor
already supports.  A geometry-only router sees the corridor manifold and selects one of:

* a nominal forward walk in a wide corridor;
* a crouched walk below a physical low ceiling;
* a lateral/reverse escape when a front obstacle blocks the route.

Every selected motion is replayed in the corresponding MuJoCo scene.  The nominal walk is
also replayed unchanged in the constrained scenes as a physical counterfactual, so the report
cannot claim adaptation merely because three unrelated clips look different.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from . import constants as C
from .corridor import _resample_corridor
from .env import G1FlatEnv
from .primitive_router import _features, _load_router, _load_windows, _normalize
from .seed_replay import ReplayConfig, SeedReplayRunner
from .seed_windows import PRIMITIVE_NAMES
from .stage2_validate import validate_trajectory


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    title: str
    source_index: int
    primitive_id: int
    sample: Path
    scene: Path


@dataclass
class ScenarioResult:
    spec: ScenarioSpec
    sample: dict[str, np.ndarray]
    execution: dict[str, np.ndarray]
    selected_summary: dict[str, object]
    nominal_summary: dict[str, object]
    confidence: float
    probabilities: list[dict[str, object]]
    geometry_confidence: float
    geometry_probabilities: list[dict[str, object]]


def _specs() -> list[ScenarioSpec]:
    report = C.REPO / "reports" / "manifold_motion"
    scenes = C.REPO / "data" / "g1_flat"
    return [
        ScenarioSpec(
            "wide", "WIDE CORRIDOR  ->  WALK", 2072, 5,
            report / "stage2_routed_walk_test_pass" / "sample.npz",
            scenes / "scene_counterfactual_normal.xml",
        ),
        ScenarioSpec(
            "low", "LOW CEILING  ->  CROUCH", 554, 2,
            report / "stage2_routed_crouch_train0" / "sample.npz",
            scenes / "scene_manifold_low_demo.xml",
        ),
        ScenarioSpec(
            "blocked", "FRONT BLOCK  ->  SIDE/REVERSE", 1348, 4,
            report / "stage2_routed_side_test" / "sample.npz",
            scenes / "scene_counterfactual_side.xml",
        ),
    ]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _route(raw: dict[str, np.ndarray], router_path: Path, source_index: int
           ) -> tuple[int, float, list[dict[str, object]]]:
    router, checkpoint = _load_router(router_path)
    include_command = bool(checkpoint.get("include_command", True))
    feature = _features(raw, include_command=include_command)[source_index:source_index + 1]
    normalized = _normalize(feature, checkpoint["feature_mean"], checkpoint["feature_std"])
    with torch.no_grad():
        probability = torch.softmax(router(torch.as_tensor(normalized)), dim=1)[0].numpy()
    active = np.asarray(checkpoint["active_primitive_ids"], dtype=np.int64)
    order = np.argsort(probability)[::-1]
    ranking = [{"primitive_id": int(active[index]),
                "primitive": PRIMITIVE_NAMES[int(active[index])],
                "probability": float(probability[index])} for index in order]
    return int(active[order[0]]), float(probability[order[0]]), ranking


def _execute(sample: dict[str, np.ndarray], sample_path: Path, scene: Path,
             primitive_id: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    trajectory = np.asarray(sample["generated_ref"])
    corridor = np.asarray(sample["condition_corridor"])
    return validate_trajectory(
        trajectory,
        source=sample_path,
        source_hz=30.0,
        config=ReplayConfig(),
        corridor=corridor,
        stratum=PRIMITIVE_NAMES[primitive_id],
        runner=SeedReplayRunner(scene),
    )


def build(args: argparse.Namespace) -> list[ScenarioResult]:
    args.out.mkdir(parents=True, exist_ok=True)
    raw = _load_windows(args.windows)
    specs = _specs()
    nominal_sample = _load_npz(specs[0].sample)
    results: list[ScenarioResult] = []
    report: dict[str, object] = {
        "contract": "(environment manifold M, task command c) -> primitive -> generated reference -> frozen SONIC/MuJoCo",
        "router": str(args.router),
        "geometry_only_router": str(args.geometry_router),
        "windows": str(args.windows),
        "counterfactual": "same nominal-walk reference replayed unchanged in every physical scene",
        "scenarios": [],
    }
    for spec in specs:
        sample = _load_npz(spec.sample)
        routed_id, confidence, ranking = _route(raw, args.router, spec.source_index)
        geometry_id, geometry_confidence, geometry_ranking = _route(
            raw, args.geometry_router, spec.source_index
        )
        if routed_id != spec.primitive_id:
            raise RuntimeError(
                f"{spec.key}: router chose {PRIMITIVE_NAMES[routed_id]}, expected "
                f"{PRIMITIVE_NAMES[spec.primitive_id]}"
            )
        if geometry_id != spec.primitive_id:
            raise RuntimeError(
                f"{spec.key}: geometry-only router chose {PRIMITIVE_NAMES[geometry_id]}, expected "
                f"{PRIMITIVE_NAMES[spec.primitive_id]}"
            )
        execution, selected_summary = _execute(sample, spec.sample, spec.scene, spec.primitive_id)
        nominal_execution, nominal_summary = _execute(
            nominal_sample, specs[0].sample, spec.scene, specs[0].primitive_id
        )
        if not bool(selected_summary["accepted"]):
            raise RuntimeError(f"{spec.key}: selected primitive failed {selected_summary['failed_checks']}")
        if spec.key != "wide" and bool(nominal_summary["accepted"]):
            raise RuntimeError(f"{spec.key}: nominal-walk counterfactual unexpectedly passed")

        scenario_dir = args.out / spec.key
        scenario_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(scenario_dir / "selected_executed.npz", **execution)
        np.savez_compressed(scenario_dir / "nominal_counterfactual_executed.npz", **nominal_execution)
        (scenario_dir / "selected_summary.json").write_text(
            json.dumps(selected_summary, indent=2) + "\n"
        )
        (scenario_dir / "nominal_counterfactual_summary.json").write_text(
            json.dumps(nominal_summary, indent=2) + "\n"
        )
        row = {
            "key": spec.key,
            "title": spec.title,
            "condition_source_index": spec.source_index,
            "scene": str(spec.scene),
            "selected_primitive_id": spec.primitive_id,
            "selected_primitive": PRIMITIVE_NAMES[spec.primitive_id],
            "router_confidence": confidence,
            "probabilities": ranking,
            "geometry_only_router_confidence": geometry_confidence,
            "geometry_only_probabilities": geometry_ranking,
            "selected": selected_summary,
            "nominal_walk_counterfactual": nominal_summary,
            "sample": str(spec.sample),
        }
        report["scenarios"].append(row)
        results.append(ScenarioResult(spec, sample, execution, selected_summary,
                                      nominal_summary, confidence, ranking,
                                      geometry_confidence, geometry_ranking))
    (args.out / "behavior_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return results


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _add_sphere(scene: mujoco.MjvScene, position: np.ndarray, radius: float,
                color: tuple[float, float, float, float]) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0.0, 0.0]), position,
                        np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
    scene.ngeom += 1


def _add_ellipsoid(scene: mujoco.MjvScene, center: np.ndarray, semi: np.ndarray,
                   yaw: float) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                         [np.sin(yaw), np.cos(yaw), 0.0],
                         [0.0, 0.0, 1.0]])
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                        np.asarray(semi, dtype=np.float64), center,
                        rotation.ravel(), np.array([0.72, 0.28, 1.0, 0.055], dtype=np.float32))
    scene.ngeom += 1


def _draw_route(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int],
                positions: np.ndarray, tick: int) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=7, fill=(19, 25, 34), outline=(72, 84, 102), width=2)
    xy = positions[:, :2] - positions[0, :2]
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(upper - lower, 0.24)
    center = 0.5 * (lower + upper)
    lower, upper = center - 0.65 * span, center + 0.65 * span
    normalized = (xy - lower) / np.maximum(upper - lower, 1e-6)
    pixels = [(int(left + 10 + p[0] * (right - left - 20)),
               int(bottom - 9 - p[1] * (bottom - top - 22))) for p in normalized]
    upto = pixels[:tick + 1]
    if len(upto) > 1:
        draw.line(upto, fill=(255, 155, 38), width=4, joint="curve")
    draw.ellipse((pixels[0][0] - 4, pixels[0][1] - 4,
                  pixels[0][0] + 4, pixels[0][1] + 4), fill=(240, 240, 240))
    current = pixels[min(tick, len(pixels) - 1)]
    draw.ellipse((current[0] - 5, current[1] - 5,
                  current[0] + 5, current[1] + 5), fill=(255, 155, 38))


def render(results: list[ScenarioResult], args: argparse.Namespace) -> None:
    panel_width, scene_height = args.panel_width, args.scene_height
    header_height, footer_height = 116, 126
    panel_height = header_height + scene_height + footer_height
    frame_count = min(len(result.execution["q_exec"]) for result in results)
    environments, renderers, cameras, options = [], [], [], []
    for result in results:
        env = G1FlatEnv(result.spec.scene)
        renderer = mujoco.Renderer(env.model, height=scene_height, width=panel_width)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        camera.distance, camera.azimuth, camera.elevation = 2.45, 125.0, -10.0
        option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(option)
        environments.append(env)
        renderers.append(renderer)
        cameras.append(camera)
        options.append(option)

    title_font, body_font, small_font = _font(18), _font(15), _font(13)
    frames: list[Image.Image] = []
    try:
        for tick in range(frame_count):
            canvas = Image.new("RGB", (panel_width * len(results), panel_height), (10, 14, 21))
            for index, result in enumerate(results):
                env, renderer = environments[index], renderers[index]
                execution = result.execution
                env.data.qpos[:3] = execution["base_pos"][tick]
                env.data.qpos[3:7] = execution["base_quat"][tick]
                env.data.qpos[env.body_qadr] = execution["q_exec"][tick][C.ISAACLAB_TO_MUJOCO]
                env.data.qpos[env.hand_qadr] = 0.0
                env.data.qvel[:] = 0.0
                mujoco.mj_forward(env.model, env.data)
                renderer.update_scene(env.data, camera=cameras[index], scene_option=options[index])
                scene = renderer.scene
                for point in execution["base_pos"][max(0, tick - 100):tick:3]:
                    _add_sphere(scene, point, 0.014, (1.0, 0.55, 0.1, 0.82))
                corridor = _resample_corridor(result.sample["condition_corridor"], frame_count)[tick]
                origin = execution["base_pos"][0]
                center = corridor[:3] + origin
                _add_ellipsoid(scene, center, corridor[3:6], float(corridor[6]))
                image = Image.fromarray(np.asarray(renderer.render()), mode="RGB")
                panel = Image.new("RGB", (panel_width, panel_height), (13, 18, 26))
                panel.paste(image, (0, header_height))
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, panel_width, header_height), fill=(17, 23, 32))
                draw.text((12, 8), result.spec.title, font=title_font, fill=(245, 247, 250))
                draw.text((12, 35),
                          f"router M,c: {PRIMITIVE_NAMES[result.spec.primitive_id]}  "
                          f"{100.0 * result.confidence:.1f}%  |  M-only {100.0 * result.geometry_confidence:.1f}%",
                          font=body_font, fill=(112, 220, 255))
                draw.text((12, 58),
                          f"selected physics: PASS   pelvis z={float(result.selected_summary['base_z_mean_m']):.3f} m",
                          font=body_font, fill=(88, 236, 124))
                nominal_pass = bool(result.nominal_summary["accepted"])
                nominal_text = "PASS" if nominal_pass else "REJECT"
                nominal_color = (88, 236, 124) if nominal_pass else (255, 92, 92)
                checks = ", ".join(result.nominal_summary["failed_checks"][:2]) or "none"
                draw.text((12, 81), f"unchanged walk: {nominal_text}  [{checks}]",
                          font=small_font, fill=nominal_color)
                footer_top = header_height + scene_height
                draw.rectangle((0, footer_top, panel_width, panel_height), fill=(14, 19, 27))
                displacement = execution["base_pos"][-1, :2] - execution["base_pos"][0, :2]
                draw.text((12, footer_top + 7),
                          f"executed route  dx={displacement[0]:+.2f} m  dy={displacement[1]:+.2f} m",
                          font=small_font, fill=(232, 235, 240))
                _draw_route(draw, (10, footer_top + 30, panel_width - 10, panel_height - 8),
                            execution["base_pos"], tick)
                canvas.paste(panel, (index * panel_width, 0))
            frames.append(canvas)
    finally:
        for renderer in renderers:
            renderer.close()
    out = args.out / "manifold_behavior.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(round(1000.0 / args.fps)), loop=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="small-amplitude manifold-conditioned behavior demo")
    parser.add_argument("--windows", type=Path,
                        default=C.REPO / "reports/manifold_motion/seed_windows_corridor_stage2_v2/seed_stage2_windows.npz")
    parser.add_argument("--router", type=Path,
                        default=C.REPO / "reports/manifold_motion/primitive_router_v1/router.pt")
    parser.add_argument("--geometry-router", type=Path,
                        default=C.REPO / "reports/manifold_motion/primitive_router_geometry_v1/router.pt")
    parser.add_argument("--out", type=Path,
                        default=C.REPO / "reports/manifold_motion/stage2_manifold_behavior_v1")
    parser.add_argument("--panel-width", type=int, default=440)
    parser.add_argument("--scene-height", type=int, default=330)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    if min(args.panel_width, args.scene_height, args.fps) <= 0:
        parser.error("panel width, scene height, and fps must be positive")
    results = build(args)
    render(results, args)
    print(json.dumps({
        "out": str(args.out / "manifold_behavior.gif"),
        "report": str(args.out / "behavior_report.json"),
        "scenarios": [{
            "key": result.spec.key,
            "selected": PRIMITIVE_NAMES[result.spec.primitive_id],
            "confidence": result.confidence,
            "geometry_only_confidence": result.geometry_confidence,
            "accepted": result.selected_summary["accepted"],
            "nominal_counterfactual_accepted": result.nominal_summary["accepted"],
        } for result in results],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
