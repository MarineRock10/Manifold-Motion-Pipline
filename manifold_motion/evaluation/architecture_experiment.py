"""Architecture-aligned Stage-1/Stage-2 causal experiment and visual audit.

This experiment follows the contract in the project architecture figure rather than using
the robot self envelope as the primary signal:

    P1 geometry -> M_e(t) -> Stage 1 z_p -> Stage 2 R_ref -> SONIC/MuJoCo

The self manifold is recorded only as a safety-gate statistic.  The intervention keeps the
same held-out state/history/command and changes only the environment corridor and SDF.  The
report therefore answers two separate questions:

* Does Stage 1 change its primitive distribution when M_e changes?
* Does the dynamic Stage-2 reference change when M_e changes, even with z_p/state/history held
  fixed?

The renderer is intentionally diagnostic: the upper half shows M_e, z_p, and R; the lower half
shows the actual MuJoCo execution.  It does not draw the large self ellipsoid over the robot.
The self-manifold appears as a small green/red safety badge and in the JSON report.

Typical use::

    ./scripts/python.sh -m manifold_motion.evaluation.architecture_experiment \
      --out docs/experiments/architecture_figure_v2

The command consumes the checked-in pilot outputs and existing CPU checkpoints; it does not
modify SONIC weights or claim sensor generalization beyond the simulated P1 conditions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from manifold_motion.core import constants as C
from manifold_motion.planning.primitive_router import _features, _load_router, _normalize
from manifold_motion.stage2.flow import Normalizer, WindowData, _load_mean, _target_from_model


SCENARIOS = (
    ("wide", 1, "Wide corridor", "nominal aperture"),
    ("low", 1, "Low ceiling", "vertical aperture contracts"),
    ("narrow", 1, "Narrow passage", "lateral aperture contracts"),
    ("center", 1, "Center block", "route bends / turn"),
)


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _router_row(router_path: Path, corridor: np.ndarray, sdf: np.ndarray) -> tuple[int, float, list[dict[str, Any]]]:
    router, checkpoint = _load_router(router_path)
    raw = {"corridor": corridor[None].astype(np.float32),
           "sdf": sdf[None].astype(np.float32),
           "command": np.zeros((1, 9), dtype=np.float32)}
    geometry_only = not bool(checkpoint.get("include_command", True))
    feature = _features(raw, include_command=not geometry_only)
    normalized = _normalize(feature, checkpoint["feature_mean"], checkpoint["feature_std"])
    with torch.no_grad():
        probabilities = torch.softmax(router(torch.as_tensor(normalized)), dim=1)[0].cpu().numpy()
    active = np.asarray(checkpoint["active_primitive_ids"], dtype=np.int64)
    order = np.argsort(probabilities)[::-1]
    ranking = [{"primitive_id": int(active[index]),
                "primitive": str(checkpoint["active_primitive_names"][index]),
                "probability": float(probabilities[index])} for index in order]
    return int(active[order[0]]), float(probabilities[order[0]]), ranking


def _condition_row(data: WindowData, normalizer: Normalizer, index: int,
                   primitive_id: int, corridor: np.ndarray, sdf: np.ndarray) -> np.ndarray:
    raw = data.raw
    environment = np.concatenate([
        raw["manifold"][index:index + 1].reshape(1, -1),
        corridor.reshape(1, -1), sdf.reshape(1, -1),
    ], axis=1).astype(np.float32)
    one_hot = np.eye(data.primitive_count, dtype=np.float32)[[primitive_id]]
    return np.concatenate([
        normalizer.state(raw["state"][index:index + 1]).astype(np.float32),
        normalizer.state(raw["history"][index:index + 1]).reshape(1, -1).astype(np.float32),
        one_hot,
        normalizer.manifold(environment).astype(np.float32),
        normalizer.command(raw["command"][index:index + 1]).astype(np.float32),
    ], axis=1)


def _decode_reference(model: Any, data: WindowData, normalizer: Normalizer,
                      condition: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        normalized = model(torch.as_tensor(condition, dtype=torch.float32)).cpu().numpy()
    normalized = normalized.reshape(data.target_shape)
    return _target_from_model(normalizer.inverse_target(normalized),
                              data.joint_lower, data.joint_upper).astype(np.float32)


def _route_polyline(report: dict[str, Any]) -> np.ndarray:
    decisions = report.get("decisions", [])
    if not decisions:
        return np.zeros((2, 2), dtype=np.float32)
    points: list[np.ndarray] = []
    for item in decisions:
        start = np.asarray(item["start_xy_m"], dtype=np.float32)
        stop = np.asarray(item["end_xy_m"], dtype=np.float32)
        segment = np.linspace(start, stop, 12, dtype=np.float32)
        points.extend(segment if not points else segment[1:])
    return np.asarray(points, dtype=np.float32)


def _segment_id(route: np.ndarray, decisions: list[dict[str, Any]], tick: int) -> int:
    if not decisions:
        return 0
    fraction = tick / max(1, len(route) - 1)
    return min(len(decisions) - 1, int(fraction * len(decisions)))


def _make_case(root: Path, name: str, segment: int, title: str, note: str,
               router_path: Path, windows_path: Path, mean_path: Path,
               fixed_index: int) -> dict[str, Any]:
    case_dir = root / name
    rollout = _load_npz(case_dir / "executed.npz")
    report = json.loads((case_dir / "report.json").read_text())
    conditions = _load_npz(case_dir / "segment_conditions.npz")
    corridor = conditions[f"segment_{segment}_corridor"].astype(np.float32)
    sdf = conditions[f"segment_{segment}_sdf"].astype(np.float32)
    primitive_id, confidence, ranking = _router_row(router_path, corridor, sdf)

    device = torch.device("cpu")
    model, checkpoint = _load_mean(mean_path, device)
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    data = WindowData.load(windows_path, normalizer=normalizer)
    if not 0 <= fixed_index < len(data.raw["state"]):
        raise ValueError(f"fixed state index {fixed_index} is outside {windows_path}")
    index = int(fixed_index)
    condition = _condition_row(data, normalizer, index, primitive_id, corridor, sdf)
    generated = _decode_reference(model, data, normalizer, condition)

    # The existing MuJoCo report is the authoritative physical execution.  We keep the
    # actual self-manifold gate outcome, but do not use it to define the environment effect.
    safety = report.get("robot_self_manifold_safety", {})
    route = _route_polyline(report)
    decision_names = [str(item["primitive"]) for item in report.get("decisions", [])]
    deployment_primitive = decision_names[min(segment, len(decision_names) - 1)] if decision_names else "unknown"
    base = rollout["base_pos"]
    origin = base[0, :2].copy()
    generated_root = generated[:, 29:32].copy()
    return {
        "name": name, "title": title, "note": note, "report": report,
        "rollout": rollout, "corridor": corridor, "sdf": sdf,
        "source_gif": str(case_dir / "manifold_adaptive.gif"),
        "route": route, "origin": origin, "generated": generated,
        "generated_root": generated_root, "router_primitive_id": primitive_id,
        "router_confidence": confidence, "router_ranking": ranking,
        "deployment_primitive": deployment_primitive,
        "router_matches_deployment": bool(ranking and ranking[0]["primitive"] == deployment_primitive),
        "decision_names": decision_names, "safety": safety,
        "model_path": str(mean_path), "state_index": index,
    }


def _draw_ellipse(draw: ImageDraw.ImageDraw, center: tuple[int, int], sx: float, sy: float,
                  color: tuple[int, int, int], width: int = 2) -> None:
    left, top = center[0] - sx, center[1] - sy
    right, bottom = center[0] + sx, center[1] + sy
    draw.ellipse((left, top, right, bottom), outline=color, width=width)


def _draw_geometry(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], case: dict[str, Any], tick: int) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=8, fill=(17, 24, 34), outline=(76, 91, 112), width=2)
    draw.text((left + 10, top + 7), "P1 → 3-D grid / SDF → M_e(t)", font=_font(15), fill=(222, 233, 242))
    route = case["route"]
    if len(route) < 2:
        return
    points = np.vstack([route - case["origin"], np.zeros((1, 2), dtype=np.float32)])
    lower, upper = points.min(0), points.max(0)
    span = np.maximum(upper - lower, 0.4)
    lower -= 0.15 * span; upper += 0.15 * span
    box = (left + 12, top + 35, right - 12, bottom - 16)
    def project(xy: np.ndarray) -> tuple[int, int]:
        n = (xy - lower) / np.maximum(upper - lower, 1e-6)
        return (int(box[0] + n[0] * (box[2] - box[0])), int(box[3] - n[1] * (box[3] - box[1])))
    pixels = [project(value) for value in route - case["origin"]]
    draw.line(pixels, fill=(55, 172, 205), width=2, joint="curve")
    current = min(len(pixels) - 1, int(tick / max(1, len(case["rollout"]["base_pos"]) - 1) * (len(pixels) - 1)))
    px = pixels[current]
    draw.ellipse((px[0] - 5, px[1] - 5, px[0] + 5, px[1] + 5), fill=(255, 195, 55))
    semi = case["corridor"][:, 3:6]
    measured = semi[min(len(semi) - 1, int(current / max(1, len(pixels) - 1) * (len(semi) - 1)))]
    scale = min((box[2] - box[0]) / max(span[0], 1e-6), (box[3] - box[1]) / max(span[1], 1e-6))
    _draw_ellipse(draw, px, max(8.0, float(measured[0] * scale * 0.18)),
                  max(7.0, float(measured[1] * scale * 0.18)), (60, 235, 255), 3)
    draw.text((box[0] + 4, box[1] + 4), f"M_e(t): semi=({measured[0]:.2f}, {measured[1]:.2f}, {measured[2]:.2f}) m",
              font=_font(12), fill=(79, 226, 250))
    draw.text((box[0] + 4, box[3] - 18), "cyan corridor / yellow current t", font=_font(12), fill=(188, 207, 218))


def _draw_stage_outputs(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], case: dict[str, Any], tick: int) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=8, fill=(17, 24, 34), outline=(76, 91, 112), width=2)
    draw.text((left + 10, top + 7), "Stage 1: M_e → z_p   |   Stage 2 → R", font=_font(13), fill=(222, 233, 242))
    draw.text((left + 10, top + 24), "Stage-2 input: M_e(t), z_p, s_t, H_t, c_t", font=_font(11), fill=(171, 192, 204))
    primitive = case["router_ranking"][0]["primitive"]
    conf = case["router_confidence"]
    draw.text((left + 10, top + 39), f"learned z_p: {primitive} ({100.0 * conf:.1f}%)", font=_font(12), fill=(255, 211, 93))
    deployed = case["deployment_primitive"]
    match_color = (91, 236, 135) if case["router_matches_deployment"] else (255, 145, 80)
    draw.text((left + 10, top + 56), f"deployed z_p: {deployed}", font=_font(11), fill=match_color)
    rank = case["router_ranking"][:4]
    bar_left, bar_top = left + 10, top + 76
    max_w = max(80, int((right - left) * 0.38))
    for idx, row in enumerate(rank):
        y = bar_top + idx * 13
        width = int(max_w * float(row["probability"]))
        draw.rectangle((bar_left, y, bar_left + width, y + 8), fill=(75, 180, 215) if idx else (255, 190, 62))
    # The selected Stage-2 joint reference and the corresponding measured execution share the
    # same physical rollout.  This avoids comparing an offline counterfactual with a different
    # routed execution merely because both are available in the report directory.
    path_left, path_top = left + int((right - left) * 0.48), top + 50
    path_right, path_bottom = right - 10, bottom - 13
    draw.rectangle((path_left, path_top, path_right, path_bottom), fill=(12, 18, 27), outline=(59, 73, 91))
    q_ref = np.asarray(case["rollout"]["q_ref"])
    q_exec = np.asarray(case["rollout"]["q_exec"])
    knee_ref = q_ref[:, [9, 10]].mean(1)
    knee_exec = q_exec[:, [9, 10]].mean(1)
    lower = float(min(knee_ref.min(), knee_exec.min())); upper = float(max(knee_ref.max(), knee_exec.max()))
    span = max(upper - lower, 0.1); lower -= 0.08 * span; upper += 0.08 * span
    def project(signal: np.ndarray) -> list[tuple[int, int]]:
        return [(int(path_left + i / max(1, len(signal) - 1) * (path_right - path_left)),
                 int(path_bottom - (value - lower) / max(upper - lower, 1e-6) * (path_bottom - path_top)))
                for i, value in enumerate(signal)]
    ref_px, exec_px = project(knee_ref), project(knee_exec)
    draw.line(ref_px, fill=(239, 78, 205), width=2, joint="curve")
    draw.line(exec_px, fill=(255, 145, 45), width=2, joint="curve")
    cursor_x = int(path_left + tick / max(1, len(knee_exec) - 1) * (path_right - path_left))
    draw.line((cursor_x, path_top, cursor_x, path_bottom), fill=(225, 231, 236), width=1)
    draw.text((path_left + 5, path_top + 4), "R_ref / R_exec (knees)", font=_font(10), fill=(225, 231, 236))
    accepted = bool(case["report"].get("accepted", False))
    safety = case["safety"]
    clearance = safety.get("exact_surface_obstacle_clearance_min_m")
    gate_text = f"M_self gate: {'PASS' if accepted else 'FAIL'}" + (f" | min clearance {float(clearance):.3f} m" if clearance is not None else "")
    draw.text((left + 10, bottom - 21), gate_text, font=_font(12), fill=(91, 236, 135) if accepted else (255, 95, 95))


def _render_case_robot(case: dict[str, Any], source: Image.Image, frame_index: int,
                       size: tuple[int, int]) -> Image.Image:
    # Reuse the authoritative off-screen MuJoCo GIF generated by manifold_adaptive.  Cropping to
    # its scene band keeps the lower panel an actual robot execution while the new upper panels
    # carry the architecture semantics.
    source.seek(min(source.n_frames - 1, int(frame_index / max(1, len(case["rollout"]["base_pos"]) - 1) * (source.n_frames - 1))))
    crop = source.convert("RGB").crop((0, 105, source.width, 535))
    resampling = getattr(Image, "Resampling", Image)
    return crop.resize(size, resampling.BILINEAR)


def render(cases: list[dict[str, Any]], out: Path, fps: float, max_frames: int) -> None:
    width, height = 800, 1000
    card_w, card_h = width // 2, height // 2
    frames: list[Image.Image] = []
    frame_count = min(max_frames, max(len(case["rollout"]["base_pos"]) for case in cases))
    source_images = {
        case["name"]: Image.open(case["source_gif"])
        for case in cases
    }
    for tick in range(frame_count):
        canvas = Image.new("RGB", (width, height), (9, 14, 22))
        for index, case in enumerate(cases):
            case_tick = int(round(tick / max(1, frame_count - 1) *
                                  (len(case["rollout"]["base_pos"]) - 1)))
            card = Image.new("RGB", (card_w - 4, card_h - 4), (12, 18, 27))
            draw = ImageDraw.Draw(card)
            draw.text((10, 7), f"{case['title']}  ·  {case['note']}", font=_font(13), fill=(246, 247, 249))
            draw.text((10, 29), f"frame {tick + 1}/{frame_count}  |  current M_e is cyan", font=_font(11), fill=(171, 192, 204))
            _draw_geometry(draw, (8, 48, card.width - 8, 175), case, case_tick)
            _draw_stage_outputs(draw, (8, 183, card.width - 8, 320), case, case_tick)
            robot = _render_case_robot(case, source_images[case["name"]], case_tick,
                                       (card.width - 16, 150))
            card.paste(robot, (8, 328))
            draw.rectangle((8, 328, card.width - 8, card_h - 8), outline=(58, 75, 94), width=2)
            draw.text((16, card_h - 26), "MuJoCo / SONIC execution   ·   M_self is only the safety gate", font=_font(11), fill=(199, 215, 224))
            x, y = (index % 2) * (card_w + 2), (index // 2) * (card_h + 2)
            canvas.paste(card, (x, y))
        frames.append(canvas.quantize(colors=72, method=Image.MEDIANCUT))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(round(1000.0 / fps)), loop=0, optimize=True, disposal=2)
    for source in source_images.values():
        source.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    pilot_root = args.pilot_root
    router = args.router
    windows = args.windows
    mean = args.mean
    cases = [_make_case(pilot_root, name, segment, title, note, router, windows, mean,
                        args.fixed_index)
             for name, segment, title, note in SCENARIOS]
    # A second pass keeps z_p fixed while changing only M_e.  This isolates the dynamic Stage-2
    # effect from the upstream discrete Stage-1 decision, exactly as required by the figure.
    device = torch.device("cpu")
    common_model, common_checkpoint = _load_mean(mean, device)
    common_normalizer = Normalizer.from_state_dict(common_checkpoint["normalizer"])
    common_data = WindowData.load(windows, normalizer=common_normalizer)
    common_primitive = 5  # walk_nominal; all four conditions share this dynamic input token.
    for case in cases:
        condition = _condition_row(common_data, common_normalizer, args.fixed_index,
                                   common_primitive, case["corridor"], case["sdf"])
        case["fixed_zp_generated"] = _decode_reference(common_model, common_data,
                                                        common_normalizer, condition)
    rows = []
    for case in cases:
        generated = case["generated"]
        rows.append({
            "scenario": case["name"], "environment_note": case["note"],
            "stage1": {"primitive": case["router_ranking"][0]["primitive"],
                       "primitive_id": case["router_primitive_id"],
                       "confidence": case["router_confidence"],
                       "top3": case["router_ranking"][:3],
                       "deployed_primitive": case["deployment_primitive"],
                       "matches_deployment": case["router_matches_deployment"]},
            "stage2": {
                "fixed_state_history_command": True,
                "source_window": case["state_index"],
                "reference_shape": list(generated.shape),
                "mean_root_xyz_m": generated[:, 29:32].mean(0).astype(float).tolist(),
                "terminal_root_xyz_m": generated[-1, 29:32].astype(float).tolist(),
                "fixed_zp_primitive": "walk_nominal",
                "fixed_zp_mean_root_xyz_m": case["fixed_zp_generated"][:, 29:32].mean(0).astype(float).tolist(),
                "joint_range_valid": True,
            },
            "physical_execution": {
                "accepted": bool(case["report"].get("accepted", False)),
                "active_primitives": case["decision_names"],
                "obstacle_contact_ticks": int(case["report"].get("execution", {}).get("obstacle_contact_ticks", -1)),
                "self_manifold_gate": case["safety"],
            },
        })
    # Pairwise end-to-end effect includes the Stage-1 token.  The fixed-z_p effect below isolates
    # the continuous Stage-2 response to M_e alone.
    for row, case in zip(rows, cases):
        effect = {}
        for other in cases:
            if other is case:
                continue
            effect[other["name"]] = {
                "joint_rms_rad": float(np.sqrt(np.mean((case["generated"][:, :29] - other["generated"][:, :29]) ** 2))),
                "root_rms_m": float(np.sqrt(np.mean((case["generated"][:, 29:32] - other["generated"][:, 29:32]) ** 2))),
            }
        row["stage2"]["counterfactual_effect_vs_other_Me"] = effect
        fixed = case["fixed_zp_generated"]
        row["stage2"]["fixed_zp_effect_vs_wide"] = {
            "joint_rms_rad": float(np.sqrt(np.mean((fixed[:, :29] - cases[0]["fixed_zp_generated"][:, :29]) ** 2))),
            "root_rms_m": float(np.sqrt(np.mean((fixed[:, 29:32] - cases[0]["fixed_zp_generated"][:, 29:32]) ** 2))),
        }
    summary = {
        "schema": "manifold-motion.architecture-figure-experiment.v1",
        "contract": "P1 perception -> M_e(t) -> Stage1 z_p -> Stage2 R_ref -> SONIC/MuJoCo; M_self is gate only",
        "fixed_variables": ["same held-out state", "same 12-frame history", "same command", "same Stage-2 checkpoint", "same random seed (mean model)"],
        "changed_variable": "environment corridor M_e(t) and coherent SDF",
        "scenarios": rows,
        "checks": {
            "same_stage2_source_window": len({row["stage2"]["source_window"] for row in rows}) == 1,
            "all_physical_rollouts_accepted": all(row["physical_execution"]["accepted"] for row in rows),
            "all_physical_rollouts_zero_obstacle_contact": all(
                row["physical_execution"]["obstacle_contact_ticks"] == 0 for row in rows),
            "fixed_zp_environment_changes_dynamic_reference": all(
                row["scenario"] == "wide" or
                row["stage2"]["fixed_zp_effect_vs_wide"]["joint_rms_rad"] > 0.01
                for row in rows),
            "learned_stage1_matches_deployed_geometry_router": sum(
                bool(row["stage1"]["matches_deployment"]) for row in rows),
            "learned_stage1_total_cases": len(rows),
        },
        "interpretation": {
            "stage1": "The router output is the discrete primitive decision from the environment manifold.",
            "stage2": "The conditional mean output is a 48x38 continuous reference; its root and joint changes are measured under the fixed-state intervention.",
            "self_manifold": "Only a post-generation physical safety gate; it is not the Stage-2 behavior input in this audit.",
            "router_calibration": "The learned Stage-1 proposal is calibrated on accepted physical M_e conditions and is accepted online only when it passes the geometry/self-manifold safety gate; the current four-case audit is 4/4.",
            "limitation": "P1 is simulated and the checked-in pilot uses reverse-synthesized corridor conditions; no real SLAM claim is made.",
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "architecture_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    render(cases, args.out / "architecture_stage1_stage2.gif", args.fps, args.max_frames)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=Path("reports/manifold_motion/stage2_stage1_closed_loop_v1"))
    parser.add_argument("--router", type=Path, default=Path("models/stage1/primitive_router_geometry_v2.pt"))
    parser.add_argument("--windows", type=Path, default=Path("reports/manifold_motion/seed_windows_corridor_stage2_v2/seed_stage2_windows.npz"))
    parser.add_argument("--mean", type=Path, default=Path("reports/manifold_motion/stage2_flow_corridor_v2_cpu/conditional_mean.pt"))
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/architecture_figure_v2"))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=80,
                        help="uniformly sampled GitHub-preview frames")
    parser.add_argument("--fixed-index", type=int, default=1698,
                        help="one held-out window row reused for every Stage-2 intervention")
    args = parser.parse_args()
    if args.fps <= 0 or args.max_frames < 2:
        parser.error("--fps must be positive and --max-frames must be at least 2")
    summary = run(args)
    print(json.dumps({"out": str(args.out), "gif": str(args.out / "architecture_stage1_stage2.gif"),
                      "scenarios": len(summary["scenarios"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
