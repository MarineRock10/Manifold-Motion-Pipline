"""Motion capability from recorded walking: which corridors can be walked, and how fast (Phase 2.0).

Phase 1 data has no `M -> z_p` supervision: the manifold was extracted *from* the motion, so a
sample says "when the command was c, this envelope happened", not "this envelope was demanded".
What the data can answer is the inverse question, which is the one Stage 1 actually needs:

    given a corridor (clearance, half-width), was this motion feasible, and how fast was it?

Feasibility is containment of the recorded body envelope inside the corridor, in the frame of
travel: the envelope's extent perpendicular to the direction of travel must fit the corridor's
half-width, and the top of the body must stay under the clearance. Both come straight from the
recorded landmarks/envelope, so this is a re-reading of Phase 1 data, not a new experiment.

    python3 -m manifold_g1.capability build      # -> reports/manifold_g1/dataset/capability.json
    python3 -m manifold_g1.capability show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C

DATASET = Path("reports/manifold_g1/dataset")
DEFAULT_CLEARANCE = (1.20, 1.45)      # tunnel height above the floor [m]
DEFAULT_HALF_WIDTH = (0.35, 0.80)     # corridor half-width [m]
GRID = (9, 9)                         # cells per axis


def _yaw(quat: np.ndarray) -> np.ndarray:
    return np.arctan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                      1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))


def tick_features(index: dict, episode: dict) -> dict:
    """Per-tick features that decide corridor feasibility: speed, envelope, travel frame."""
    data = np.load(Path(index["clips_dir"]) / episode["file"])
    semi = data["envelope_semi"]                     # [E, 3] pelvis-anchored half-extents
    ticks = len(data["t"])
    stride = max(1, ticks // len(semi))
    idx = np.minimum(np.arange(ticks) // stride, len(semi) - 1)

    vel = data["base_lin_vel"][:, :2]
    speed = np.linalg.norm(vel, axis=1)
    travel = np.arctan2(vel[:, 1], vel[:, 0])        # direction of motion, world frame
    heading = _yaw(data["base_quat"])                # body x axis, world frame
    turn = np.abs((travel - heading + np.pi / 2) % np.pi - np.pi / 2)   # angle between them

    env = semi[idx]
    top = data["envelope_top"][idx]
    # axis-aligned envelope rotated by `turn`: the extent perpendicular to the travel direction
    half_width = env[:, 0] * np.abs(np.sin(turn)) + env[:, 1] * np.abs(np.cos(turn))
    return {"speed": speed, "half_width": half_width, "top": top,
            "turn_deg": np.degrees(turn), "ticks": ticks}


def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    """Max over `window` ticks; the result is aligned to the window start."""
    if len(values) < window:
        return np.full(0, np.inf)
    return np.lib.stride_tricks.sliding_window_view(values, window).max(axis=1)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full(0, np.nan)
    return np.lib.stride_tricks.sliding_window_view(values, window).mean(axis=1)


def build(args) -> int:
    index = json.loads((args.dataset / "index.json").read_text())
    speed, half_width, top, turn = [], [], [], []
    sequences = []
    for episode in index["episodes"]:
        features = tick_features(index, episode)
        sequences.append(features)
        speed.append(features["speed"])
        half_width.append(features["half_width"])
        top.append(features["top"])
        turn.append(features["turn_deg"])
    speed = np.concatenate(speed)
    half_width = np.concatenate(half_width)
    top = np.concatenate(top)
    turn = np.concatenate(turn)
    moving = speed > 0.15        # ticks that are actually travelling (not turning in place)

    # Walking through a corridor means the *whole* body is inside it for a stretch of time, so
    # feasibility is judged over a sliding window ("all ticks fit"), never per tick: a single
    # tick only says the body happened to be low/narrow at that instant, which biases any
    # statistic taken over instantaneous fits.
    window = int(round(args.window / C.CONTROL_DT))
    clearances = np.linspace(*args.clearance, GRID[0])
    widths = np.linspace(*args.half_width, GRID[1])
    table = np.full((len(clearances), len(widths)), np.nan)
    counts = np.zeros_like(table, dtype=int)
    for i, clearance in enumerate(clearances):
        for j, width in enumerate(widths):
            speeds = []
            for seq in sequences:
                top_max = _rolling_max(seq["top"], window)
                hw_max = _rolling_max(seq["half_width"], window)
                ok = (top_max <= clearance) & (hw_max <= width)
                if not ok.any():
                    continue
                mean_speed = _rolling_mean(seq["speed"], window)[ok]
                # a sustained walk also needs to be moving; a stationary window is not a walk
                speeds.append(mean_speed[mean_speed > 0.15])
            if speeds and sum(len(v) for v in speeds) >= 10:
                allspeeds = np.concatenate(speeds)
                # how fast this corridor can be walked: the fastest sustained window
                table[i, j] = float(np.percentile(allspeeds, 95))
                counts[i, j] = len(allspeeds)

    print(f"{len(sequences)} episodes, {len(speed)} ticks; envelope top p5-p95 "
          f"{np.percentile(top, 5):.3f}-{np.percentile(top, 95):.3f} m; "
          f"sustained window {window} ticks ({args.window:.1f} s), all ticks must fit")
    print(f"\nsustained walkable speed [m/s] (p95 over feasible windows)")
    header = "  clearance \\ half-width " + " ".join(f"{w:5.2f}" for w in widths)
    print(header)
    for i, clearance in enumerate(clearances):
        cells = " ".join("  -- " if np.isnan(v) else f"{v:5.2f}" for v in table[i])
        print(f"  {clearance:20.2f} {cells}")
    print("\nfeasible windows per cell (a cell needs >=10 windows to be reported)")
    print(header)
    for i, clearance in enumerate(clearances):
        print(f"  {clearance:20.2f} " + " ".join(f"{c:5d}" for c in counts[i]))

    result = {
        "clearance": clearances.tolist(), "half_width": widths.tolist(),
        "speed_mps": table.tolist(), "samples": counts.tolist(),
        "envelope_top": {"p5": float(np.percentile(top, 5)), "p50": float(np.percentile(top, 50)),
                         "p95": float(np.percentile(top, 95)), "max": float(top.max())},
        "half_width_dist": {"p5": float(np.percentile(half_width, 5)),
                            "p50": float(np.percentile(half_width, 50)),
                            "p95": float(np.percentile(half_width, 95)),
                            "max": float(half_width.max())},
        "speed_when_fitting": {"p5": float(np.percentile(speed[moving], 5)),
                               "p50": float(np.percentile(speed[moving], 50)),
                               "p95": float(np.percentile(speed[moving], 95))},
        "window_s": args.window,
        "walkable_speed_stat": "p95 of mean speed over windows where every tick fits",
        "turn_when_moving_deg": {"p50": float(np.percentile(turn[moving], 50)),
                                 "p95": float(np.percentile(turn[moving], 95))},
    }
    out = args.dataset / "capability.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nsaved {out}")
    return 0


def show(args) -> int:
    result = json.loads((args.dataset / "capability.json").read_text())
    print(json.dumps(result, indent=2)[:1500])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Corridor capability from recorded walking")
    parser.add_argument("mode", choices=("build", "show"))
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--clearance", type=float, nargs=2, default=DEFAULT_CLEARANCE)
    parser.add_argument("--half-width", type=float, nargs=2, default=DEFAULT_HALF_WIDTH)
    parser.add_argument("--window", type=float, default=1.0,
                        help="sustained feasibility window in seconds")
    args = parser.parse_args()
    return build(args) if args.mode == "build" else show(args)


if __name__ == "__main__":
    raise SystemExit(main())
