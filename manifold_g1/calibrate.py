"""Command -> achieved-motion calibration for the walking primitive (Phase 1.1).

`planner_sonic.onnx` takes a motion command (`target_vel`, `movement`, `facing`, `mode`) and
emits a gait, which the frozen SONIC controller then executes. What the robot actually does is
a property of the controller, not of the command, so the command space has to be measured
before anything can sample it.

    python3 -m manifold_g1.calibrate sweep     # record the grid (one clip per command)
    python3 -m manifold_g1.calibrate table     # aggregate what is on disk
    python3 -m manifold_g1.calibrate plot      # optional: speed vs command

The table is the first slice of the robot's motion-capability manifold `M^R`: it says which
commands produce motion at all, how fast, and how far the body envelope moves.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CLIP_DIR = Path("reports/manifold_g1/clips")

MODES = list(range(1, 28))          # planner_sonic.onnx exposes 27 motion modes

# (tag, extra CLI args) - the grid Phase 1 samples from
SWEEP = [
    ("cal_v-1.0_fwd", ["--target-vel", "-1.0"]),
    ("cal_v+0.6_fwd", ["--target-vel", "0.6"]),
    ("cal_v+0.8_fwd", ["--target-vel", "0.8"]),
    ("cal_v+1.0_fwd", ["--target-vel", "1.0"]),
    ("cal_v+1.3_fwd", ["--target-vel", "1.3"]),
    ("cal_v+1.6_fwd", ["--target-vel", "1.6"]),
    ("cal_mov_side", ["--target-vel", "1.0", "--movement", "0", "1", "0"]),
    ("cal_mov_back", ["--target-vel", "1.0", "--movement", "-1", "0", "0"]),
    ("cal_face_side", ["--target-vel", "1.0", "--facing", "0", "1", "0"]),
    ("cal_face_back", ["--target-vel", "1.0", "--facing", "-1", "0", "0"]),
]


def sweep(seconds: float, out: Path) -> int:
    for tag, extra in SWEEP:
        cmd = [sys.executable, "-m", "manifold_g1.clip", "--tag", tag,
               "--seconds", str(seconds), "--out", str(out)] + extra
        print(f"--- {tag}: {' '.join(extra)}", flush=True)
        subprocess.run(cmd, check=True)
    return 0


def mode_sweep(seconds: float, out: Path, modes: list[int], target_vel: float) -> int:
    """One short clip per planner mode: which modes produce usable motion, and how they style it.

    Style is read from the recorded joint means, so this doubles as the multi-style source the
    imitation side of the primitive model needs (Phase 2): mode = a style family.
    """
    import numpy as np
    from . import constants as C

    rows = []
    for mode in modes:
        tag = f"mode{mode:02d}"
        cmd = [sys.executable, "-m", "manifold_g1.clip", "--tag", tag, "--seconds", str(seconds),
               "--mode", str(mode), "--target-vel", str(target_vel), "--out", str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  mode {mode:2d}: FAILED ({result.stderr.strip().splitlines()[-1][:60]})")
            continue
        summary = json.loads((out / f"{tag}.json").read_text())
        data = np.load(out / f"{tag}.npz")
        q = data["q_act"]
        speed = summary["speed_mean"]
        verdict = ("fell" if summary["fell"] else
                   "moved" if speed > 0.15 else "held still")
        # style fingerprint: mean of a few expressive joints (deg)
        def mean_of(joint):
            idx = C.ISAACLAB_TO_MUJOCO[C.MOTOR_NAMES.index(joint)]
            return float(np.degrees(q[:, idx].mean()))
        row = {"mode": mode, "verdict": verdict, "speed": speed, "track": summary["track_err_mean_rad"],
               "base_z": summary["base_z_mean"], "travelled": summary["travelled_m"],
               "envelope": summary["envelope_semi_mean"], "top": summary["envelope_top_mean"],
               "arm": mean_of("left_shoulder_pitch"), "elbow": mean_of("left_elbow"),
               "waist": mean_of("waist_yaw"), "hip": mean_of("left_hip_pitch"),
               "knee": mean_of("left_knee")}
        rows.append(row)
        print(f"  mode {mode:2d}: {verdict:>11} speed {row['speed']:.2f}  track {row['track']:.3f}  "
              f"top {row['top']:.3f}  waist {row['waist']:+6.1f}  arm {row['arm']:+6.1f}  "
              f"knee {row['knee']:+6.1f}", flush=True)

    (out / "mode_table.json").write_text(json.dumps(rows, indent=2))
    usable = [r for r in rows if r["verdict"] == "moved"]
    print(f"\n{len(usable)}/{len(rows)} modes produced usable locomotion")
    if len(usable) >= 2:
        keys = ["waist", "arm", "elbow", "hip", "knee"]
        print("style spread across usable modes (std over modes, deg): " +
              ", ".join(f"{k} {np.std([r[k] for r in usable]):.1f}" for k in keys))
    print(f"saved {out / 'mode_table.json'}")
    return 0


def table(out: Path) -> int:
    rows = []
    for path in sorted(out.glob("cal_*.json")):
        data = json.loads(path.read_text())
        cfg = data["config"]
        rows.append({
            "tag": path.stem,
            "mode": cfg["mode"],
            "target_vel": cfg["target_vel"],
            "movement": cfg["movement"],
            "facing": cfg["facing"],
            # recomputed from the recorded speed: a lateral clip is motion even with vx ≈ 0
            "verdict": ("fell" if data["fell"] else
                        "moved" if data["speed_mean"] > 0.15 else "held still"),
            "vx": data["vx_mean"], "vy": data["vy_mean"], "speed": data["speed_mean"],
            "yaw_start": data["yaw_start_deg"], "yaw_end": data["yaw_end_deg"],
            "yaw_change": data["yaw_change_deg"],
            "track_err": data["track_err_mean_rad"],
            "contact": data["contact_ratio"],
            "envelope": data["envelope_semi_mean"],
            "base_z": data["base_z_mean"],
        })
    if not rows:
        print(f"no calibration clips in {out}; run `calibrate sweep` first")
        return 1

    print(f"{'command':>16} {'target_vel':>10} {'movement':>14} {'facing':>12} {'verdict':>11} "
          f"{'vx':>7} {'vy':>7} {'yaw Δ':>8} {'track':>7}")
    for r in rows:
        print(f"{r['tag']:>16} {r['target_vel']:>10.2f} {str(r['movement']):>14} "
              f"{str(r['facing']):>12} {r['verdict']:>11} {r['vx']:>+7.3f} {r['vy']:>+7.3f} "
              f"{r['yaw_change']:>+8.1f} {r['track_err']:>7.4f}")

    forward = [r for r in rows if r["movement"] == [1.0, 0.0, 0.0]
               and r["verdict"] == "moved" and r["target_vel"] > 0]
    if len(forward) >= 2:
        tv = np.array([r["target_vel"] for r in forward])
        vx = np.array([r["vx"] for r in forward])
        slope, intercept = np.polyfit(tv, vx, 1)
        print(f"\nachieved vx ≈ {slope:.3f} x target_vel {intercept:+.3f}  "
              f"(speed range sampled: {vx.min():.2f} .. {vx.max():.2f} m/s)")

    (out / "command_table.json").write_text(json.dumps(rows, indent=2))
    print(f"saved {out / 'command_table.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Walking command calibration")
    parser.add_argument("mode", choices=("sweep", "table", "modes"))
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--target-vel", type=float, default=1.0)
    parser.add_argument("--modes", type=int, nargs="+", default=MODES)
    parser.add_argument("--out", type=Path, default=CLIP_DIR)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.mode == "sweep":
        return sweep(args.seconds, args.out)
    if args.mode == "modes":
        return mode_sweep(args.seconds, args.out, args.modes, args.target_vel)
    return table(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
