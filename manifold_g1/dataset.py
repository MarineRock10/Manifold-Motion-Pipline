"""Sliding-window dataset from recorded clips (Phase 1.4/1.5).

The clips store whole episodes; a training sample is a window inside one of them:

    X_t = (M_t, z_p,t, s_t, H_t, c_t)   ->   Y_t = R_{t:t+H}

`M_t` is extracted *reversely* from the motion: the body envelope the robot actually occupied
at that tick (pelvis-anchored extremity landmarks, with a small margin) - what the robot needed
in order to be there. `z_p,t` is the primitive the robot was executing, read off from the
command log (Phase 1 has one primitive: walk with a velocity command). `R` is the future of the
same episode, so the sample never crosses an episode boundary.

Windows are indexed, not materialized: the per-episode arrays are already on disk, and a window
is (episode, start index). That keeps storage at the episode level and lets the loader slice.

    python3 -m manifold_g1.dataset build --clips reports/manifold_g1/clips \
        --out reports/manifold_g1/dataset
    python3 -m manifold_g1.dataset show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C

HORIZON = 50          # ticks in the prediction window (1 s at 50 Hz)
HISTORY = 25          # ticks of history the sample carries (0.5 s)
STRIDE = 5            # 10 Hz window starts
MARGIN = 1.05         # body envelope -> manifold, a little room so the policy is not on the edge


def _arrays(path: Path) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def episode_index(path: Path, summary: dict, stride: int = STRIDE,
                  horizon: int = HORIZON, history: int = HISTORY,
                  max_track: float | None = None) -> dict:
    """Valid window starts for one episode: enough history, enough future, not fallen."""
    ticks = int(summary["ticks"])
    starts = np.arange(history, max(history, ticks - horizon), stride, dtype=np.int64)
    if max_track is not None:
        # drop windows whose command tracking was poor somewhere inside
        err = summary.get("track_err_mean_rad")
        if err is not None and err > max_track:
            starts = starts[:0]
    return {"file": path.name, "ticks": ticks, "starts": starts.tolist(),
            "fell": bool(summary["fell"]), "track_err": float(summary["track_err_mean_rad"])}


def build(clips: Path, out: Path, stride: int = STRIDE, horizon: int = HORIZON,
          history: int = HISTORY, exclude_prefix: str = "cal_") -> int:
    summaries = sorted(clips.glob("*.json"))
    episodes, skipped = [], 0
    for js in summaries:
        if js.stem.startswith(exclude_prefix) or js.stem.startswith("batch_"):
            skipped += 1
            continue
        npz = clips / f"{js.stem}.npz"
        if not npz.exists():
            skipped += 1
            continue
        with np.load(npz) as handle:            # a sample needs the command channel
            if "cmd" not in handle.files:
                skipped += 1
                continue
        summary = json.loads(js.read_text())
        if summary["fell"]:
            skipped += 1
            continue
        episodes.append(episode_index(npz, summary, stride, horizon, history))

    index = {"clips_dir": str(clips), "stride": stride, "horizon": horizon,
             "history": history, "margin": MARGIN,
             "episodes": episodes,
             "windows": int(sum(len(e["starts"]) for e in episodes))}
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"dataset: {len(episodes)} episodes, {index['windows']} windows "
          f"(skipped {skipped} files), H={horizon} ticks, stride={stride}, history={history}")
    return 0


def load_episode(index: dict, episode: dict) -> dict:
    return _arrays(Path(index["clips_dir"]) / episode["file"])


def sample(index: dict, episode: dict, data: dict, t: int) -> dict:
    """One training sample: (M, z_p, s, H, c) and the future R."""
    h, stride, margin = index["horizon"], index["stride"], index["margin"]
    window = slice(t, t + h)
    pelvis = data["base_pos"][t]
    semi = data["envelope_semi"][t // stride if t // stride < len(data["envelope_semi"]) else -1]
    return {
        "M": {"center": data["base_pos"][t].copy(), "semi": semi * margin,
              "quat": data["base_quat"][t].copy()},
        "z_p": {"type": "walk", "vel": data["cmd"][t][:1].copy()},
        "s": {"q": data["q_act"][t], "dq": data["dq_act"][t], "base_pos": data["base_pos"][t],
              "base_quat": data["base_quat"][t], "base_lin_vel": data["base_lin_vel"][t],
              "base_ang_vel": data["base_ang_vel"][t], "contact": data["contact"][t]},
        "H": {"q": data["q_act"][t - index["history"]:t],
              "base_pos": data["base_pos"][t - index["history"]:t]},
        "c": data["cmd"][t],
        "R": {"t": data["t"][window], "q": data["q_act"][window], "dq": data["dq_act"][window],
              "base_pos": data["base_pos"][window], "base_quat": data["base_quat"][window],
              "contact": data["contact"][window]},
    }


def show(out: Path) -> int:
    index = json.loads((out / "index.json").read_text())
    print(f"windows {index['windows']} from {len(index['episodes'])} episodes "
          f"(H={index['horizon']}, stride={index['stride']}, history={index['history']}, "
          f"margin {index['margin']})")

    speeds, semi, top, angles, ticks = [], [], [], [], 0
    for episode in index["episodes"]:
        data = load_episode(index, episode)
        starts = np.array(episode["starts"])
        ticks += episode["ticks"]
        if "cmd" not in data:
            continue
        speeds.append(np.linalg.norm(data["base_lin_vel"][:, :2], axis=1))
        semi.append(data["envelope_semi"])
        top.append(data["envelope_top"])
        angles.append(np.arctan2(data["cmd"][:, 4], data["cmd"][:, 3]))   # facing direction
    speed = np.concatenate(speeds)[50:]
    semi = np.concatenate(semi)
    top = np.concatenate(top)
    angles = np.concatenate(angles)

    def q(a, p):
        return np.percentile(a, p)

    print(f"sim time {ticks / 50.0:.1f} s   ({ticks} ticks)")
    print(f"speed   m/s : p5 {q(speed,5):.2f}  p50 {q(speed,50):.2f}  p95 {q(speed,95):.2f}  "
          f"max {speed.max():.2f}")
    for i, name in enumerate("xyz"):
        print(f"envelope {name}   : p5 {q(semi[:,i],5):.3f}  p50 {q(semi[:,i],50):.3f}  "
              f"p95 {q(semi[:,i],95):.3f}  spread {semi[:,i].max()-semi[:,i].min():.3f} m")
    print(f"envelope top : p5 {q(top,5):.3f}  p50 {q(top,50):.3f}  p95 {q(top,95):.3f} m")

    hist, _ = np.histogram(np.degrees(angles) % 360, bins=8, range=(0, 360))
    coverage = float((hist > 0).mean())
    print(f"facing coverage over 8 sectors: {coverage*100:.0f}%   "
          f"counts {hist.tolist()}")
    speed_bins = np.histogram(speed, bins=4, range=(0.0, 1.0))[0]
    print(f"speed coverage over 4 bins (0-1 m/s): "
          f"{(speed_bins > 0).mean()*100:.0f}%   counts {speed_bins.tolist()}")

    health = {
        "windows": index["windows"], "episodes": len(index["episodes"]),
        "sim_seconds": ticks / 50.0,
        "speed": {"p5": float(q(speed,5)), "p50": float(q(speed,50)), "p95": float(q(speed,95))},
        "envelope": {n: {"p5": float(q(semi[:,i],5)), "p50": float(q(semi[:,i],50)),
                         "p95": float(q(semi[:,i],95)),
                         "spread": float(semi[:,i].max()-semi[:,i].min())}
                     for i, n in enumerate("xyz")},
        "facing_coverage": coverage, "speed_coverage": float((speed_bins > 0).mean()),
    }
    (out / "health.json").write_text(json.dumps(health, indent=2))
    print(f"saved {out / 'health.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sliding-window dataset (Phase 1.4/1.5)")
    parser.add_argument("mode", choices=("build", "show"))
    parser.add_argument("--clips", type=Path, default=Path("reports/manifold_g1/clips"))
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_g1/dataset"))
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--history", type=int, default=HISTORY)
    args = parser.parse_args()
    if args.mode == "build":
        return build(args.clips, args.out, args.stride, args.horizon, args.history)
    return show(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
