"""Random-walk data collection (Phase 1.2/1.3).

Samples a random walk schedule (speed, travel direction, body heading, per 2-5 s segment),
drives the planner, has frozen SONIC execute it in MuJoCo, and records every tick. Episodes
run in parallel processes: each owns its own MuJoCo model and ONNX sessions.

    python3 -m manifold_g1.collect --episodes 8 --jobs 4 --seconds 20
    python3 -m manifold_g1.collect --episodes 100 --jobs 6 --seconds 30 --out reports/manifold_g1/clips

The recorded `q_act` is the ground truth (the frozen controller is path dependent and does not
adopt every keyframe), and the per-tick command is kept so a sample can be labelled with what
was asked for as well as what happened.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from .clip import ClipConfig, ClipRecorder, Segment

# calibrated command range: `target_vel` is not m/s (achieved ≈ 0.49 x command)
SPEED_RANGE = (0.8, 1.6)          # -> about 0.4 .. 0.8 m/s
SEGMENT_SECONDS = (2.0, 5.0)
TURN_PROBABILITY = 0.5            # probability a segment also changes the body heading


def random_schedule(rng: np.random.Generator, seconds: float,
                    speed_range=SPEED_RANGE) -> list[Segment]:
    """A random walk: 8-direction travel, heading that sometimes leads or lags the direction."""
    schedule, elapsed = [], 0.0
    while elapsed < seconds:
        duration = float(rng.uniform(*SEGMENT_SECONDS))
        angle = float(rng.uniform(0, 2 * np.pi))
        movement = (float(np.cos(angle)), float(np.sin(angle)), 0.0)
        if rng.random() < TURN_PROBABILITY:                    # walk one way, face another
            heading = angle + float(rng.uniform(-np.pi / 2, np.pi / 2))
        else:                                                  # walk where you look
            heading = angle
        facing = (float(np.cos(heading)), float(np.sin(heading)), 0.0)
        schedule.append(Segment(duration=duration, movement=movement, facing=facing,
                                target_vel=float(rng.uniform(*speed_range))))
        elapsed += duration
    return schedule


def run_episode(index: int, seed: int, seconds: float, out_dir: str, warmup: float,
                replan_every: int = 5) -> dict:
    """One worker: build a recorder, run a random walk, save the clip."""
    rng = np.random.default_rng(seed)
    schedule = random_schedule(rng, seconds)
    cfg = ClipConfig(seconds=seconds, schedule=schedule, warmup_seconds=warmup,
                     replan_every=replan_every)
    recorder = ClipRecorder()
    data = recorder.run(cfg)
    recorder.last_data = data
    tag = f"ep{index:04d}_seed{seed}"
    recorder.save(Path(out_dir), tag)
    return recorder.summary | {"clip": tag, "seed": int(seed)}


def _run_job(job: tuple) -> dict:
    """Unpack one (index, seed, seconds, out_dir, warmup) tuple for `imap_unordered`."""
    return run_episode(*job)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect random-walk clips in parallel")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=4,
                        help="parallel processes; each ONNX session is capped at "
                             "constants.ONNX_THREADS")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=0.4)
    parser.add_argument("--replan-every", type=int, default=5,
                        help="control ticks between planner calls (5 = 10 Hz, 10 = 5 Hz)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_g1/clips"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(i, args.seed + i, args.seconds, str(args.out), args.warmup, args.replan_every)
            for i in range(args.episodes)]
    wall = time.perf_counter()
    results = []
    context = mp.get_context("spawn")
    with context.Pool(min(args.jobs, args.episodes)) as pool:
        for result in pool.imap_unordered(_run_job, jobs):
            results.append(result)
            print(f"[{len(results):3d}/{len(jobs)}] {result['clip']}: "
                  f"{result['travelled_m']:6.2f} m in {result['ticks']} ticks, "
                  f"track {result['track_err_mean_rad']:.4f} rad, "
                  f"{'FELL' if result['fell'] else 'ok'}", flush=True)
    wall = time.perf_counter() - wall

    ticks = sum(r["ticks"] for r in results)
    good = [r for r in results if not r["fell"] and r["track_err_mean_rad"] < 0.06]
    batch = {
        "episodes": len(results), "jobs": args.jobs, "seconds_each": args.seconds,
        "sim_seconds": sum(r["ticks"] for r in results) / 50.0,
        "wall_seconds": wall, "realtime_factor": sum(r["ticks"] for r in results) / 50.0 / wall,
        "good_rate": len(good) / len(results),
        "travelled_mean": float(np.mean([r["travelled_m"] for r in results])),
        "track_err_mean": float(np.mean([r["track_err_mean_rad"] for r in results])),
        "envelope_mean": [float(x) for x in np.mean([r["envelope_semi_mean"] for r in results], axis=0)],
        "clips": [r["clip"] for r in results],
    }
    (args.out / f"batch_seed{args.seed}.json").write_text(json.dumps(batch, indent=2))
    print(f"\n{len(results)} episodes, {batch['sim_seconds']:.1f} s of sim in {wall:.1f} s wall "
          f"({batch['realtime_factor']:.2f}x realtime on {args.jobs} jobs)")
    print(f"good rate {batch['good_rate']:.2f} (no fall, tracking < 0.06 rad), "
          f"travelled {batch['travelled_mean']:.2f} m/episode, "
          f"envelope {np.round(batch['envelope_mean'], 3).tolist()}")
    print(f"saved {args.out}/batch_seed{args.seed}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
