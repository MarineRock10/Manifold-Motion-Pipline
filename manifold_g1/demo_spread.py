"""How much the recorded poses vary inside one manifold.

The three stages are judged on three different things, but two of them share one question: a
policy that maps manifolds to poses is only useful if the manifold *determines* the pose. If the
recorded poses under one envelope are scattered, there is nothing to clone and nothing to track -
the stage's number would then be measuring the data, not the policy.

That is measured here, once, so `show_bc` and `show_rl` report it the same way and a low score
can be attributed: either the policy, or a manifold the data leaves ambiguous.

    python3 -m manifold_g1.demo_spread            # summary over every recorded manifold
"""

from __future__ import annotations

import argparse

import numpy as np

from . import constants as C
from .demos import demo_values, load_demos

# Near-spread criterion: a recorded pose counts as "the same posture" when its mean absolute
# joint difference from the manifold's least-extreme pose is under this. 0.10 rad is about the
# execution gap between requesting a pose and SONIC reaching it (measured 0.06-0.14), so poses
# closer together than this are not distinguishable through the controller anyway.
NEAR_RAD = 0.10


def spread(poses: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Return (per-pose mean |q - median|, that statistic's median, near-copy count).

    The median is the robust centre here: a manifold can legitimately hold two very different
    postures (arms tucked vs. arms out), and the median stays between them instead of being
    dragged by whichever cluster happens to be larger.
    """
    poses = np.atleast_2d(np.asarray(poses, dtype=np.float64))
    deviations = np.abs(poses - np.median(poses, axis=0)).mean(axis=1)
    near = int((deviations <= NEAR_RAD).sum())
    return deviations, float(np.median(deviations)), near


def table(min_poses: int = 4) -> dict[str, dict]:
    """Per-manifold spread over the whole recorded set, keyed by the demo's manifold key."""
    _, all_demos = load_demos()
    out = {}
    for key, poses in all_demos.items():
        if len(poses) < min_poses:
            continue
        deviations, median_dev, near = spread(poses)
        values = demo_values(key)
        out[key] = {
            "semi": values[:3], "center": values[3:], "count": len(poses),
            "spread": median_dev, "near": near, "near_frac": near / len(poses),
            "worst": float(deviations.max()),
        }
    return out


def nearest_key(all_demos: dict, semi: np.ndarray) -> str:
    """The recorded manifold whose envelope is closest to `semi` (in semi-axes only).

    The recorded envelopes and this query are both the same measured object, so the nearest key
    is the same manifold up to the key's 2-decimal rounding - which is exactly the tolerance
    that makes a lookup meaningful rather than accidental. Used by `eval_session` to report the
    recorded-pose spread of the manifold a row is judged on.
    """

    keys = np.stack([demo_values(k)[:3] for k in all_demos])
    return list(all_demos)[int(np.argmin(np.abs(keys - np.asarray(semi)[None, :]).mean(axis=1)))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Recorded-pose spread inside each manifold")
    parser.add_argument("--min-poses", type=int, default=4)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    stats = table(args.min_poses)
    if not stats:
        print("no manifold has enough recorded poses; build the demo set first")
        return 1
    medians = np.array([s["spread"] for s in stats.values()])
    near = np.array([s["near_frac"] for s in stats.values()])
    counts = np.array([s["count"] for s in stats.values()])
    print(f"{len(stats)} manifolds with >= {args.min_poses} recorded poses, "
          f"{int(counts.sum())} poses total")
    print(f"  spread (median |q - median(q)|):  median {np.median(medians):.3f} rad  "
          f"p10 {np.percentile(medians, 10):.3f}  p90 {np.percentile(medians, 90):.3f}")
    print(f"  near-copies of the manifold's centre pose (<= {NEAR_RAD:.2f} rad):  "
          f"median fraction {np.median(near):.0%}")
    print(f"  poses per manifold: median {int(np.median(counts))}, max {int(counts.max())}")

    # the two ends are what a viewer will show: the most consistent manifold and the least
    ordered = sorted(stats.items(), key=lambda kv: kv[1]["spread"])
    print(f"\n  {'key':>34} {'n':>4} {'spread':>7} {'near':>5}")
    for key, s in ordered[:args.examples]:
        print(f"  {key:>34} {s['count']:>4} {s['spread']:>7.3f} {s['near_frac']:>5.0%}")
    if len(ordered) > 2 * args.examples:
        print("  ...")
    for key, s in ordered[-args.examples:]:
        print(f"  {key:>34} {s['count']:>4} {s['spread']:>7.3f} {s['near_frac']:>5.0%}")
    print("\nThe top rows are manifolds that pin the pose down; the bottom rows are ones the")
    print("data leaves ambiguous, where a policy cannot be blamed for producing one of several.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
