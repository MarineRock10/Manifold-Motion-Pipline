"""Recompute stored envelopes with the current body-surface definition (offline, no simulation).

The clips recorded `envelope_semi` from the eight extremity landmarks - a fast approximation
whose points (fingertips, toe tips) sit outside the body surface, and which misses convex parts
like the shoulders. Since then the body definition has been unified on the sampled *surface*
(mesh vertices plus capsule/sphere surfaces), so every stored envelope is on a different ruler
than the one the models are trained and judged with.

Re-running the whole collection would take ~20 minutes of simulation, but it is not needed: the
clips already store the measured joint angles per tick, and the envelope is a pure function of
the pose. So the envelopes are recomputed here from `q_act`, with the same code the training
path uses, and written back into the npz alongside the old ones for comparison.

    python3 -m manifold_motion.recompute_envelopes --clips reports/manifold_motion/clips
    python3 -m manifold_motion.recompute_envelopes --analyze      # compare old vs new
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .body_envelope import body_points, read_landmarks
from .env import G1FlatEnv


class PoseSurfacer:
    """Body-surface envelope for a batch of poses, reusing one MuJoCo model."""

    def __init__(self, max_vertices: int = 120):
        self.env = G1FlatEnv()
        self.max_vertices = max_vertices

    def envelope(self, q_policy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Ellipsoid semi-axes, body centre, and top height for one joint vector.

        The first two are **pelvis-anchored** (the frame every manifold in this project is built
        in); the top height is in the floor frame, because it is compared against a corridor
        clearance. Writing the centre in world coordinates instead silently breaks every
        containment test downstream - the ellipsoid ends up near the floor while the body is
        described around the pelvis, and the ratio comes out around 10.

        The semi-axes come from `fit_ellipsoid`, not from `max|rel|` per axis. That distinction
        is the whole point of this function:

          * `max|rel|` per axis is a **box**, and a box is not an ellipsoid. Using box
            half-extents as ellipsoid semi-axes means the body's own points sit at
            r = 1.18-1.57 (measured over 60 ticks; median 1.40) instead of <= 1 - so "fit inside
            the manifold", which every reward and every verification test measures, was not
            satisfiable even by the pose the envelope was measured from. 39% of the standing
            surface scored outside its own envelope.
          * `fit_ellipsoid` multiplies the half-extents by `max_i ||rel_i / semi||`, one scalar
            for all three axes. That is exactly "inflate the box until the body fits", and it
            makes r <= 1 true (verified: r = 1.0000 on every sampled tick).

        Be precise about what this is and is not. It is a **uniform** scale of the box, so the
        aspect ratio is unchanged; it is not a per-axis refit. A single global constant cannot
        replace it, because the factor varies per tick (1.18-1.57, median 1.40) - a constant
        would over-inflate the roomy ticks and still clip the tight ones.
        """
        import mujoco

        from .body_envelope import fit_ellipsoid

        env = self.env
        env.data.qpos[env.body_qadr] = q_policy[C.ISAACLAB_TO_MUJOCO]
        mujoco.mj_kinematics(env.model, env.data)
        points = body_points(env.model, env.data, max_vertices=self.max_vertices)
        pelvis = env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
        rel = points - pelvis
        fit = fit_ellipsoid(rel, center=np.zeros(3))
        return np.asarray(fit["semi"], dtype=np.float64), np.asarray(fit["center"],
                                                                    dtype=np.float64), \
            float(points[:, 2].max())


def recompute(clips: Path, stride: int, out: Path | None, overwrite: bool) -> int:
    surfacer = PoseSurfacer()
    files = sorted(clips.glob("*.npz"))
    report = []
    for path in files:
        with np.load(path) as handle:
            data = {k: handle[k] for k in handle.files}
        if "q_act" not in data:
            continue
        q = data["q_act"]
        ticks = len(q)
        old_semi = data.get("envelope_semi")
        old_stride = max(1, ticks // len(old_semi)) if old_semi is not None else stride
        idx = np.minimum(np.arange(0, ticks, stride), ticks - 1)

        semi = np.zeros((len(idx), 3))
        centers = np.zeros((len(idx), 3))
        tops = np.zeros(len(idx))
        for k, tick in enumerate(idx):
            semi[k], centers[k], tops[k] = surfacer.envelope(q[tick])

        if overwrite:
            data["envelope_semi"] = semi
            data["envelope_top"] = tops
            data["envelope_t"] = data["t"][idx]
            data["envelope_center"] = centers            # pelvis frame, like the extents
            np.savez_compressed(path, **data)

        if old_semi is not None:
            # compare on the same ticks: the old record was sampled at its own stride
            old_idx = np.minimum(idx // old_stride, len(old_semi) - 1)
            report.append({
                "clip": path.stem,
                "old_semi": old_semi[old_idx].mean(axis=0).tolist(),
                "new_semi": semi.mean(axis=0).tolist(),
                "delta": (semi.mean(axis=0) - old_semi[old_idx].mean(axis=0)).tolist(),
            })
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    if report:
        old = np.array([r["old_semi"] for r in report])
        new = np.array([r["new_semi"] for r in report])
        print(f"{len(report)} clips recomputed (stride {stride})"
              + ("  [written back]" if overwrite else "  [dry run]"))
        print(f"  old landmark envelope: x {old[:,0].mean():.3f}  y {old[:,1].mean():.3f}  "
              f"z {old[:,2].mean():.3f}")
        print(f"  new surface envelope : x {new[:,0].mean():.3f}  y {new[:,1].mean():.3f}  "
              f"z {new[:,2].mean():.3f}")
        print(f"  mean shrink: {np.round((new - old).mean(axis=0), 4)}  "
              f"(negative = the surface is tighter than the landmark points)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute stored envelopes on the surface ruler")
    parser.add_argument("--clips", type=Path, default=Path("reports/manifold_motion/clips"))
    parser.add_argument("--stride", type=int, default=5, help="control ticks between envelopes")
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_motion/clips/envelope_diff.json"))
    parser.add_argument("--overwrite", action="store_true",
                        help="write the recomputed envelopes back into each npz")
    args = parser.parse_args()
    return recompute(args.clips, args.stride, args.out, args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
