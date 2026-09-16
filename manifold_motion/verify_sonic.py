"""Does SONIC actually hold the poses the primitive model proposes?

Every other check in this project judges a pose against a geometric body model: containment in
the ellipsoid, centre of mass over the feet, joint-limit margins. None of that says the frozen
controller will adopt the pose - it tracks the joints it was trained to track and ignores
others, so arm-heavy or waist-pitched poses come back substantially unchanged.

This runs the loop for real: for each manifold, ask the primitive model for a pose, hand that
29-joint vector to SONIC as a keyframe, let it settle, and compare

  * the requested pose against the joints the robot actually reached (per-joint error, and the
    error grouped by leg / waist / arm),
  * the containment the model computed for the request against the containment of what happened,
  * which of the two is worse and by how much - the executability gap.

    python3 -m manifold_motion.verify_sonic --policy reports/manifold_motion/bc/policy.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .family import report_specs
from .manifold import quat_to_matrix
from .paths import RL_POLICY, VERIFY_REPORT

OUT = VERIFY_REPORT


def policy_pose(ppo, kin, obs_dim: int, manifold, max_steps: int = 8):
    """Roll the policy on one manifold and return the pose it settles on (device-free numpy)."""
    import torch

    from .torch_env import Config, TorchPrimitiveEnv
    from .demos import load_demos

    mean, allv = load_demos()
    env = TorchPrimitiveEnv(kin, np.concatenate(list(allv.values())), Config(), n=1, pool=1, seed=0)
    primitive = manifold.primitives[0]
    env.pool["semi"][0] = torch.as_tensor(primitive.semi, dtype=kin.dtype, device=kin.device)
    env.pool["center"][0] = torch.as_tensor(primitive.center, dtype=kin.dtype, device=kin.device)
    env.pool["rot"][0] = torch.as_tensor(quat_to_matrix(primitive.quat), dtype=kin.dtype,
                                         device=kin.device)
    env.reset()
    pose = np.zeros(29)
    with torch.no_grad():
        for _ in range(max_steps):
            action, _, _ = ppo.act(env.obs()[0].cpu().numpy(), deterministic=True)
            obs, _, done, info = env.step(
                torch.as_tensor(action[None, :], dtype=kin.dtype, device=kin.device))
            pose = info["pose"][0].cpu().numpy()
            if bool(done[0]):
                break
    return pose


def main() -> int:
    from .keyframe_env import KeyframeEnv
    from .ppo import PPO
    from .body_model import BodyModel

    parser = argparse.ArgumentParser(description="Validate primitive poses on frozen SONIC")
    parser.add_argument("--policy", type=Path,
                        default=RL_POLICY)
    parser.add_argument("--settle", type=float, default=1.5, help="seconds to hold before reading")
    parser.add_argument("--steps", type=int, default=8, help="policy steps per manifold")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from .kinematics import TorchKinematics

    kin = TorchKinematics(device=args.device)
    body = BodyModel()
    ppo = None
    env = KeyframeEnv()
    print(f"{'manifold':>26} {'r_model':>8} {'r_sonic':>8} {'joint err':>10} {'legs':>7} "
          f"{'waist':>7} {'arms':>7} {'outcome':>9}")
    rows = []
    for spec in report_specs():
        manifold = spec.build()
        if ppo is None:
            from .torch_env import Config, TorchPrimitiveEnv
            from .demos import load_demos
            mean, allv = load_demos()
            probe = TorchPrimitiveEnv(kin, np.concatenate(list(allv.values())), Config(),
                                      n=1, pool=1, seed=0)
            obs_dim = probe.obs().shape[1]
            ppo = PPO(obs_dim, 29)
            ppo.load(args.policy)
        pose = policy_pose(ppo, kin, obs_dim, manifold, args.steps)

        # what the model thinks this pose is worth
        r_model = float(manifold.radii(body.mesh_points_pose(pose)).max())

        # what SONIC actually reaches when asked to hold it
        env.reset()
        env.set_joints(pose)
        for _ in range(int(args.settle / C.CONTROL_DT)):
            env.step()
        reached = env.state()["q_hw"][C.MUJOCO_TO_ISAACLAB] - C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB]
        r_sonic = float(manifold.radii(body.mesh_points_pose(reached)).max())

        err = np.abs(reached - pose)
        legs, waist, arms = err[:12].mean(), err[12:15].mean(), err[15:].mean()
        outcome = "holds" if r_sonic <= 1.0 else "outside"
        print(f"{spec.label():>26} {r_model:>8.2f} {r_sonic:>8.2f} {err.mean():>10.3f} "
              f"{legs:>7.3f} {waist:>7.3f} {arms:>7.3f} {outcome:>9}")
        rows.append({"manifold": spec.label(), "pose": pose.tolist(), "reached": reached.tolist(),
                     "r_model": r_model, "r_sonic": r_sonic, "joint_err": float(err.mean()),
                     "legs": float(legs), "waist": float(waist), "arms": float(arms),
                     "outcome": outcome})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    holds = sum(1 for r in rows if r["outcome"] == "holds")
    r_s = np.array([r["r_sonic"] for r in rows])
    r_m = np.array([r["r_model"] for r in rows])
    print(f"\nSONIC held {holds}/{len(rows)} of the proposed poses inside their manifold")
    # The binary count is dominated by poses that sit *on* the boundary: containment moves by
    # ~0.05 with a 0.03 rad tracking error, so a pose at r = 1.01 and one at r = 0.99 are the
    # same result. Report the distribution as well, or a near-miss reads as a failure.
    print(f"  r(sonic)   mean {r_s.mean():.3f}  median {np.median(r_s):.3f}  "
          f"p10 {np.percentile(r_s, 10):.3f}  p90 {np.percentile(r_s, 90):.3f}  "
          f"max {r_s.max():.3f}")
    print(f"  r(model)   mean {r_m.mean():.3f}  median {np.median(r_m):.3f}")
    print(f"  within 1.05 of the boundary: {int((r_s <= 1.05).sum())}/{len(rows)}   "
          f"inside 1.10: {int((r_s <= 1.10).sum())}/{len(rows)}")
    print(f"  mean joint tracking error {np.mean([r['joint_err'] for r in rows]):.3f} rad")
    print(f"saved {OUT.relative_to(C.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
