"""Compare policies at the stage each one belongs to, with the metric that stage defines.

The earlier comparison table was wrong because it judged every stage with `verify_sonic`, which
samples `family.report_specs()` - a hand-picked list that is in *no* stage's training
distribution. That measures generalisation to unseen manifolds, which is a legitimate metric but
not the one the stages optimise. Judging BC by it and concluding "BC is bad" mistakes a
distribution shift for a failure.

Each stage has its own metric:

  BC           pose error against demonstrations, on the manifolds those demonstrations carry
  SONIC-in-loop  success and achieved containment on the perturbed manifolds it trains on, and
               (independently) the measured landing pose from the real controller

This script reports all of them side by side, and labels which is which, so a number cannot be
read as belonging to the wrong stage.

    python3 -m manifold_g1.compare --bc .../bc_policy.pt --rl .../policy_sonicrl.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .family import report_specs
from .keyframe_env import KeyframeEnv
from .pose_policy import OBS_DIM, POSE_DIM, action_to_pose, observation
from .ppo import PPO
from .primitive import load_demos
from .static_fit import BodyModel

OUT = C.REPO / "reports" / "manifold_g1" / "compare.json"


def _rollout(ppo, body, manifold, steps: int = 8) -> np.ndarray:
    """Ask the policy for a pose, then hold it: SONIC drives, physics settles."""
    pose = np.zeros(POSE_DIM)
    for _ in range(steps):
        r = float(manifold.radii(body.mesh_points_pose(pose)).max())
        obs = observation(pose, manifold.primitives[0], r,
                          body.at_pose(pose)[body.pelvis_index()])
        import torch
        with torch.no_grad():
            action, _, _ = ppo.act(obs, deterministic=True)
        target = np.asarray(action_to_pose(action, torch))
        pose = pose + (target - pose) * 0.5
    return pose


def measure(ppo, body, sonic, manifold, repeats: int = 1) -> dict:
    """Requested and achieved containment for one manifold, through the real controller."""
    best = None
    for _ in range(repeats):
        pose = _rollout(ppo, body, manifold)
        sonic.reset()
        sonic.set_joints(pose)
        for _ in range(int(0.8 / C.CONTROL_DT)):
            sonic.step()
        landed = (sonic.state()["q_hw"][C.MUJOCO_TO_ISAACLAB]
                  - C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB])
        r_req = float(manifold.radii(body.mesh_points_pose(pose)).max())
        r_act = float(manifold.radii(body.mesh_points_pose(landed)).max())
        track = float(np.abs(landed - pose).mean())
        if best is None or r_act < best["r_achieved"]:
            best = {"r_requested": r_req, "r_achieved": r_act, "track": track}
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-stage policy comparison")
    parser.add_argument("--bc", type=Path,
                        default=C.REPO / "reports/manifold_g1/primitive_torch/bc_policy.pt")
    parser.add_argument("--rl", type=Path,
                        default=C.REPO / "reports/manifold_g1/sonic_rl/policy_sonicrl.pt")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    body = BodyModel()
    sonic = KeyframeEnv()
    policies = {}
    for name, path in (("BC", args.bc), ("BC+在环RL", args.rl)):
        if not Path(path).exists():
            print(f"skip {name}: {path} not found")
            continue
        ppo = PPO(OBS_DIM, POSE_DIM, device=args.device)
        ppo.load(path)
        policies[name] = ppo
    if not policies:
        return 1

    # ---- ① the metric BC is trained on: agreement with demonstrations, in-distribution ----
    from .manifold import EllipsoidManifold, Primitive
    _, all_demos = load_demos()
    keys = [k for k in all_demos if len(all_demos[k])]
    rng = np.random.default_rng(0)
    picked = rng.choice(len(keys), min(400, len(keys)), replace=False)
    poses = np.concatenate([all_demos[keys[i]] for i in picked])
    # owner indexes into the *picked* rows (0..len(picked)-1), not into `keys`
    owner = np.concatenate([np.full(len(all_demos[keys[i]]), j) for j, i in enumerate(picked)])
    semi = np.stack([np.array([float(v) for v in keys[i].split(",")][:3]) for i in picked])
    center = np.stack([np.array([float(v) for v in keys[i].split(",")][3:]) for i in picked])
    print("① 与示范的姿态误差（BC 的口径：示范自己的流形，同分布）")
    for name, ppo in policies.items():
        import torch
        rows = []
        for k in range(0, len(poses), 256):
            chunk = slice(k, k + 256)
            ps = poses[chunk]
            obs = []
            for pose, s, c in zip(ps, semi[owner[chunk]], center[owner[chunk]]):
                m = EllipsoidManifold([Primitive(center=c, semi=s)])
                rr = float(m.radii(body.mesh_points_pose(pose)).max())
                obs.append(observation(pose, m.primitives[0], rr,
                                       body.at_pose(pose)[body.pelvis_index()]))
            with torch.no_grad():
                pred = np.asarray(action_to_pose(ppo.model(torch.as_tensor(np.stack(obs)))[0], torch))
            rows.append(np.abs(pred - ps).mean(axis=1))
        err = np.concatenate(rows)
        print(f"   {name:>10}: 姿态误差 均值 {err.mean():.4f}  p90 {np.percentile(err,90):.4f} rad")

    # ---- ② the metric the RL stage is trained on: training-distribution manifolds ----
    print()
    print(f"② 训练分布内的流形（扰动包络，各自 {args.repeats} 次取最好，真 SONIC 执行）")
    print(f"   {'流形':>20} " + " ".join(f"{n:>26}" for n in policies))
    rows = []
    for i in range(args.samples):
        key = keys[int(rng.integers(len(keys)))]
        v = np.array([float(x) for x in key.split(",")])
        semi_i = v[:3] * np.clip(1.0 + rng.normal(0, 0.08, 3), 0.85, 1.15)
        center_i = v[3:] + np.array([rng.normal(0, 0.03), rng.normal(0, 0.03), 0.0])
        from .manifold import EllipsoidManifold as EM, Primitive as P
        manifold_i = EM([P(center=center_i, semi=semi_i)])
        cells, record = [], {"semi": semi_i.tolist()}
        for name, ppo in policies.items():
            r = measure(ppo, body, sonic, manifold_i, repeats=args.repeats)
            record[name] = r
            cells.append(f"r {r['r_achieved']:.2f} trk {r['track']:.3f}")
        print(f"   {str(np.round(semi_i,2)):>20} " + " ".join(f"{c:>26}" for c in cells))
        rows.append(record)

    print()
    for name in policies:
        r = np.array([row[name]["r_achieved"] for row in rows])
        t = np.array([row[name]["track"] for row in rows])
        print(f"   {name:>10}: r(achieved) 中位 {np.median(r):.3f}   r≤1.0 {int((r<=1).sum())}/{len(r)}"
              f"   r≤1.05 {int((r<=1.05).sum())}/{len(r)}   跟踪 {t.mean():.3f} rad")

    # ---- ③ generalisation: the hand-picked manifolds, a *different* metric ----
    print()
    print("③ 泛化：report_specs 手挑流形（这不在任何阶段的训练分布里，是另一项指标）")
    print(f"   {'流形':>22} " + " ".join(f"{n:>26}" for n in policies))
    for spec in report_specs()[:6]:
        manifold_s = spec.build()
        cells = []
        for name, ppo in policies.items():
            r = measure(ppo, body, sonic, manifold_s, repeats=args.repeats)
            cells.append(f"r {r['r_achieved']:.2f} trk {r['track']:.3f}")
        print(f"   {spec.label():>22} " + " ".join(f"{c:>26}" for c in cells))

    print(f"\n(① 同分布精度   ② 训练分布内的真机表现   ③ 跨分布泛化 - 三者不可互换)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
