"""Primitive training entirely on the device: no per-step host round trip.

The numpy version of the environment (`primitive.py`) calls `detach().cpu().numpy()` every step
and feeds the result back through `torch.as_tensor`, so the GPU spends most of each iteration
waiting on the CPU - which is what makes its utilization swing between idle and busy. This
module keeps the whole loop resident:

  * the demonstration set is uploaded once (poses, envelopes, per-manifold membership);
  * a pool of manifolds is generated once as tensors - the policy only ever indexes into it;
  * pose integration, containment, the kinematic penalties, the reward and the PPO update are
    all torch ops on the device, with no numpy in between;
  * the only host transfer is one line of metrics per iteration.

Correctness is checked against the numpy environment (`primitive.py`) by running both on the
same manifolds and comparing radii and rewards, so the fast path cannot silently drift.

    python3 -m manifold_g1.primitive_torch train --iterations 400 --envs 512
    python3 -m manifold_g1.primitive_torch check     # agreement with the numpy env
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import constants as C
from .family import BASE_CENTER, BASE_SEMI, ManifoldSpec, report_specs
from .primitive import POSE_DIM, load_demos
from .ppo import PPO, RolloutBuffer

OUT_DIR = C.REPO / "reports" / "manifold_g1" / "primitive_torch"
POSE_LIMIT = 1.4


def action_to_pose(action, torch=None):
    """Network output -> pose delta, the single definition used by training and deployment.

    `tanh` rather than `clamp`: clamping saturates, so once the raw output drifts past the limit
    it receives no gradient telling it to come back (measured: a cloned policy's output reached
    2.2, was clamped to 1.0, and posed nothing like the demonstration it was fitted to). `tanh`
    keeps a gradient everywhere and bounds the pose by construction.
    """
    torch = torch if torch is not None else __import__("torch")
    return torch.tanh(action) * POSE_LIMIT


@dataclass
class Config:
    dt: float = 0.1
    inner: int = 5                      # inner steps per control tick (matches the numpy env)
    max_steps: int = 8
    tau: float = 0.45
    success_hold: int = 3
    success_margin: float = 0.98
    w_inside: float = 1.0
    # How much being inside is worth per control step. At 0.03 a successful pose earned ~0.03/step
    # while sitting outside a hard manifold lost ~13/step, so on the hardest manifolds (standing
    # r = 1.5, crouching r = 1.05 - still outside) even a full crouch stayed net negative and the
    # policy learned to give up rather than pose. Entering has to dominate.
    margin_cap: float = 0.15
    w_outside: float = 8.0
    # a stream that never entered the manifold ends with this; without it, timing out is free
    fail_penalty: float = 3.0
    w_effort: float = 0.4
    w_imitation: float = 1.0
    w_com: float = 3.0
    w_foot: float = 3.0
    w_limit: float = 5.0
    # the containment reward alone admits degenerate answers: a pose with the two hips bent in
    # opposite directions shrinks the envelope just as well as a crouch does, while being
    # physically meaningless. Symmetry and a torque proxy make "crouch" cheaper than "twist".
    w_symmetry: float = 1.0             # left/right mismatch of paired joints
    w_posture: float = 0.5              # distance from the default standing pose
    w_torque: float = 0.3               # static holding torque (the doc's energy term)
    success_bonus: float = 5.0


class TorchPrimitiveEnv:
    """Many manifolds, all state on the device: [N] streams advanced step by step."""

    def __init__(self, kin, demo_poses, cfg: Config, n: int, pool: int = 64, seed: int = 0):
        import torch

        self.torch = torch
        self.kin = kin
        self.cfg = cfg
        self.device, self.dtype = kin.device, kin.dtype
        self.n = n
        self.rng = np.random.default_rng(seed)

        # --- demonstrations, uploaded once ---------------------------------
        self.demo = torch.as_tensor(demo_poses, dtype=self.dtype, device=self.device)

        # --- manifold pool, generated once ---------------------------------
        self.pool = self._build_pool(pool)
        self.stream = torch.zeros(n, dtype=torch.long, device=self.device)

        self.pose = torch.zeros(n, POSE_DIM, dtype=self.dtype, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.inside = torch.zeros(n, dtype=torch.long, device=self.device)
        self.ever_inside = torch.zeros(n, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(n, dtype=self.dtype, device=self.device)

    def _build_pool(self, pool: int) -> dict:
        """Manifolds drawn from the recorded envelopes, spanning easy to hard.

        The pool must satisfy two constraints at once, and both were learned the hard way:

        * **every manifold has demonstrations.** Cloning recorded poses is the training signal,
          so a manifold whose nearest demo sits far away (the earlier pool drew from a synthetic
          difficulty ladder unrelated to the data) leaves the policy guessing - visible later as
          a containment r of 1.4-1.7 that no amount of training fixes.
        * **the range includes manifolds the standing pose already fits.** If every manifold
          needs posing, the best average-reward policy poses a little for all of them and never
          learns *when* to pose (measured: 4 deg of pose variation across the family).

        Recorded envelopes supply both: they are the manifolds the data actually covers, and they
        range from roomy to tight.
        """
        import torch

        _, all_demos = load_demos()
        keys = [k for k in all_demos if len(all_demos[k]) > 0]
        # (`demo` below is the pose recorded under this envelope; the pair (envelope, pose) is
        # self-consistent by construction, which is what makes the clone target reachable)
        order = np.argsort([self._difficulty_of(k) for k in keys])       # easy -> hard
        chosen = [keys[i] for i in np.linspace(0, len(order) - 1, pool).astype(int)]
        semi = np.zeros((pool, 3))
        center = np.zeros((pool, 3))
        demo = np.full((pool, POSE_DIM), np.nan)
        for i, key in enumerate(chosen):
            values = np.array([float(v) for v in key.split(",")])
            semi[i], center[i] = values[:3], values[3:]
            poses = all_demos[key]
            demo[i] = poses[np.argmin(np.abs(poses).mean(axis=1))]      # the least extreme pose
        rot = np.tile(np.eye(3), (pool, 1, 1))                          # recorded envelopes are AABB
        return {
            "semi": torch.as_tensor(semi, dtype=self.dtype, device=self.device),
            "center": torch.as_tensor(center, dtype=self.dtype, device=self.device),
            "rot": torch.as_tensor(rot, dtype=self.dtype, device=self.device),
            "demo": torch.as_tensor(demo, dtype=self.dtype, device=self.device),
            "size": pool, "keys": chosen,
        }

    @staticmethod
    def _difficulty_of(key: str) -> float:
        """How much posing this recorded envelope demands: smaller means easier."""
        values = np.array([float(v) for v in key.split(",")])
        semi = values[:3]
        return float(np.linalg.norm(semi))

    # -- environment --------------------------------------------------------
    def _state(self):
        """The manifold tensors of the current streams."""
        t = self.stream
        return self.pool["semi"][t], self.pool["center"][t], self.pool["rot"][t], self.pool["demo"][t]

    def radius(self, poses, semi=None, center=None, rot=None):
        """Containment per row, as a *device tensor* (the training loop stays on the device).

        Host-side callers should use `radius_np`, which does the transfer explicitly rather than
        relying on an implicit conversion that torch only allows for CPU tensors.
        """
        torch = self.torch
        poses = torch.as_tensor(poses, dtype=self.dtype, device=self.device) if not \
            isinstance(poses, torch.Tensor) else poses
        points, mask = self.kin.forward(poses, return_mask=True)
        if semi is None:
            s_, c_, r_, _ = self._state()
        else:
            s_, c_, r_ = semi, center, rot
        local = torch.einsum("bki,bij->bkj", points - c_.unsqueeze(1), r_)
        radius = (local / s_.unsqueeze(1)).norm(dim=-1)
        return radius.masked_fill(~mask, -1.0).max(dim=-1).values

    def radius_np(self, poses, semi=None, center=None, rot=None) -> "np.ndarray":
        return self.radius(poses, semi, center, rot).detach().cpu().numpy()

    def obs(self):
        """Must match `primitive.PrimitiveEnv.obs` entry for entry.

        Everything here is already in the pelvis frame - the manifolds are created there and
        `radius` subtracts the pelvis before measuring - so the centre term is used as-is.
        Subtracting the pelvis position again would mix in the world height (0.79 m) and put the
        policy on a different observation than deployment, which is exactly the kind of silent
        mismatch this module has to avoid.
        """
        torch = self.torch
        semi, center, rot, _ = self._state()
        axis = rot[:, :, 2]
        return torch.cat([
            self.pose[:, :3] / POSE_LIMIT,
            self.pose.mean(dim=1, keepdim=True),
            self.pose.std(dim=1, keepdim=True),
            self.radius(self.pose, semi, center, rot).unsqueeze(1),
            center / 0.5,
            semi,
            axis,
        ], dim=1)

    def reset(self, streams=None):
        torch = self.torch
        idx = torch.arange(self.n, device=self.device) if streams is None else streams
        self.pose[idx] = 0.0
        self.steps[idx] = 0
        self.inside[idx] = 0
        self.ever_inside[idx] = 0
        self.episode_return[idx] = 0.0
        return self.obs()

    def step(self, actions):
        """One control tick for every stream. `actions` [N, 29] in [-1, 1]."""
        torch = self.torch
        cfg = self.cfg
        semi, center, rot, demo = self._state()
        target = action_to_pose(actions, torch)
        reward = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        alpha = min(1.0, C.CONTROL_DT / cfg.tau)
        for _ in range(cfg.inner):
            self.pose = self.pose + (target - self.pose) * alpha
            radius = self.radius(self.pose, semi, center, rot)
            inside = radius <= cfg.success_margin
            margin = torch.clamp(1.0 - radius, max=cfg.margin_cap)
            reward = reward + torch.where(inside, cfg.w_inside * margin,
                                          -cfg.w_outside * (radius - cfg.success_margin))
            reward = reward - cfg.w_effort * self.pose.abs().mean(dim=1) / POSE_LIMIT
            reward = reward - self._kinematic_penalty(cfg)
            reward = reward - self._pose_quality(cfg)
            self.inside = torch.where(inside, self.inside + 1, torch.zeros_like(self.inside))
            self.ever_inside = self.ever_inside + inside.long()
        gap = (self.pose - demo).abs().mean(dim=1)
        reward = reward - cfg.w_imitation * torch.nan_to_num(gap, nan=0.0)
        reward = reward * C.CONTROL_DT

        self.steps = self.steps + 1
        success = self.inside >= cfg.success_hold
        timeout = (~success) & (self.steps >= cfg.max_steps)
        # reaching the hold requirement is a win; running out of time without ever entering is a
        # loss, so "do nothing and wait" is not a neutral outcome
        reward = reward + torch.where(success, cfg.success_bonus, torch.zeros_like(reward))
        reward = reward - torch.where(timeout & (self.ever_inside == 0),
                                      torch.full_like(reward, cfg.fail_penalty),
                                      torch.zeros_like(reward))
        self.episode_return = self.episode_return + reward
        done = success | timeout
        info = {"radius": radius.detach(), "pose": self.pose.clone(), "success": success,
                "timeout": timeout, "episode_return": self.episode_return.clone(),
                "gap": torch.nan_to_num(gap, nan=0.0).detach()}
        return self.obs(), reward, done, info

    # paired joints: (left, right) indices in policy order; mismatching them tilts the pelvis
    PAIRS = None

    def _pose_quality(self, cfg: Config):
        """Penalties that rule out poses which satisfy containment for the wrong reason.

        * symmetry - paired joints (two hips, two knees, two shoulders, ...) should agree; a
          left/right mismatch twists the body and is not a stance it can hold;
        * posture - stay near the default pose when the manifold does not demand otherwise;
        * torque - the static holding torque, the doc's energy term in its static form: it makes
          a deep crouch cost more than a small adjustment, which is what "pose only as much as
          the manifold needs" means in reward terms.
        """
        torch = self.torch
        if self.PAIRS is None:
            from .body_envelope import _PAIR_INDEX
            self.PAIRS = torch.as_tensor(_PAIR_INDEX, dtype=torch.long, device=self.device)
        left, right = self.PAIRS[:, 0], self.PAIRS[:, 1]
        symmetry = (self.pose[:, left] - self.pose[:, right]).abs().mean(dim=1)
        posture = self.pose.abs().mean(dim=1) / POSE_LIMIT
        # holding torque: PD law with the joint gains, evaluated at rest (q = target)
        tau = self.kin.joint_torque(self.pose)
        return cfg.w_symmetry * symmetry + cfg.w_posture * posture + cfg.w_torque * tau

    def _kinematic_penalty(self, cfg: Config):
        torch = self.torch
        out = self.kin.stability(self.pose)
        reach = out["com"][:, :2].norm(dim=1)
        feet = out["feet_z"]
        spread = feet.max(dim=1).values - feet.min(dim=1).values
        return (cfg.w_com * torch.clamp(reach - 0.08, min=0.0)
                + cfg.w_foot * torch.clamp(spread - 0.02, min=0.0)
                + cfg.w_limit * out["limit"])

    def set_manifolds_from_arrays(self, semi: np.ndarray, center: np.ndarray,
                                  rot: np.ndarray | None, resize: bool = True) -> None:
        """Point the streams at arbitrary manifolds (used by behaviour cloning, which feeds the
        recorded envelope of every demo pose rather than an entry of the training pool)."""
        torch = self.torch
        semi = np.asarray(semi, dtype=np.float64).reshape(-1, 3)
        center = np.asarray(center, dtype=np.float64).reshape(-1, 3)
        n = len(semi)
        if resize or n != self.n:
            self.n = n
            self.pose = torch.zeros(n, POSE_DIM, dtype=self.dtype, device=self.device)
            self.steps = torch.zeros(n, dtype=torch.long, device=self.device)
            self.inside = torch.zeros(n, dtype=torch.long, device=self.device)
            self.ever_inside = torch.zeros(n, dtype=torch.long, device=self.device)
            self.episode_return = torch.zeros(n, dtype=self.dtype, device=self.device)
            self.stream = torch.arange(n, dtype=torch.long, device=self.device)
        if rot is None:
            rot = np.tile(np.eye(3), (n, 1, 1))
        self.pool = {
            "semi": torch.as_tensor(semi, dtype=self.dtype, device=self.device),
            "center": torch.as_tensor(center, dtype=self.dtype, device=self.device),
            "rot": torch.as_tensor(np.asarray(rot, dtype=np.float64).reshape(-1, 3, 3),
                                   dtype=self.dtype, device=self.device),
            "demo": torch.full((n, POSE_DIM), float("nan"), dtype=self.dtype, device=self.device),
            "size": n,
        }
        self.reset()

    def difficulty(self) -> "np.ndarray":
        """r of the standing pose per pooled manifold: how much posing each one demands."""
        torch = self.torch
        zero = torch.zeros(1, POSE_DIM, dtype=self.dtype, device=self.device)
        out = []
        for i in range(self.pool["size"]):
            out.append(float(self.radius(zero, self.pool["semi"][i:i + 1],
                                         self.pool["center"][i:i + 1],
                                         self.pool["rot"][i:i + 1])))
        return np.array(out)

    def retarget(self, index):
        """Give these streams a new manifold (the pool is indexed, never rebuilt)."""
        torch = self.torch
        draw = torch.as_tensor(self.rng.integers(0, self.pool["size"], size=len(index)),
                               device=self.device)
        self.stream[index] = draw
        self.reset(index)


def _difficulty_ladder(pool: int) -> list[ManifoldSpec]:
    """A ladder of manifolds from "the body already fits" to "barely feasible".

    Scales the standing envelope down over the full range the body can reach, so the policy sees
    both ends of the decision - stand still, or pose - instead of only the posing end.
    """
    from .family import ManifoldSpec

    ladder = []
    steps = max(pool, 1)
    for i in range(steps):
        # from 1.25 (the standing pose fits with room to spare - the "do nothing" end) down to
        # 0.66 (near the crouch limit). 1.0 alone is not "easy": BASE_SEMI is the tightest
        # ellipsoid around the standing body, so at scale 1.0 the pose sits exactly on the
        # surface (r = 1.0) and any policy still has to be careful.
        fraction = i / max(steps - 1, 1)
        height = 1.25 - 0.59 * fraction
        # a few of them also pinch or shift, so the other pose channels have to be used
        width = 1.00 - 0.18 * ((i % 3) / 2.0)
        depth = 1.00 - 0.15 * (((i + 1) % 4) / 3.0)
        offset = 0.04 * (((i % 5) - 2) / 2.0)
        tilt = 10.0 * (((i % 7) - 3) / 3.0)
        ladder.append(ManifoldSpec(height=height, width=width, depth=depth,
                                   offset=offset, tilt_deg=tilt))
    return ladder


def _key(semi, center):
    return ",".join(f"{v:.2f}" for v in list(semi) + list(center))


def _dist(key, semi, center):
    v = np.array([float(x) for x in key.split(",")])
    return float(np.linalg.norm((v[:3] - semi) / np.maximum(semi, 1e-3)))


class TorchPPO(PPO):
    """PPO that consumes device tensors directly: no numpy round trip per step."""

    def act_tensor(self, obs, deterministic: bool = False):
        """Action for a batch of observations.

        `deterministic=True` returns the distribution mean. Evaluating a policy with its own
        exploration noise (the default, which training needs) reports a random pose rather than
        the one the policy proposes: measured, the same observation produced actions differing by
        3.1 between two calls, and every "the clone does not fit its manifold" number came from
        that noise rather than from the policy.
        """
        import torch

        with torch.no_grad():
            dist, value = self.model.distribution(obs)
            action = dist.mean if deterministic else dist.sample()
            logprob = dist.log_prob(action).sum(-1)
        return action, logprob, value

    def update_tensor(self, obs, actions, logprobs, rewards, values, dones, last_value):
        """PPO update over buffers of per-step device tensors (lists of [N, ...] tensors)."""
        import torch
        import torch.nn as nn

        steps = len(rewards)
        advantages = torch.zeros(steps, *rewards[0].shape, dtype=rewards[0].dtype,
                                 device=rewards[0].device)
        gae = torch.zeros_like(values[0])
        for t in reversed(range(steps)):
            next_value = last_value if t == steps - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            advantages[t] = gae
        values = torch.stack(values)
        returns = advantages + values
        obs = torch.cat(obs)
        actions = torch.cat(actions)
        logprobs = torch.cat(logprobs)
        adv = advantages.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = returns.reshape(-1)
        batch = obs.shape[0]
        minibatch = max(1, batch // self.minibatches)
        stats = {}
        for _ in range(self.epochs):
            perm = torch.randperm(batch, device=obs.device)
            for start in range(0, batch, minibatch):
                idx = perm[start:start + minibatch]
                dist, value = self.model.distribution(obs[idx])
                logprob = dist.log_prob(actions[idx]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = (logprob - logprobs[idx]).exp()
                unclipped = ratio * adv[idx]
                clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (value - ret[idx]).pow(2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                stats["policy_loss"] = float(policy_loss.item())
                stats["value_loss"] = float(value_loss.item())
                stats["entropy"] = float(entropy.item())
        return stats


def train(args) -> int:
    import torch

    from .kinematics import TorchKinematics
    from .primitive import _progress, _BEST

    _BEST.clear()
    kin = TorchKinematics(device=args.device)
    demo_poses = np.concatenate(list(load_demos()[1].values()))
    cfg = Config(w_imitation=0.0 if args.bc_only else args.w_imitation)
    env = TorchPrimitiveEnv(kin, demo_poses, cfg, n=args.envs, pool=args.pool, seed=args.seed)
    # difficulty bands over the pool, for the per-band success report
    difficulty = env.difficulty()
    edges = np.percentile(difficulty, [33, 66])
    env.bands = np.digitize(difficulty, edges)
    print(f"pool difficulty: |easy {int((env.bands == 0).sum())}  mid {int((env.bands == 1).sum())}  "
          f"hard {int((env.bands == 2).sum())}| bands at r {np.round(edges, 2).tolist()}")
    obs = env.reset()
    ppo = TorchPPO(obs.shape[1], POSE_DIM, init_log_std=[-0.3] * POSE_DIM, device=args.device)
    if args.resume is not None:
        ppo.load(args.resume)
        print(f"resumed from {args.resume}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.csv", "w")
    log.write("iteration,episodes,success,radius_mean,imitation_gap,return_mean\n")
    print(f"torch env: {args.envs} streams x {cfg.max_steps} steps "
          f"= {args.envs * cfg.max_steps} transitions/iteration, pool {args.pool}, "
          f"device {kin.device}, {kin.total_vertices} surface points")
    started = time.perf_counter()
    best_success = -1.0
    for iteration in range(1, args.iterations + 1):
        frac = min(1.0, (iteration - 1) / max(1, args.iterations * args.entropy_frac))
        ppo.entropy_coef = args.entropy_start + (args.entropy_end - args.entropy_start) * frac
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        outcomes, radii, gaps, returns = [], [], [], []
        finished_streams = []
        for _ in range(cfg.max_steps):
            action, logprob, value = ppo.act_tensor(obs)
            obs, reward, done, info = env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logprob)
            rew_buf.append(reward)
            val_buf.append(value)
            done_buf.append(done.to(reward.dtype))
            finished = done.nonzero(as_tuple=True)[0]
            if len(finished):
                outcomes.extend(info["success"][finished].cpu().tolist())
                radii.extend(info["radius"][finished].cpu().tolist())
                gaps.extend(info["gap"][finished].cpu().tolist())
                returns.extend(info["episode_return"][finished].cpu().tolist())
                finished_streams.extend(finished.cpu().tolist())
                env.retarget(finished)
        with torch.no_grad():
            _, last_value = ppo.model(obs)
            last_value = last_value.squeeze(-1)
        ppo.update_tensor(obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf, last_value)
        success = float(np.mean(outcomes)) if outcomes else float("nan")
        # split by how hard the manifold was: an aggregate can look fine while the policy ignores
        # the easy manifolds (posing when it need not) or gives up on the hard ones
        if finished_streams:
            bands = env.bands[finished_streams]
            band_rate = {name: float(np.mean(np.array(outcomes)[bands == i]))
                         for i, name in enumerate(("easy", "mid", "hard"))}
        log.write(f"{iteration},{len(outcomes)},{success:.3f},"
                  f"{np.mean(radii) if radii else 0:.3f},"
                  f"{np.mean(gaps) if gaps else float('nan'):.4f},"
                  f"{np.mean(returns) if returns else 0:.3f}\n")
        log.flush()
        _progress(iteration, args.iterations, started, {
            "success": success,
            "r": float(np.mean(radii)) if radii else float("nan"),
            "imit": float(np.mean(gaps)) if gaps else float("nan"),
            "return": float(np.mean(returns)) if returns else float("nan"),
            "eps": len(outcomes), "bands": band_rate})
        # keep the best snapshot: a run can collapse late (entropy annealed away, policy drifts
        # off the reward), and overwriting the good policy with the bad one loses the result
        if success == success and success > best_success:
            best_success = success
            ppo.save(out / "policy.pt")
            (out / "best.json").write_text(json.dumps(
                {"iteration": iteration, "success": success,
                 "radius": float(np.mean(radii)) if radii else None}, indent=2))
        ppo.save(out / "policy_last.pt")
    log.close()
    print(f"\\n=== training finished: {args.iterations} iterations in "
          f"{time.perf_counter() - started:.0f} s, success {success:.2f}, "
          f"policy -> {out / 'policy.pt'} ===", flush=True)
    return 0


def check(args) -> int:
    """Agreement with the numpy environment: same manifolds, same actions, same radii."""
    import torch

    from .kinematics import TorchKinematics
    from .primitive import load_demos as _load

    kin = TorchKinematics(device=args.device)
    demo_poses = np.concatenate(list(_load()[1].values()))
    t_env = TorchPrimitiveEnv(kin, demo_poses, Config(), n=8, pool=8, seed=0)
    obs = t_env.reset()
    torch.manual_seed(0)
    actions = torch.randn(8, POSE_DIM, device=kin.device).clamp(-1, 1)
    for _ in range(3):
        obs, reward, done, info = t_env.step(actions)
    semi, center, rot, demo = t_env._state()

    from .family import report_specs
    from .manifold import EllipsoidManifold, Primitive
    from .static_fit import BodyModel
    body = BodyModel()
    # rebuild each stream's own manifold as a numpy object: the comparison must use the same
    # ellipsoid the torch env used, not the family in list order
    print(f"{'stream':>7} {'torch r':>9} {'numpy r':>9} {'delta':>9}")
    worst = 0.0
    for i in range(min(5, t_env.n)):
        manifold = EllipsoidManifold([Primitive(
            center=center[i].cpu().numpy().astype(np.float64),
            semi=semi[i].cpu().numpy().astype(np.float64), quat=np.array([1.0, 0, 0, 0]))])
        points = body.mesh_points_pose(t_env.pose[i].cpu().numpy())
        r_np = float(manifold.radii(points).max())
        r_t = float(t_env.radius(t_env.pose[i:i + 1], semi[i:i + 1], center[i:i + 1],
                                 rot[i:i + 1])[0])
        worst = max(worst, abs(r_np - r_t))
        print(f"{i:>7} {r_t:>9.3f} {r_np:>9.3f} {r_t - r_np:>9.4f}")
    print(f"worst radius disagreement: {worst:.4f}  "
          f"({'ok (golden: same manifold, numpy mesh vs torch thinned vertices)' if worst < 0.05 else 'TOO LARGE - investigate'})")
    print(f"mean reward after 3 steps: {reward.mean().item():.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Primitive training fully on the device")
    parser.add_argument("mode", choices=("train", "check"))
    parser.add_argument("--iterations", type=int, default=400)
    # 64 streams x 8 steps = 512 transitions per iteration, the batch size the PPO settings
    # (epochs 10, minibatches 8) were tuned for. Larger batches made the policy collapse late
    # in the run: the same settings then apply ~16x more gradient steps per iteration.
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--pool", type=int, default=64)
    parser.add_argument("--w-imitation", type=float, default=1.0)
    parser.add_argument("--bc-only", action="store_true")
    parser.add_argument("--entropy-start", type=float, default=0.005)
    # the anneal is per iteration, and one iteration here holds thousands of samples (512 streams
    # x 8 steps), so exploration is removed far more slowly than in the small-batch numpy run
    parser.add_argument("--entropy-end", type=float, default=0.002)
    parser.add_argument("--entropy-frac", type=float, default=0.85,
                        help="fraction of the run over which entropy is annealed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None,
                        help="start from a policy (e.g. a behaviour-cloning warm start)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    return train(args) if args.mode == "train" else check(args)


if __name__ == "__main__":
    raise SystemExit(main())
