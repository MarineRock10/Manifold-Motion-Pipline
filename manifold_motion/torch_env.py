"""The batched pose environment, all state on the device.

One class, `TorchPrimitiveEnv`, advances [N] manifold streams step by step with no host round
trip: pose integration, containment, the kinematic penalties and the reward are all torch ops.
It exists for the two things the pipeline actually needs from it:

  * **the observation layout** (`obs`), which is what the clone is trained on and what every
    viewer feeds the policy - so training and deployment cannot drift apart;
  * **containment** (`radius`, `radius_np`), the same ruler the manifold family is defined on,
    computed from the GPU forward kinematics instead of MuJoCo.

It also carries the `Config` the observation and containment depend on. The reward fields in it
are legacy: they configured the batched *geometric* trainer (which optimised a residual model's
prediction of what SONIC would do), and that trainer is gone - the in-the-loop fine-tune
(`sonic_rl.py`) measures the real controller instead. Only `dt`, `inner` and `tau` still affect
anything reachable from here.

    python3 -m manifold_motion.torch_env    # self-check: radii agree with the MuJoCo path
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .demos import demo_values, load_demos
from .pose_policy import POSE_DIM
from .pose_policy import POSE_LIMIT

OUT_DIR = C.REPO / "reports" / "manifold_motion" / "bc"


# `action_to_pose` has exactly one definition, in `pose_policy`; re-exported here because this is
# the module callers already reach for and a second copy is how the action scale drifted apart
# between trainers before.
from .pose_policy import action_to_pose  # noqa: F401


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
            values = demo_values(key)
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
        values = demo_values(key)
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

