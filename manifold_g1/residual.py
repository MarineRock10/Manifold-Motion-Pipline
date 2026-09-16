"""SONIC execution residual: what the frozen controller actually does with a commanded pose.

The geometric model judges a pose by kinematics alone, so it happily proposes poses the
controller will not adopt - a keyframe that asks for an arm out to the side comes back with the
arm roughly where it was. Training against the geometry therefore optimises a target that the
execution layer does not share.

Rather than putting SONIC in the training loop (a keyframe hold costs ~1.5 s of physics, six
orders of magnitude above the geometric step), the discrepancy is learned instead:

    q_achieved  ~=  q_commanded + delta(q_commanded)

The pairs needed to fit `delta` are already on disk: every collected clip stores both the
commanded reference and the measured joints per tick. The residual model is then used *inside*
the fast geometric environment, so containment there means containment after execution.

    python3 -m manifold_g1.residual fit      # -> reports/manifold_g1/sonic_residual.pt
    python3 -m manifold_g1.residual show     # per-joint residual statistics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .demos import DEFAULT_ISAAC

CLIPS = Path("reports/manifold_g1/clips")
OUT = C.REPO / "reports" / "manifold_g1" / "sonic_residual.pt"
GROUPS = {"legs": slice(0, 12), "waist": slice(12, 15), "arms": slice(15, 29)}


def collect_pairs(clips: Path = CLIPS, only_raw: bool = True,
                  static: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Every (commanded, achieved) joint pair, in policy order.

    `static` (from `residual_data.py`) is the part that matters: walking pairs sit in the joint
    range the controller tracks well (residual ~0.05 rad), while held keyframes reach ~0.15 rad.
    Mixing both keeps the model honest in the regime the primitive actually uses.
    """
    commanded, achieved = [], []
    if static is not None and Path(static).exists():
        with np.load(static) as handle:
            commanded.append(handle["q_cmd"])
            achieved.append(handle["q_act"])
        print(f"  including {len(commanded[0])} static-pose pairs from {static}")
    for path in sorted(clips.glob("*.npz")):
        if only_raw and (path.stem.startswith("mode") or path.stem.startswith("cal_")):
            pass                                    # keep: these are held-keyframe segments too
        with np.load(path) as handle:
            if "q_cmd" not in handle.files or "q_act" not in handle.files:
                continue
            commanded.append(handle["q_cmd"])
            achieved.append(handle["q_act"])
    if not commanded:
        raise SystemExit(f"no (q_cmd, q_act) pairs under {clips}")
    return (np.concatenate(commanded) - DEFAULT_ISAAC,
            np.concatenate(achieved) - DEFAULT_ISAAC)


class ResidualModel:
    """A small MLP: commanded joint delta -> the delta the controller adds to it."""

    def __init__(self, hidden: int = 128):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.net = nn.Sequential(nn.Linear(29, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 29))
        self.scale = None

    def fit(self, commanded: np.ndarray, achieved: np.ndarray, epochs: int = 400,
            lr: float = 1e-3, device: str = "cuda", seed: int = 0):
        import torch

        torch.manual_seed(seed)
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.net.to(dev)
        x = torch.as_tensor(commanded, dtype=torch.float32, device=dev)
        y = torch.as_tensor(achieved - commanded, dtype=torch.float32, device=dev)
        self.scale = (x.std(dim=0).mean().item(), y.std(dim=0).mean().item())
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        split = int(0.9 * len(x))
        for epoch in range(epochs):
            perm = torch.randperm(len(x), device=dev)
            for start in range(0, split, 4096):
                idx = perm[start:start + 4096]
                pred = self.net(x[idx])
                loss = (pred - y[idx]).pow(2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if (epoch + 1) % max(1, epochs // 8) == 0:
                with torch.no_grad():
                    val = (self.net(x[split:]) - y[split:]).pow(2).mean().item()
                print(f"  epoch {epoch + 1:4d}  train {loss.item():.3e}  val {val:.3e}", flush=True)
        return self

    def predict(self, commanded: np.ndarray) -> np.ndarray:
        """Achieved joints the controller would reach for these commanded deltas."""
        import torch

        dev = next(self.net.parameters()).device
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(commanded, dtype=np.float64), dtype=torch.float32,
                                device=dev)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            out = commanded + self.net(x).cpu().numpy()
        return out

    def save(self, path: Path = OUT) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(),
                    "meta": {"pose_dim": 29, "scale": self.scale}}, path)

    @classmethod
    def load(cls, path: Path = OUT) -> "ResidualModel":
        import torch

        model = cls()
        blob = torch.load(path, map_location="cpu")
        model.net.load_state_dict(blob["state"])
        model.scale = blob["meta"].get("scale")
        return model


def fit(args) -> int:
    commanded, achieved = collect_pairs(args.clips, static=args.static)
    print(f"pairs: {len(commanded)}  commanded |dq| mean {np.abs(commanded).mean():.3f} rad")
    residual = achieved - commanded
    print(f"residual |dq|: mean {np.abs(residual).mean():.4f}  "
          f"per group " + ", ".join(
              f"{name} {np.abs(residual[:, sl]).mean():.4f}" for name, sl in GROUPS.items()))
    model = ResidualModel().fit(commanded, achieved, args.epochs, args.lr, args.device, args.seed)
    model.save(args.out)
    # held-out quality, expressed in the units that matter: does it predict the achievement?
    rng = np.random.default_rng(0)
    idx = rng.choice(len(commanded), min(2000, len(commanded)), replace=False)
    pred = model.predict(commanded[idx])
    err = np.abs(pred - achieved[idx])
    baseline = np.abs(commanded[idx] - achieved[idx])
    print(f"\nheld-out {len(idx)} samples:")
    print(f"  without the model (assume achieved = commanded): |err| {baseline.mean():.4f} rad")
    print(f"  with the residual model                        : |err| {err.mean():.4f} rad")
    print(f"  reduction {100 * (1 - err.mean() / baseline.mean()):.0f}%  -> "
          f"{'useful' if err.mean() < 0.7 * baseline.mean() else 'NOT better than ignoring it'}")
    print(f"saved {args.out}")
    return 0


def show(args) -> int:
    commanded, achieved = collect_pairs(args.clips, static=args.static)
    residual = achieved - commanded
    print(f"{'joint':>24} {'cmd |dq|':>9} {'residual':>9} {'std':>8}")
    order = np.argsort(-np.abs(residual).mean(axis=0))[:12]
    for i in order:
        name = C.MOTOR_NAMES[int(C.ISAACLAB_TO_MUJOCO[i])]
        print(f"{name:>24} {np.abs(commanded[:, i]).mean():>9.3f} "
              f"{residual[:, i].mean():>+9.3f} {residual[:, i].std():>8.3f}")
    for name, sl in GROUPS.items():
        print(f"{name:>24} {np.abs(commanded[:, sl]).mean():>9.3f} "
              f"{np.abs(residual[:, sl]).mean():>9.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SONIC execution-residual model")
    parser.add_argument("mode", choices=("fit", "show"))
    parser.add_argument("--clips", type=Path, default=CLIPS)
    parser.add_argument("--static", type=Path,
                        default=C.REPO / "reports" / "manifold_g1" / "residual_pairs.npz")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    return fit(args) if args.mode == "fit" else show(args)


if __name__ == "__main__":
    raise SystemExit(main())
