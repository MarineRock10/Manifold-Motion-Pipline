"""Single-environment frozen-SONIC loop on flat ground.

Usage:
  python -m manifold_g1.run --mode stand --seconds 5
  python -m manifold_g1.run --mode walk --seconds 8 --out reports/manifold_g1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import G1FlatEnv
from .loop import LoopConfig, SonicLoop
from .planner import SonicPlanner
from .reference import PlannedReference, ReferenceBuffer
from .sonic import SonicController


def main() -> int:
    parser = argparse.ArgumentParser(description="MuJoCo + frozen SONIC single-environment loop")
    parser.add_argument("--mode", choices=("stand", "walk"), default="stand")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--warmup", type=float, default=0.4)
    parser.add_argument("--target-vel", type=float, default=-1.0)
    parser.add_argument("--plan-mode", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("reports/manifold_g1"))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--render", action="store_true", help="open the MuJoCo viewer")
    parser.add_argument("--video", type=Path, default=None, help="write an mp4 of the run")
    args = parser.parse_args()

    env = G1FlatEnv()
    controller = SonicController()
    env.reset()
    robot_quat = env.state()["base_quat"]

    planner = None
    reference: ReferenceBuffer = ReferenceBuffer.static_stand(robot_quat)
    if args.mode == "walk":
        planner = SonicPlanner()
        reference = PlannedReference.static_stand(robot_quat)

    viewer = renderer = None
    if args.render:
        import mujoco
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    if args.video is not None:
        import mujoco
        renderer = mujoco.Renderer(env.model, height=480, width=640)

    cfg = LoopConfig(seconds=args.seconds, warmup_seconds=args.warmup,
                     plan_mode=args.plan_mode, target_vel=args.target_vel)
    loop = SonicLoop(env, controller, reference, planner=planner, cfg=cfg,
                     viewer=viewer, renderer=renderer)
    summary = loop.run()
    tag = args.tag or f"{args.mode}_{args.seconds:g}s"
    path = loop.save(args.out, tag)

    if args.video is not None and loop.frames:
        _write_video(args.video, loop.frames)
    if viewer is not None:
        viewer.close()

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"saved: {path}")
    return 0 if not summary["fell"] else 2


def _write_video(path: Path, frames) -> None:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
        for frame in frames:
            writer.write(frame[:, :, ::-1])
        writer.release()
        print(f"[video] wrote {path}")
        return
    except Exception as exc:  # noqa: BLE001 - fall through to other backends
        print(f"[video] cv2 writer unavailable: {exc}")
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(path, fps=10) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"[video] wrote {path}")
    except Exception as exc:  # noqa: BLE001
        out = path.with_suffix(".npz")
        np.savez_compressed(out, frames=np.stack(frames))
        print(f"[video] no video backend ({exc}); wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
