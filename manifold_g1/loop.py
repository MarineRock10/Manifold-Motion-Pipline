"""Single-environment control loop around the frozen SONIC controller."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import constants as C


@dataclass
class LoopConfig:
    seconds: float = 5.0
    warmup_seconds: float = 0.4
    plan_mode: int = 2
    plan_movement: tuple = (1.0, 0.0, 0.0)
    plan_facing: tuple = (1.0, 0.0, 0.0)
    target_vel: float = -1.0
    height: float = -1.0
    log_every: float = 0.5
    fall_height: float = 0.45


@dataclass
class TickRecord:
    t: float
    base_z: float
    vx: float
    vy: float
    roll: float
    pitch: float
    track_err: float
    action_rms: float


class SonicLoop:
    def __init__(self, env, controller, reference, planner=None, cfg: LoopConfig | None = None,
                 viewer=None, renderer=None, video_stride: int = 5):
        self.env = env
        self.controller = controller
        self.reference = reference
        self.planner = planner
        self.cfg = cfg or LoopConfig()
        self.viewer = viewer
        self.renderer = renderer
        self.video_stride = video_stride
        self.records: list[TickRecord] = []
        self.frames: list[np.ndarray] = []

    def _plan_kwargs(self) -> dict:
        cfg = self.cfg
        return dict(mode=cfg.plan_mode, movement=cfg.plan_movement, facing=cfg.plan_facing,
                    target_vel=cfg.target_vel, height=cfg.height)

    def run(self, verbose: bool = True) -> dict:
        cfg = self.cfg
        warmup_ticks = int(round(cfg.warmup_seconds / C.CONTROL_DT))
        total_ticks = warmup_ticks + int(round(cfg.seconds / C.CONTROL_DT))
        log_stride = max(1, int(round(cfg.log_every / C.CONTROL_DT)))

        self.env.reset()
        self.controller.reset()
        self.records = []

        planned_once = False
        wall_start = time.time()
        for tick in range(total_ticks):
            state = self.env.state()
            self.controller.append_state(state["q_hw"], state["dq_hw"], state["base_quat"],
                                         state["base_ang_vel"])
            if tick < warmup_ticks:
                action = np.zeros(29)
                q_des = C.DEFAULT_ANGLES.copy()
            else:
                if self.planner is not None and not planned_once:
                    self.reference.maybe_replan(self.planner, reserve=10 ** 9, **self._plan_kwargs())
                    planned_once = True
                action, q_des, _ = self.controller.act(self.reference, state["base_quat"])
                self.reference.advance()
                if self.planner is not None:
                    self.reference.maybe_replan(self.planner, **self._plan_kwargs())

            self.env.set_target(q_des)
            self.env.step()
            if self.viewer is not None:
                self.viewer.sync()
            if self.renderer is not None and tick % self.video_stride == 0:
                self.renderer.update_scene(self.env.data)
                self.frames.append(self.renderer.render().copy())

            roll, pitch = _roll_pitch(state["base_quat"])
            record = TickRecord(
                t=self.env.time,
                base_z=float(state["base_pos"][2]),
                vx=float(state["base_lin_vel"][0]),
                vy=float(state["base_lin_vel"][1]),
                roll=roll,
                pitch=pitch,
                track_err=float(np.mean(np.abs(q_des - state["q_hw"]))),
                action_rms=float(np.sqrt(np.mean(np.square(action)))),
            )
            self.records.append(record)
            if verbose and (tick < warmup_ticks or (tick - warmup_ticks) % log_stride == 0):
                phase = "warmup" if tick < warmup_ticks else "policy"
                print(f"[{record.t:6.2f}s {phase}] z={record.base_z:.3f} vx={record.vx:+.3f} "
                      f"roll={np.degrees(record.roll):+6.1f} pitch={np.degrees(record.pitch):+6.1f} "
                      f"track={record.track_err:.4f}", flush=True)

        self._summary = self.summary(wall_start)
        return self._summary

    def summary(self, wall_start: float) -> dict:
        recs = self.records
        policy = [r for r in recs if r.t > self.cfg.warmup_seconds + 0.5]
        arr_z = np.array([r.base_z for r in policy]) if policy else np.zeros(1)
        arr_vx = np.array([r.vx for r in policy]) if policy else np.zeros(1)
        arr_roll = np.array([r.roll for r in policy]) if policy else np.zeros(1)
        fell = bool(arr_z.min() < self.cfg.fall_height) if policy else True
        summary = {
            "ticks": len(recs),
            "duration_s": recs[-1].t if recs else 0.0,
            "wall_time_s": time.time() - wall_start,
            "fell": fell,
            "base_z_mean": float(arr_z.mean()),
            "base_z_min": float(arr_z.min()),
            "base_z_max": float(arr_z.max()),
            "forward_m": float(np.sum([r.vx for r in policy]) * C.CONTROL_DT) if policy else 0.0,
            "vx_mean": float(arr_vx.mean()),
            "roll_rms_deg": float(np.degrees(np.sqrt(np.mean(np.square(arr_roll))))) if policy else 0.0,
            "track_err_mean": float(np.mean([r.track_err for r in policy])) if policy else 0.0,
            "action_rms_mean": float(np.mean([r.action_rms for r in policy])) if policy else 0.0,
        }
        return summary

    def save(self, out_dir: Path, tag: str) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = getattr(self, "_summary", None)
        if summary is None:
            summary = self.summary(time.time())
        (out_dir / f"{tag}.json").write_text(json.dumps(summary, indent=2))
        np.savez(
            out_dir / f"{tag}.npz",
            t=np.array([r.t for r in self.records]),
            base_z=np.array([r.base_z for r in self.records]),
            vx=np.array([r.vx for r in self.records]),
            vy=np.array([r.vy for r in self.records]),
            roll=np.array([r.roll for r in self.records]),
            pitch=np.array([r.pitch for r in self.records]),
            track_err=np.array([r.track_err for r in self.records]),
            action_rms=np.array([r.action_rms for r in self.records]),
        )
        return out_dir / f"{tag}.json"


def _roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)
