"""Flat-ground MuJoCo environment for the G1 with the deploy's PD torque control."""

from __future__ import annotations

import numpy as np
import mujoco

from . import constants as C


class G1FlatEnv:
    """Single G1 on a plane, torque-controlled through the frozen SONIC action interface."""

    def __init__(self, scene_path=C.FLAT_SCENE, sim_dt: float = C.SIM_DT,
                 decimation: int = C.DECIMATION, hand_kp: float = 20.0, hand_kd: float = 1.0):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.model.opt.timestep = sim_dt
        self.data = mujoco.MjData(self.model)
        self.decimation = int(decimation)
        self.hand_kp = hand_kp
        self.hand_kd = hand_kd

        self.body_act = np.array([self._actuator_id(name) for name in C.MOTOR_NAMES])
        self.hand_act = np.array([self._actuator_id(name) for name in C.HAND_MOTOR_NAMES])
        self.body_qadr, self.body_dadr = self._joint_addrs(self.body_act)
        self.hand_qadr, self.hand_dadr = self._joint_addrs(self.hand_act)

        self.ctrl_min = self.model.actuator_ctrlrange[self.hand_act, 0].copy()
        self.ctrl_max = self.model.actuator_ctrlrange[self.hand_act, 1].copy()

        self.q_des = C.DEFAULT_ANGLES.copy()
        self.time = 0.0

    def _actuator_id(self, name: str) -> int:
        act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id < 0:
            raise KeyError(f"actuator '{name}' not found in {self.model} model")
        return act_id

    def _joint_addrs(self, act_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joint_ids = self.model.actuator_trnid[act_ids, 0]
        return self.model.jnt_qposadr[joint_ids].copy(), self.model.jnt_dofadr[joint_ids].copy()

    def reset(self, height: float = C.DEFAULT_HEIGHT, x: float = 0.0,
              joint_offset: np.ndarray | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = (float(x), 0.0, height)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.body_qadr] = C.DEFAULT_ANGLES + (
            0.0 if joint_offset is None else np.asarray(joint_offset, dtype=np.float64))
        self.data.qpos[self.hand_qadr] = 0.0
        self.data.qvel[:] = 0.0
        self.q_des = C.DEFAULT_ANGLES.copy()
        self.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_target(self, q_des_hw: np.ndarray) -> None:
        self.q_des = np.asarray(q_des_hw, dtype=np.float64).copy()

    def _apply_torques(self) -> None:
        q = self.data.qpos[self.body_qadr]
        dq = self.data.qvel[self.body_dadr]
        torque = C.KP * (self.q_des - q) - C.KD * dq
        self.data.ctrl[self.body_act] = np.clip(torque, -C.EFFORT_LIMITS, C.EFFORT_LIMITS)

        hand_q = self.data.qpos[self.hand_qadr]
        hand_dq = self.data.qvel[self.hand_dadr]
        hand_tau = self.hand_kp * (0.0 - hand_q) - self.hand_kd * hand_dq
        self.data.ctrl[self.hand_act] = np.clip(hand_tau, self.ctrl_min, self.ctrl_max)

    def step(self) -> None:
        for _ in range(self.decimation):
            self._apply_torques()
            mujoco.mj_step(self.model, self.data)
        self.time += self.decimation * self.model.opt.timestep

    def state(self) -> dict:
        return {
            "base_pos": self.data.qpos[0:3].copy(),
            "base_quat": self.data.qpos[3:7].copy(),
            "base_lin_vel": self.data.qvel[0:3].copy(),
            "base_ang_vel": self.data.qvel[3:6].copy(),
            "q_hw": self.data.qpos[self.body_qadr].copy(),
            "dq_hw": self.data.qvel[self.body_dadr].copy(),
        }
