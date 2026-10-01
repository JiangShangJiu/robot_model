"""MuJoCo 跟踪采集。

控制器跟踪参考轨迹，采样器记录运动和力矩，并由测量速度估计加速度。
此模块用于比较闭环采集与理想逆动力学数据；带对象扰动和执行器动态的
实验见 realistic.py。MuJoCo 在创建采集器时导入。
"""

from __future__ import annotations

import numpy as np

from ..data.dataset import MotionData
from ..data.filtering import estimate_derivatives
from ..excitation.trajectory import FourierTrajectory


def _require_mujoco():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise ImportError(
            "从 MuJoCo 采集需要 mujoco，请安装：pip install mujoco"
        ) from exc
    return mujoco


class MujocoTrackingSource:
    """用「逆动力学前馈 + PD 反馈」跟踪轨迹，并记录传感器读数。

    前馈来自 ``mj_inverse``，所以 PD 增益只需修正积分误差，可以取得很小。
    增益过大反而会在显式积分下失稳：稳定性大致要求 ``kd * dt`` 远小于
    最小关节惯量，本仓库 3R（最小惯量约 0.02，dt=2 ms）取 kd=20 就发散了。
    """

    def __init__(
        self,
        model,
        *,
        kp: float = 50.0,
        kd: float = 2.0,
        joint_sensors: bool = True,
    ):
        self.mujoco = _require_mujoco()
        self.model = model
        self.data = self.mujoco.MjData(model)
        self._ff = self.mujoco.MjData(model)
        self.kp = float(kp)
        self.kd = float(kd)
        self.dof = model.nu
        self.joint_sensors = joint_sensors and model.nsensor > 0
        self._adr = self._sensor_addresses() if self.joint_sensors else None

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def _sensor_addresses(self) -> dict | None:
        """找 enc{i} / tach{i} / trq{i} 命名的关节传感器；缺任一组就退回读状态。"""
        mj = self.mujoco
        adr = {}
        for prefix in ("enc", "tach", "trq"):
            ids = [
                mj.mj_name2id(
                    self.model, mj.mjtObj.mjOBJ_SENSOR, f"{prefix}{i + 1}"
                )
                for i in range(self.dof)
            ]
            if any(i < 0 for i in ids):
                return None
            adr[prefix] = [int(self.model.sensor_adr[i]) for i in ids]
        return adr

    def _read(self, prefix: str, fallback: np.ndarray) -> np.ndarray:
        if self._adr is None:
            return np.array(fallback, dtype=float)
        return np.array(
            [self.data.sensordata[a] for a in self._adr[prefix]], dtype=float
        )

    def feedforward(self, q, dq, ddq) -> np.ndarray:
        """期望状态下的逆动力学力矩。"""
        d = self._ff
        d.qpos[:] = q
        d.qvel[:] = dq
        d.qacc[:] = ddq
        self.mujoco.mj_inverse(self.model, d)
        return np.array(d.qfrc_inverse, dtype=float)

    def run(
        self,
        trajectory: FourierTrajectory,
        *,
        periods: int = 1,
        decimate: int = 1,
        cutoff: float | None = None,
        sensor_model=None,
    ) -> MotionData:
        """跑一遍跟踪并采集。

        ``cutoff`` 给定时对速度做零相位低通后再微分得加速度；
        ``None`` 则裸微分（只适合无噪声仿真）。

        每次运行重置仿真状态。传感器与直接读状态的后备路径都在
        前向计算后、积分前采样，保证 q / dq / tau 对应同一时刻。

        ``sensor_model`` 给定时在采集后施加测量失真（量化、延迟、
        力矩增益等）。注意控制器用的仍是真实状态——真机上反馈也走
        测量值，但那会让跟踪误差和测量噪声耦合，不利于单独归因。
        """
        mj = self.mujoco
        dt = self.timestep
        n_steps = int(round(trajectory.period * periods / dt))

        mj.mj_resetData(self.model, self.data)
        mj.mj_resetData(self.model, self._ff)
        q0, dq0, _ = trajectory.evaluate(0.0)
        self.data.qpos[:] = q0
        self.data.qvel[:] = dq0

        t_rec, q_rec, dq_rec, tau_rec = [], [], [], []
        for k in range(n_steps):
            t = k * dt
            qd, dqd, ddqd = trajectory.evaluate(t)
            tau_ff = self.feedforward(qd, dqd, ddqd)
            self.data.ctrl[:] = (
                tau_ff
                + self.kp * (qd - self.data.qpos)
                + self.kd * (dqd - self.data.qvel)
            )
            # mj_step 之后 qpos/qvel 已被积分，而 sensordata 对应步前；
            # 统一在积分前读取，避免无传感器路径错开一个物理步。
            mj.mj_forward(self.model, self.data)
            if k % decimate == 0:
                t_rec.append(t)
                q_rec.append(self._read("enc", self.data.qpos))
                dq_rec.append(self._read("tach", self.data.qvel))
                # ctrl 是请求值，限幅或执行器动态会改变实际关节力矩。
                tau_rec.append(self._read("trq", self.data.qfrc_actuator))
            mj.mj_step(self.model, self.data)

        dq = np.array(dq_rec)
        sample_dt = dt * decimate
        data = MotionData(
            t=np.array(t_rec),
            q=np.array(q_rec),
            dq=dq,
            ddq=estimate_derivatives(dq, dt=sample_dt, cutoff=cutoff),
            tau=np.array(tau_rec),
        )
        if sensor_model is None:
            return data
        return sensor_model.apply(data, cutoff=cutoff)
