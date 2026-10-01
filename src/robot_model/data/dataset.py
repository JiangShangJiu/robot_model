"""观测数据：沿轨迹采样，可注入测量噪声。

力矩来源由调用方以 ``tau_func`` 注入（MuJoCo ``mj_inverse``、真机记录、
或本库自己的逆动力学），所以这一层不依赖任何仿真器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from ..excitation.trajectory import FourierTrajectory


@dataclass(frozen=True)
class MotionData:
    """一段运动的观测：形状均为 ``(n_samples, dof)``。"""

    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    tau: np.ndarray

    @property
    def n_samples(self) -> int:
        return self.q.shape[0]

    @property
    def dof(self) -> int:
        return self.q.shape[1]

    def __len__(self) -> int:
        return self.n_samples

    def subset(self, index) -> MotionData:
        """按索引取子集，用于划分训练 / 验证段。"""
        return MotionData(
            t=self.t[index],
            q=self.q[index],
            dq=self.dq[index],
            ddq=self.ddq[index],
            tau=self.tau[index],
        )


def sample_trajectory(
    trajectory: FourierTrajectory,
    tau_func: Callable,
    *,
    periods: int = 1,
    rate: float = 100.0,
) -> MotionData:
    """沿轨迹逐点调用 ``tau_func(q, dq, ddq) -> tau`` 采集数据。"""
    t = trajectory.timestamps(periods=periods, rate=rate)
    q, dq, ddq = trajectory.evaluate(t)
    tau = np.array(
        [
            np.asarray(tau_func(q[i], dq[i], ddq[i]), dtype=float).reshape(-1)
            for i in range(t.size)
        ]
    )
    return MotionData(t=t, q=q, dq=dq, ddq=ddq, tau=tau)


def drop_low_speed(data: MotionData, threshold: float) -> MotionData:
    """剔除任一关节速度接近零的采样点。

    库仑摩擦列使用 ``sign(dq)``，不能充分描述过零附近的摩擦。
    一个时刻只要有任一关节低于阈值，就剔除该时刻的全部关节数据。
    """
    if threshold <= 0:
        return data
    keep = np.where(np.all(np.abs(data.dq) >= threshold, axis=1))[0]
    if keep.size == 0:
        raise ValueError(f"阈值 {threshold} 过大，没有采样点保留")
    return data.subset(keep)


def align_measurements(data: MotionData, *, encoder_delay_steps=0, torque_delay_steps=0) -> MotionData:
    """离线补偿已知的整数采样延迟，仅重新索引测量数据，不使用真值。

    返回时间戳表示信号对应的时刻，而非接收时刻。未知延迟应先标定；
    此函数不补偿测速滤波器相位，也不猜测延迟。
    """
    from ..utils.validation import integer

    e = integer(encoder_delay_steps, "encoder_delay_steps")
    r = integer(torque_delay_steps, "torque_delay_steps")
    n = len(data) - max(e, r)
    if n < 3 or not np.allclose(np.diff(data.t), np.diff(data.t)[0]) or np.diff(data.t)[0] <= 0:
        raise ValueError("延迟补偿要求至少 3 个剩余样本和递增、等间隔时间戳")
    return MotionData(data.t[:n], data.q[e:e+n], data.dq[e:e+n],
                      data.ddq[e:e+n], data.tau[r:r+n])


def filter_measurements(
    data: MotionData,
    *,
    cutoff: float,
    order: int = 4,
    recompute_acceleration: bool = True,
) -> MotionData:
    """对 q / dq / tau 做零相位低通；默认顺带由滤波后的 dq 重算 ddq。"""
    from .filtering import estimate_derivatives, zero_phase_lowpass

    dt = float(np.mean(np.diff(data.t)))
    fs = 1.0 / dt
    def smooth(x):
        return zero_phase_lowpass(x, fs=fs, cutoff=cutoff, order=order)

    dq = smooth(data.dq)
    ddq = (
        estimate_derivatives(data.dq, dt=dt, cutoff=cutoff, order=order)
        if recompute_acceleration
        else smooth(data.ddq)
    )
    return MotionData(
        t=data.t, q=smooth(data.q), dq=dq, ddq=ddq, tau=smooth(data.tau)
    )


def add_noise(
    data: MotionData,
    rng: np.random.Generator,
    *,
    tau_std: float = 0.0,
    q_std: float = 0.0,
    dq_std: float = 0.0,
    ddq_std: float = 0.0,
) -> MotionData:
    """加独立高斯白噪声。

    注意这是理想化的：真实系统里 ddq 多半由 q 数值微分得到，
    噪声会被微分放大且在时间上相关，比这里悲观得多。
    """

    def perturb(x, std):
        return x if std == 0.0 else x + rng.normal(0.0, std, size=x.shape)

    return MotionData(
        t=data.t,
        q=perturb(data.q, q_std),
        dq=perturb(data.dq, dq_std),
        ddq=perturb(data.ddq, ddq_std),
        tau=perturb(data.tau, tau_std),
    )
