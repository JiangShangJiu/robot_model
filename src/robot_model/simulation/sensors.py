"""离线测量误差模型：量化、偏置、噪声与通道延迟。

编码器和力矩通道可以配置不同延迟，用于检查信号错位对辨识的影响。
默认关闭各项误差；在线反馈中的传感器模型见 hardware.py。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.dataset import MotionData
from ..data.filtering import differentiate


@dataclass(frozen=True)
class SensorModel:
    """测量链模型；默认全部关闭，等价于理想传感器。"""

    #: 编码器位数（整个 ``encoder_span`` 上的分辨率），None 表示不量化
    encoder_bits: int | None = None
    encoder_span: float = 2 * np.pi
    #: 真机多半没有独立测速，速度由编码器差分而来——量化噪声会被放大
    velocity_from_encoder: bool = False
    #: 各通道延迟步数，差异才是问题所在
    encoder_delay: int = 0
    torque_delay: int = 0
    #: 力矩测量增益误差（电机常数标定偏差），1.0 为准确
    torque_gain: float = 1.0
    #: 力矩测量死区，小于此值读作 0
    torque_deadband: float = 0.0

    @property
    def encoder_lsb(self) -> float | None:
        if self.encoder_bits is None:
            return None
        return self.encoder_span / float(2**self.encoder_bits)

    def quantize(self, q: np.ndarray) -> np.ndarray:
        lsb = self.encoder_lsb
        if lsb is None:
            return np.asarray(q, dtype=float)
        return np.round(np.asarray(q, dtype=float) / lsb) * lsb

    def measure_torque(self, tau: np.ndarray) -> np.ndarray:
        out = np.asarray(tau, dtype=float) * self.torque_gain
        if self.torque_deadband > 0:
            out = np.where(np.abs(out) < self.torque_deadband, 0.0, out)
        return out

    def apply(self, data: MotionData, *, cutoff: float | None = None) -> MotionData:
        """把测量失真施加到一段已采集的数据上。

        延迟通过序列移位实现，并丢掉开头 ``max(延迟)`` 个样本，
        避免用填充值污染。``cutoff`` 给定时用它重算加速度。
        """
        dt = float(np.mean(np.diff(data.t)))

        q = self.quantize(data.q)
        if self.velocity_from_encoder:
            dq = differentiate(q, dt)
        else:
            dq = self.quantize(data.dq) if self.encoder_bits else data.dq
        tau = self.measure_torque(data.tau)

        skip = max(self.encoder_delay, self.torque_delay)
        n = data.n_samples

        def shift(x, delay):
            # 测量比真实滞后 delay 步：x_meas[k] = x_true[k-delay]
            return x[skip - delay : n - delay]

        q, dq = shift(q, self.encoder_delay), shift(dq, self.encoder_delay)
        tau = shift(tau, self.torque_delay)
        t = data.t[skip:]

        from ..data.filtering import estimate_derivatives

        return MotionData(
            t=t,
            q=q,
            dq=dq,
            ddq=estimate_derivatives(dq, dt=dt, cutoff=cutoff),
            tau=tau,
        )


#: 一套有代表性的工业伺服参数，用于测试
TYPICAL_SERVO = SensorModel(
    encoder_bits=17,
    velocity_from_encoder=True,
    encoder_delay=1,
    torque_delay=2,
    torque_gain=1.02,
    torque_deadband=0.01,
)
