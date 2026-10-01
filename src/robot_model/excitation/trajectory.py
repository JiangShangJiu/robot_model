"""有限项傅里叶激励轨迹。

参考位置、速度和加速度都有解析式，可用于约束检查与回归矩阵评分。
基频和谐波数决定带宽，系数控制幅度。闭环辨识仍使用实际测量数据。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FourierTrajectory:
    r"""每个关节一条有限项傅里叶级数::

        q_i(t)   = q0_i + sum_k [ a_ik/(w k) sin(w k t) - b_ik/(w k) cos(w k t) ]
        dq_i(t)  =        sum_k [ a_ik cos(w k t) + b_ik sin(w k t) ]
        ddq_i(t) =        sum_k [ -a_ik w k sin(w k t) + b_ik w k cos(w k t) ]

    ``w = 2 pi f_base``，周期为 ``1/f_base``。
    """

    q0: np.ndarray  # (dof,) 偏置
    a: np.ndarray  # (dof, n_harmonics) 余弦系数（作用在 dq 上）
    b: np.ndarray  # (dof, n_harmonics) 正弦系数
    base_freq: float = 0.1

    def __post_init__(self):
        self.q0 = np.asarray(self.q0, dtype=float).reshape(-1)
        self.a = np.atleast_2d(np.asarray(self.a, dtype=float))
        self.b = np.atleast_2d(np.asarray(self.b, dtype=float))
        if self.a.shape != self.b.shape:
            raise ValueError("a 与 b 形状必须一致")
        if self.a.shape[0] != self.q0.size:
            raise ValueError("系数行数必须等于自由度")

    @property
    def dof(self) -> int:
        return self.q0.size

    @property
    def n_harmonics(self) -> int:
        return self.a.shape[1]

    @property
    def period(self) -> float:
        return 1.0 / self.base_freq

    def evaluate(self, t):
        """返回 ``(q, dq, ddq)``；``t`` 为标量时形状 ``(dof,)``。"""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        w = 2.0 * np.pi * self.base_freq
        k = np.arange(1, self.n_harmonics + 1)
        wk = w * k  # (n_harmonics,)
        phase = np.einsum("t,k->tk", t, wk)
        sin, cos = np.sin(phase), np.cos(phase)

        q = self.q0 + (
            np.einsum("ik,tk->ti", self.a / wk, sin)
            - np.einsum("ik,tk->ti", self.b / wk, cos)
        )
        dq = np.einsum("ik,tk->ti", self.a, cos) + np.einsum(
            "ik,tk->ti", self.b, sin
        )
        ddq = -np.einsum("ik,tk->ti", self.a * wk, sin) + np.einsum(
            "ik,tk->ti", self.b * wk, cos
        )
        if t.size == 1:
            return q[0], dq[0], ddq[0]
        return q, dq, ddq

    def timestamps(self, *, periods: int = 1, rate: float = 100.0):
        """一个或多个完整周期上的采样时刻（不含右端点，避免重复采样）。"""
        n = int(round(self.period * periods * rate))
        return np.arange(n) / rate

    def peak_values(self, *, periods: int = 1, rate: float = 100.0):
        """``(|q|max, |dq|max, |ddq|max)`` 每关节，用于检查是否超限。"""
        q, dq, ddq = self.evaluate(self.timestamps(periods=periods, rate=rate))
        return (
            np.abs(q).max(axis=0),
            np.abs(dq).max(axis=0),
            np.abs(ddq).max(axis=0),
        )

    def scaled(self, factor) -> FourierTrajectory:
        """整体缩放幅值（偏置不动），用于压进关节限位。"""
        f = np.asarray(factor, dtype=float).reshape(-1, 1)
        return FourierTrajectory(
            q0=self.q0, a=self.a * f, b=self.b * f, base_freq=self.base_freq
        )

    def fit_limits(
        self,
        lower,
        upper,
        *,
        margin: float = 0.05,
        periods: int = 1,
        rate: float = 100.0,
    ) -> FourierTrajectory:
        """偏置取区间中点、幅值整体缩放，使 ``q(t)`` 落在限位内。

        真实机型的限位往往不对称（Franka 的 joint4 是 ``[-3.07, -0.07]``），
        零偏置的随机轨迹必然越界。缩放只动幅值不动频谱，激励特性仍在，
        但条件数会变——所以要在挑轨迹**之前**压限位，不是之后。
        """
        lo = np.asarray(lower, dtype=float).reshape(-1)
        hi = np.asarray(upper, dtype=float).reshape(-1)
        if lo.size != self.dof or hi.size != self.dof:
            raise ValueError("限位长度必须等于自由度")
        if np.any(hi <= lo):
            raise ValueError("上限必须大于下限")

        centered = FourierTrajectory(
            q0=np.zeros(self.dof), a=self.a, b=self.b, base_freq=self.base_freq
        )
        q, _, _ = centered.evaluate(
            centered.timestamps(periods=periods, rate=rate)
        )
        amp = np.abs(q).max(axis=0)
        room = 0.5 * (hi - lo) * (1.0 - margin)
        factor = np.where(amp > room, room / np.where(amp > 0, amp, 1.0), 1.0)
        scaled = self.scaled(factor)
        return FourierTrajectory(
            q0=0.5 * (lo + hi),
            a=scaled.a,
            b=scaled.b,
            base_freq=self.base_freq,
        )

    def constrained(self, *, lower=None, upper=None, velocity=None, acceleration=None, margin=0.10):
        """用各谐波幅值之和作保守上界，约束整个连续周期而非离散采样点。

        q0 保持不变；位置余量不足时缩小振幅。轨迹限幅不等价于实际
        机器人不超限（跟踪误差仍需由仿真诊断检查）。
        """
        from ..utils.validation import vector

        if not 0 <= margin < 1 or not np.isfinite(self.base_freq) or self.base_freq <= 0:
            raise ValueError("margin 必须在 [0,1)，base_freq 必须为正")
        if not np.isfinite(np.concatenate([self.q0, self.a.ravel(), self.b.ravel()])).all():
            raise ValueError("轨迹系数必须有限")
        wk = 2*np.pi*self.base_freq*np.arange(1, self.n_harmonics+1)
        amplitudes = np.hypot(self.a, self.b)
        factor = np.ones(self.dof)

        def bound(peak, room):
            nonlocal factor
            ratio = np.ones(self.dof)
            np.divide(room*(1-margin), peak, out=ratio, where=peak > 0)
            factor = np.minimum(factor, ratio)

        if (lower is None) != (upper is None):
            raise ValueError("lower/upper 必须同时给定")
        if lower is not None:
            lo, hi = vector(lower, self.dof, "lower"), vector(upper, self.dof, "upper")
            room = np.minimum(self.q0-lo, hi-self.q0)
            if np.any(hi <= lo) or np.any(room <= 0):
                raise ValueError("轨迹 q0 必须严格位于关节限位内")
            bound(np.sum(amplitudes/wk, axis=1), room)
        if velocity is not None:
            bound(np.sum(amplitudes, axis=1), vector(velocity, self.dof, "velocity", positive=True))
        if acceleration is not None:
            bound(np.sum(amplitudes*wk, axis=1), vector(acceleration, self.dof, "acceleration", positive=True))
        return self.scaled(factor)

    @classmethod
    def random(
        cls,
        dof: int,
        rng: np.random.Generator,
        *,
        n_harmonics: int = 5,
        base_freq: float = 0.1,
        vel_scale: float = 1.0,
        q0=None,
    ) -> FourierTrajectory:
        """随机系数。``vel_scale`` 直接控制速度量级（系数就作用在 dq 上）。

        高次谐波按 ``1/k`` 衰减，免得加速度被高频项主导。
        """
        k = np.arange(1, n_harmonics + 1)
        scale = vel_scale / (k * np.sqrt(n_harmonics))
        a = rng.normal(0.0, 1.0, size=(dof, n_harmonics)) * scale
        b = rng.normal(0.0, 1.0, size=(dof, n_harmonics)) * scale
        return cls(
            q0=np.zeros(dof) if q0 is None else np.asarray(q0, float),
            a=a,
            b=b,
            base_freq=base_freq,
        )
