"""根据解析回归矩阵条件数搜索激励轨迹。"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..data.dataset import MotionData
from .trajectory import FourierTrajectory


def search_excitation(
    regressor_func: Callable,
    dof: int,
    rng: np.random.Generator,
    *,
    trials: int = 30,
    n_harmonics: int = 5,
    base_freq: float = 0.1,
    vel_scale: float = 1.0,
    q0=None,
    limits=None,
    velocity_limits=None,
    acceleration_limits=None,
    periods: int = 1,
    rate: float = 50.0,
    refine: bool = True,
    refine_maxiter: int = 20,
) -> tuple[FourierTrajectory, float]:
    """随机搜条件数最小的激励轨迹，可选局部约束优化精炼。

    返回 ``(轨迹, 条件数)``。条件数直接决定噪声被放大多少倍。
    只接受列满秩的回归矩阵；欠定或列相关的候选不能用于唯一辨识。

    ``limits`` 给 ``(lower, upper)`` 时，每条候选先经
    :meth:`FourierTrajectory.fit_limits` 压进限位再算条件数——否则返回的
    条件数属于另一条（越界的）轨迹。此时偏置由限位中点决定，``q0`` 失效。
    ``velocity_limits/acceleration_limits`` 给定时，再用谐波幅值上界约束
    整个周期的位置/速度/加速度，并对约束后的轨迹计算条件数。

    ``refine=True``（默认）且三类限位齐全、SciPy 可用时，对最优随机候选再做
    SLSQP 约束优化（最小化 ``log cond``）；否则保持随机搜索结果。
    """
    from .screening import regressor_quality

    best, best_cond = None, np.inf
    for _ in range(trials):
        traj = FourierTrajectory.random(
            dof,
            rng,
            n_harmonics=n_harmonics,
            base_freq=base_freq,
            vel_scale=vel_scale,
            q0=q0,
        )
        if limits is not None:
            traj = traj.fit_limits(*limits, periods=periods, rate=rate)
        if velocity_limits is not None or acceleration_limits is not None:
            traj = traj.constrained(
                lower=None if limits is None else limits[0],
                upper=None if limits is None else limits[1],
                velocity=velocity_limits,
                acceleration=acceleration_limits,
            )
        t = traj.timestamps(periods=periods, rate=rate)
        if t.size < 2:
            raise ValueError(
                "激励轨迹采样点不足，请提高采样率 rate 或增加记录周期 periods"
            )
        q, dq, ddq = traj.evaluate(t)
        preview = MotionData(t, q, dq, ddq, np.zeros_like(q))
        cond = regressor_quality(regressor_func, preview)["condition_number"]
        if cond < best_cond:
            best, best_cond = traj, cond
    if best is None:
        raise ValueError(
            "未找到列满秩且条件数有限的激励轨迹；请增加采样率 rate / 记录周期 periods，"
            "增强激励，或检查基参数回归矩阵的列相关性及 trials"
        )

    can_refine = (
        refine
        and limits is not None
        and velocity_limits is not None
        and acceleration_limits is not None
    )
    if can_refine:
        try:
            from .optimization import optimize_excitation
        except ImportError:
            optimize_excitation = None
        if optimize_excitation is not None:
            refined = optimize_excitation(
                regressor_func,
                best,
                limits=limits,
                velocity_limits=velocity_limits,
                acceleration_limits=acceleration_limits,
                maxiter=refine_maxiter,
            )
            if refined.report["adopted"]:
                best = refined.trajectory
                t = best.timestamps(periods=periods, rate=rate)
                q, dq, ddq = best.evaluate(t)
                preview = MotionData(t, q, dq, ddq, np.zeros_like(q))
                best_cond = regressor_quality(regressor_func, preview)[
                    "condition_number"
                ]

    return best, best_cond
