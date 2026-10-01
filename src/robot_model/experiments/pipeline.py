"""跨机器人可复用的实验编排辅助函数。"""

from __future__ import annotations

from typing import Callable

import numpy as np

from robot_model.data import (
    MotionData,
    align_measurements,
    average_periods,
    drop_low_speed,
    filter_measurements,
)
from robot_model.excitation import (
    FourierTrajectory,
    optimize_excitation,
    regressor_quality,
)


def prepare_measurements(
    data: MotionData,
    *,
    encoder_delay_steps: int = 0,
    torque_delay_steps: int = 0,
    cutoff: float = 5.0,
    period: float | None = None,
    cycles: int | None = None,
    edge_seconds: float = 0.3,
    min_speed: float = 0.08,
) -> MotionData:
    """延迟补偿 → 可选周期平均 → 零相位滤波 → 去边缘 → 去低速。"""
    data = align_measurements(
        data,
        encoder_delay_steps=encoder_delay_steps,
        torque_delay_steps=torque_delay_steps,
    )
    if period is not None:
        data = average_periods(data, period, cycles=cycles).data
    data = filter_measurements(data, cutoff=cutoff)
    dt = float(np.mean(np.diff(data.t)))
    edge = int(np.ceil(edge_seconds / dt))
    if len(data) <= 2 * edge + 3:
        raise ValueError("数据不足以去掉滤波边缘，请增加轨迹周期")
    return drop_low_speed(data.subset(slice(edge, -edge)), min_speed)


def analytic_condition(
    regressor_func: Callable, traj: FourierTrajectory, samples: int = 96
) -> float:
    t = np.arange(samples) * traj.period / samples
    q, dq, ddq = traj.evaluate(t)
    return regressor_quality(
        regressor_func, MotionData(t, q, dq, ddq, np.zeros_like(q))
    )["condition_number"]


def refine_excitation_candidates(
    regressor_func: Callable,
    candidates: list[FourierTrajectory],
    *,
    limits,
    velocity_limits,
    acceleration_limits,
    starts: int = 2,
    maxiter: int = 20,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[FourierTrajectory], list[dict]]:
    """对解析条件数最好的若干候选做 SLSQP 精炼，优化结果追加进候选池。"""
    if limits is None:
        raise ValueError("约束优化需要 joint_lower/joint_upper")
    order = sorted(
        range(len(candidates)),
        key=lambda i: analytic_condition(regressor_func, candidates[i]),
    )[:starts]
    reports: list[dict] = []
    for i in order:
        if progress is not None:
            progress(f"约束优化候选 {i + 1} 的傅里叶系数…")
        optimized = optimize_excitation(
            regressor_func,
            candidates[i],
            limits=limits,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
            maxiter=maxiter,
        )
        record = {"initial_candidate_id": i, **optimized.report}
        if optimized.report["adopted"]:
            record["optimized_candidate_id"] = len(candidates)
            candidates.append(optimized.trajectory)
        reports.append(record)
        if progress is not None:
            before = record["before"]["condition_number"]
            after = record["after"]["condition_number"]
            progress(
                f"  验证网格 cond: {before:.3g} → {after:.3g}；"
                f"收敛={record['converged']}"
            )
    return candidates, reports


def torque_rms(predicted, actual) -> list[float]:
    return np.sqrt(np.mean((predicted - actual) ** 2, axis=0)).tolist()
