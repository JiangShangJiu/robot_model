"""激励轨迹筛选：解析预筛 → 闭环试跑 → 有效测量回归矩阵评分。

只比较同一机器人、同一基参数定义下的候选。不使用力矩拟合残差或
真实参数误差挑选，避免先看答案再选训练数据。仿真诊断用于可执行性检查。
"""
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import numpy as np

from ..data.dataset import MotionData
from ..identification.estimator import stack_regressor
from ..utils.validation import integer, vector
from .trajectory import FourierTrajectory


@dataclass(frozen=True)
class ExcitationCriteria:
    """候选必须全部满足；阈值是实验策略，不代表硬件安全认证。

    min_singular_value 针对 W/sqrt(采样点数)，避免仅靠增加采样点获胜。
    不按每条轨迹单独做列归一化，否则会掩盖某些参数几乎未被激励。
    """
    min_samples: int = 200
    min_retained_fraction: float = 0.25
    max_condition_number: float = 1e4
    min_singular_value: float = 0.01
    max_tracking_rms_rad: float = 0.05
    max_tracking_error_rad: float = 0.15
    max_saturation_fraction: float = 0.01
    max_slew_limit_fraction: float = 0.05
    objective: str = "condition_number"

    def __post_init__(self):
        integer(self.min_samples, "min_samples", 1)
        for key in ("min_retained_fraction", "max_saturation_fraction", "max_slew_limit_fraction"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} 必须在 [0,1]")
        for key in ("max_condition_number", "max_tracking_rms_rad", "max_tracking_error_rad"):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} 必须为有限正数")
        if not np.isfinite(self.min_singular_value) or self.min_singular_value < 0:
            raise ValueError("min_singular_value 必须为有限非负数")
        if self.objective not in ("condition_number", "min_singular_value"):
            raise ValueError("objective 必须是 condition_number 或 min_singular_value")


def trajectory_record(traj):
    return {"q0": traj.q0.tolist(), "a": traj.a.tolist(), "b": traj.b.tolist(),
            "base_freq": float(traj.base_freq)}


def regressor_quality(regressor_func, data):
    """秩不足时条件数为 inf，包括行数少于参数数目的情形。"""
    if len(data) < 1:
        raise ValueError("没有有效采样点")
    W, _ = stack_regressor(regressor_func, data)
    if W.ndim != 2 or W.shape[1] == 0 or not np.isfinite(W).all():
        raise ValueError("回归矩阵为空或含非有限值")
    singular = np.linalg.svd(W, compute_uv=False)
    tolerance = np.finfo(float).eps * max(W.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    full_rank = rank == W.shape[1]
    return {
        "samples": len(data), "n_parms": W.shape[1], "rank": rank,
        "condition_number": float(singular[0]/singular[-1]) if full_rank else float("inf"),
        "min_singular_value": float(singular[-1]/np.sqrt(len(data))) if full_rank else 0.0,
    }


def generate_excitation_candidates(dof, rng, *, trials=12, frequencies=(0.10, 0.15, 0.20),
                                    harmonics=(3, 4, 5), vel_scale=1.0, q0=None,
                                    limits=None, velocity_limits=None, acceleration_limits=None):
    """生成不同频率、谐波数和幅值的可重放傅里叶候选，连续时间保守限幅。"""
    integer(trials, "trials", 1)
    integer(dof, "dof", 1)
    frequencies = np.asarray(frequencies, float)
    if frequencies.ndim != 1 or frequencies.size == 0 or not np.isfinite(frequencies).all() or np.any(frequencies <= 0):
        raise ValueError("frequencies 必须是非空的正数序列")
    harmonics = tuple(integer(x, "harmonics", 1) for x in harmonics)
    if not harmonics or not np.isfinite(vel_scale) or vel_scale <= 0:
        raise ValueError("harmonics 不能为空，vel_scale 必须为正")
    if q0 is None:
        q0 = (np.zeros(dof) if limits is None else
              (vector(limits[0], dof, "lower") + vector(limits[1], dof, "upper"))/2)
    candidates = []
    for _ in range(trials):
        traj = FourierTrajectory.random(
            dof, rng, n_harmonics=int(rng.choice(harmonics)),
            base_freq=float(rng.choice(frequencies)),
            vel_scale=vel_scale*float(rng.uniform(.65, 1.0)), q0=q0,
        ).constrained(
            lower=None if limits is None else limits[0],
            upper=None if limits is None else limits[1],
            velocity=velocity_limits, acceleration=acceleration_limits,
        )
        candidates.append(traj)
    return candidates


class NoFeasibleExcitation(ValueError):
    """全部候选未通过筛选；report 保存各候选的淘汰原因。"""
    def __init__(self, report):
        self.report = report
        super().__init__("没有合格激励轨迹；请检查筛选报告，调整候选范围或控制器后重试")


@dataclass(frozen=True)
class ExcitationSelection:
    trajectory: FourierTrajectory
    run: object
    data: MotionData
    report: dict


def _json_finite(value):
    """严格 JSON：秩不足产生的 inf 写为 null，另由 rank/reasons 解释。"""
    if isinstance(value, dict):
        return {k: _json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def screen_excitation(regressor_func: Callable, candidates: Iterable[FourierTrajectory],
                      runner: Callable, preprocess: Callable, *, criteria=None,
                      preview_rate=50.0, progress: Callable | None = None,
                      retention_multiplier=1):
    """runner(traj) 返回 SimulationRun；preprocess 只接收 measured 数据。

    对可行候选按实测回归矩阵条件数最小或最小奇异值最大排序。返回获胜
    试跑的数据，不再次采集一个不同噪声实现冒充已筛选数据。候选依次运行，
    runner 应重置状态、使用固定种子与相同的预热/记录周期，以公平比较。
    周期平均后可用 retention_multiplier 折算保留比例；有效样本数仍按
    实际相位点数计算，不能用重复次数虚增样本数或矩阵秩。
    """
    criteria = criteria or ExcitationCriteria()
    integer(retention_multiplier, "retention_multiplier", 1)
    if not np.isfinite(preview_rate) or preview_rate <= 0:
        raise ValueError("preview_rate 必须为有限正数")
    records, winner = [], None
    winner_score = float("inf")
    for index, traj in enumerate(candidates):
        rec = {"candidate_id": index, "trajectory": trajectory_record(traj),
               "accepted": False, "reasons": [], "preview": None, "measured": None,
               "diagnostics": None}
        records.append(rec)
        run, data = None, None
        try:
            if preview_rate <= 2*traj.base_freq*traj.n_harmonics:
                raise ValueError("预筛采样率不足以覆盖最高谐波频率")
            t = traj.timestamps(rate=preview_rate)
            if len(t) < 2:
                raise ValueError("预筛采样点不足")
            q, dq, ddq = traj.evaluate(t)
            preview = MotionData(t, q, dq, ddq, np.zeros_like(q))
            rec["preview"] = regressor_quality(regressor_func, preview)
            if rec["preview"]["rank"] < rec["preview"]["n_parms"]:
                rec["reasons"].append("解析轨迹的回归矩阵秩不足")
            if not rec["reasons"]:
                run = runner(traj)
                d = run.diagnostics
                rec["diagnostics"] = d
                # fail closed：非有限诊断量不能通过阈值比较蒙混过关。
                metrics = np.concatenate([np.asarray(d[key], float).reshape(-1) for key in (
                    "tracking_rms_rad", "tracking_max_rad", "saturation_fraction",
                    "slew_limit_fraction", "joint_limit_fraction", "contact_fraction",
                    "max_speed_rad_s", "max_acceleration_rad_s2")])
                if not np.isfinite(metrics).all():
                    raise ValueError("闭环诊断量含非有限值")
                for key, threshold, reason in (
                    ("tracking_rms_rad", criteria.max_tracking_rms_rad, "跟踪 RMS 超限"),
                    ("tracking_max_rad", criteria.max_tracking_error_rad, "最大跟踪误差超限"),
                    ("saturation_fraction", criteria.max_saturation_fraction, "力矩饱和率超限"),
                    ("slew_limit_fraction", criteria.max_slew_limit_fraction, "力矩变化率受限比例超限"),
                    ("joint_limit_fraction", 0, "触发关节限位"),
                    ("contact_fraction", 0, "出现接触"),
                ):
                    if np.max(d[key]) > threshold:
                        rec["reasons"].append(reason)
                if any(d["speed_limit_exceeded"]):
                    rec["reasons"].append("实际速度超限")
                if any(d["acceleration_limit_exceeded"]):
                    rec["reasons"].append("实际加速度超限")
                if not rec["reasons"]:
                    data = preprocess(run.measured)
                    if len(data) == 0 or not all(np.isfinite(getattr(data, key)).all() for key in ("t", "q", "dq", "ddq", "tau")):
                        raise ValueError("预处理数据为空或含非有限值")
                    # 周期平均压缩行数但不等于剔除数据；样本门槛仍按独立相位点计。
                    retention = min(1.0, len(data)*retention_multiplier/len(run.measured))
                    rec["retained_fraction"] = retention
                    quality = rec["measured"] = regressor_quality(regressor_func, data)
                    if len(data) < criteria.min_samples:
                        rec["reasons"].append("有效采样点不足")
                    if retention < criteria.min_retained_fraction:
                        rec["reasons"].append("有效样本保留比例过低")
                    if quality["rank"] < quality["n_parms"]:
                        rec["reasons"].append("实测回归矩阵秩不足")
                    if quality["condition_number"] > criteria.max_condition_number:
                        rec["reasons"].append("实测回归矩阵条件数过大")
                    if quality["min_singular_value"] < criteria.min_singular_value:
                        rec["reasons"].append("实测激励强度不足")
            if not rec["reasons"]:
                rec["accepted"] = True
                score = (rec["measured"]["condition_number"] if criteria.objective == "condition_number"
                         else -rec["measured"]["min_singular_value"])
                if score < winner_score:
                    winner_score = score
                    winner = (index, traj, run, data)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            rec["reasons"].append(f"候选失败：{type(exc).__name__}: {exc}")
        if progress is not None:
            progress(_json_finite(rec))
    report = _json_finite({
        "criteria": asdict(criteria), "preview_rate_hz": preview_rate,
        "retention_multiplier": retention_multiplier,
        "selected_candidate_id": None if winner is None else winner[0],
        "candidate_count": len(records),
        "accepted_count": sum(r["accepted"] for r in records),
        "candidates": records,
    })
    if winner is None:
        raise NoFeasibleExcitation(report)
    return ExcitationSelection(winner[1], winner[2], winner[3], report)
