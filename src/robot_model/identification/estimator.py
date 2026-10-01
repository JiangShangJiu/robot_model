"""基参数的最小二乘估计与误差统计。

regressor_func 应使用 Dynamics.gen_base_regressor() 生成的 Hb。
完整回归矩阵 H 存在结构相关列，直接求解不能唯一确定完整参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..data.dataset import MotionData


def stack_regressor(
    regressor_func: Callable, data: MotionData
) -> tuple[np.ndarray, np.ndarray]:
    """把逐时刻的回归矩阵摞成 ``(n_samples*dof, n_parms)`` 与观测向量。"""
    n = data.n_samples
    if n < 1:
        raise ValueError("没有有效采样点")
    first = np.asarray(
        regressor_func(data.q[0], data.dq[0], data.ddq[0]), dtype=float
    )
    if first.ndim != 2:
        raise ValueError("回归矩阵须为二维")
    dof, n_parms = first.shape
    W = np.empty((n * dof, n_parms), dtype=float)
    W[:dof] = first
    for i in range(1, n):
        block = np.asarray(
            regressor_func(data.q[i], data.dq[i], data.ddq[i]), dtype=float
        )
        if block.shape != (dof, n_parms):
            raise ValueError(
                f"第 {i} 个采样点回归矩阵形状 {block.shape} 与首帧 {(dof, n_parms)} 不一致"
            )
        W[i * dof : (i + 1) * dof] = block
    y = data.tau.reshape(-1)
    if W.shape[0] != y.size:
        raise ValueError(
            f"回归矩阵行数 {W.shape[0]} 与观测数 {y.size} 不一致"
        )
    return W, y


@dataclass(frozen=True)
class IdentificationResult:
    """辨识结果与诊断量。"""

    parms: np.ndarray
    condition_number: float
    residual_rms: float
    relative_residual: float
    n_samples: int
    rank: int

    @property
    def n_parms(self) -> int:
        return self.parms.size

    def summary(self) -> str:
        return (
            f"{self.n_parms} 个参数 / {self.n_samples} 个采样点，"
            f"cond={self.condition_number:.3g}，"
            f"残差 RMS={self.residual_rms:.3g}"
            f"（相对 {self.relative_residual:.2%}）"
        )


def identify(
    regressor_func: Callable,
    data: MotionData,
    *,
    weights=None,
) -> IdentificationResult:
    """加权最小二乘求解 ``tau = Hb pi_b``。

    ``weights`` 给每个关节一个权重（长度 ``dof``），一般取测量噪声标准差
    的倒数；量纲差距大的关节不加权会被大力矩关节带偏。
    """
    W, y = stack_regressor(regressor_func, data)
    Wf, yf = W, y
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != data.dof:
            raise ValueError("weights 长度必须等于自由度")
        rw = np.tile(w, data.n_samples)
        Wf, yf = W * rw[:, None], y * rw

    parms, _, rank, sv = np.linalg.lstsq(Wf, yf, rcond=None)
    residual = W @ parms - y
    scale = np.linalg.norm(y)
    return IdentificationResult(
        parms=parms,
        condition_number=float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf,
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        relative_residual=(
            float(np.linalg.norm(residual) / scale) if scale > 0 else 0.0
        ),
        n_samples=data.n_samples,
        rank=int(rank),
    )


def predict_torque(
    regressor_func: Callable, data: MotionData, parms: np.ndarray
) -> np.ndarray:
    """用给定参数预测力矩，形状同 ``data.tau``。"""
    W, _ = stack_regressor(regressor_func, data)
    return (W @ np.asarray(parms, dtype=float)).reshape(data.tau.shape)


def parameter_error(estimated, reference) -> dict:
    """辨识值与真值的偏差。``rel`` 以真值向量的范数为基准。"""
    est = np.asarray(estimated, dtype=float).reshape(-1)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    if est.shape != ref.shape:
        raise ValueError("参数向量长度不一致")
    diff = est - ref
    ref_norm = np.linalg.norm(ref)
    return {
        "abs_max": float(np.max(np.abs(diff))),
        "rel_norm": float(np.linalg.norm(diff) / ref_norm)
        if ref_norm > 0
        else float("inf"),
        "per_parm": diff,
    }


__all__ = [
    "IdentificationResult",
    "identify",
    "parameter_error",
    "predict_torque",
    "stack_regressor",
]
