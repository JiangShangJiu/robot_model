"""离线测量滤波与数值微分。

未直接测量加速度时，用位置或速度估计导数。微分会放大高频噪声，
因此先低通滤波，微分后再滤波。这里使用前后向滤波，避免额外相移；
该方法需要整段数据，不适用于在线反馈。
"""

from __future__ import annotations

import numpy as np


def _require_scipy():
    try:
        from scipy.signal import butter, filtfilt
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise ImportError(
            "滤波需要 scipy，请安装：pip install scipy"
        ) from exc
    return butter, filtfilt


def zero_phase_lowpass(
    x: np.ndarray, *, fs: float, cutoff: float, order: int = 4
) -> np.ndarray:
    """Butterworth 零相位低通，沿时间轴（第 0 维）滤波。

    ``cutoff`` 要明显高于激励带宽、明显低于噪声频段；截得太狠会削掉
    真实的高频动态，反而引入偏差。
    """
    butter, filtfilt = _require_scipy()
    nyq = 0.5 * fs
    if not 0 < cutoff < nyq:
        raise ValueError(f"cutoff 必须落在 (0, {nyq})，当前 {cutoff}")
    b, a = butter(order, cutoff / nyq)
    return filtfilt(b, a, np.asarray(x, dtype=float), axis=0)


def differentiate(x: np.ndarray, dt: float) -> np.ndarray:
    """沿时间轴中心差分（端点降为单边），形状不变。"""
    return np.gradient(np.asarray(x, dtype=float), dt, axis=0)


def estimate_derivatives(
    dq: np.ndarray,
    *,
    dt: float,
    cutoff: float | None = None,
    order: int = 4,
) -> np.ndarray:
    """由速度估计加速度：可选先滤波、微分、再滤一次。

    微分后再滤一次是因为差分本身会把残余噪声抬到高频。
    ``cutoff=None`` 时退化为裸微分，只适合仿真出来的干净数据。
    """
    dq = np.asarray(dq, dtype=float)
    if cutoff is None:
        return differentiate(dq, dt)
    fs = 1.0 / dt
    smooth = zero_phase_lowpass(dq, fs=fs, cutoff=cutoff, order=order)
    return zero_phase_lowpass(
        differentiate(smooth, dt), fs=fs, cutoff=cutoff, order=order
    )
