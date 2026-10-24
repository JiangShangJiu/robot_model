"""演示共享的产物目录、指标保存与绘图初始化。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def ensure_output_dir(output_dir) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_metrics(path, metrics: dict) -> None:
    """统一保存指标 JSON：UTF-8、缩进、禁止 NaN（强制显式处理缺失值）。"""
    Path(path).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def rounded(value, digits: int = 5):
    """递归四舍五入，metrics JSON 更易读；同时把 numpy 标量转为内建类型。"""
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [rounded(item, digits) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [rounded(item, digits) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def use_headless_matplotlib():
    """无显示环境出图；演示图采用英文标注，与 docs/figures 约定一致。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "figure.autolayout": False})
    return plt


def rms(values, axis=0):
    return np.sqrt(np.mean(np.square(np.asarray(values, float)), axis=axis))
