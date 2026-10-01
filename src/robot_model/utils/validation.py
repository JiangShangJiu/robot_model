"""数值配置校验：供数据、轨迹与仿真模块共用。"""

import numpy as np


def vector(value, dof, name, *, positive=False, nonnegative=False):
    a = np.asarray(value, dtype=float)
    if a.ndim > 1 or a.size not in (1, dof):
        raise ValueError(f"{name} 必须为标量或长度 {dof} 的向量")
    a = np.broadcast_to(a.reshape(-1), (dof,)).copy()
    if not np.isfinite(a).all():
        raise ValueError(f"{name} 必须有限")
    if (positive and np.any(a <= 0)) or (nonnegative and np.any(a < 0)):
        raise ValueError(f"{name} 超出有效范围")
    return a


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} 必须为 >= {minimum} 的整数")
    return int(value)
