"""从辨识结果反解 M / c / g。

辨识给出基参数向量 ``pi_b``，通常不能唯一反解每根连杆的 ``m / l / Le``，
因为基参数是完整参数的线性组合。但**力矩映射**是完全确定的，而
``tau = M(q) ddq + c(q,dq) + g(q)`` 三项各自都能由 ``Hb`` 在特定状态上
取值得到：

    g(q)        = Hb(q, 0, 0) pi_b
    M(q) 第 i 列 = Hb(q, 0, e_i) pi_b - g(q)
    c(q, dq)    = Hb(q, dq, 0) pi_b - g(q)

本模块直接恢复辨识模型的数值 M/c/g，用于前馈补偿和重力补偿。
若需要逐连杆的质量、质心和惯性张量，见 ``identification.recovery``：
它通过参数搜索与物理检查恢复一组可行的完整参数，不保证解唯一。

注意摩擦：粘性项 ``fv*dq`` 与库仑项 ``fc*sign(dq)`` 会落在 ``c`` 里，
与刚体科氏项混在一起。若启用恒定偏置 ``fo``，它在静止时仍存在，因此落在
``g`` 里；此时 ``g`` 是静态补偿力矩，不能单独解释为纯重力。本模块不拆分这些项。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class RigidBodyTerms:
    """某个状态下的动力学分解结果。"""

    M: np.ndarray  # (dof, dof) 惯性矩阵
    c: np.ndarray  # (dof,) 科氏 / 离心项（含粘性和库仑摩擦）
    g: np.ndarray  # (dof,) 重力项（含恒定偏置，若模型启用 fo）

    @property
    def dof(self) -> int:
        return self.g.size

    def torque(self, ddq) -> np.ndarray:
        """重组逆动力学力矩，用于自检。"""
        return self.M @ np.asarray(ddq, dtype=float).reshape(-1) + self.c + self.g

    def symmetry_error(self) -> float:
        """``M`` 的非对称度；理论为零，可用来判断分解是否可信。"""
        return float(np.max(np.abs(self.M - self.M.T)))


class DynamicsDecomposer:
    """用辨识出的 ``pi_b`` 和基回归 ``Hb`` 复原 M / c / g。

    ``regressor_func`` 必须是基回归 ``Hb``（与辨识时同一个），
    ``parms`` 是对应的 ``pi_b``。
    """

    def __init__(self, regressor_func: Callable, parms, dof: int):
        self.regressor_func = regressor_func
        self.parms = np.asarray(parms, dtype=float).reshape(-1)
        self.dof = int(dof)
        self._zero = np.zeros(self.dof)

    def _tau(self, q, dq, ddq) -> np.ndarray:
        Hb = np.asarray(self.regressor_func(q, dq, ddq), dtype=float)
        if Hb.shape[1] != self.parms.size:
            raise ValueError(
                f"回归矩阵列数 {Hb.shape[1]} 与参数维数 {self.parms.size} 不一致"
            )
        return Hb @ self.parms

    def gravity(self, q) -> np.ndarray:
        """``g(q)``：静止且无加速度时的力矩，包括模型中的恒定偏置。"""
        return self._tau(q, self._zero, self._zero)

    def inertia(self, q, *, g=None) -> np.ndarray:
        """``M(q)``：逐列施加单位加速度并扣除重力。"""
        g = self.gravity(q) if g is None else g
        M = np.empty((self.dof, self.dof))
        for i in range(self.dof):
            e = np.zeros(self.dof)
            e[i] = 1.0
            M[:, i] = self._tau(q, self._zero, e) - g
        return M

    def coriolis(self, q, dq, *, g=None) -> np.ndarray:
        """``c(q,dq)``：零加速度下扣除重力。"""
        g = self.gravity(q) if g is None else g
        return self._tau(q, dq, self._zero) - g

    def at(self, q, dq) -> RigidBodyTerms:
        """一次给出 ``M / c / g``，共用同一份 ``g(q)``。"""
        g = self.gravity(q)
        return RigidBodyTerms(
            M=self.inertia(q, g=g), c=self.coriolis(q, dq, g=g), g=g
        )
