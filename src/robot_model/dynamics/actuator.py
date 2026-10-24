"""驱动器侧附加项：关节摩擦与驱动（转子）惯量。"""

from __future__ import annotations

from typing import Callable

from sympy import exp, sign, zeros

from ..utils.mathutil import identity
from ..utils.symbols import RobotSymbols

_FRICTION_TERMS = {"Coulomb", "viscous", "Stribeck", "offset"}


def friction_force(symbols: RobotSymbols, ifunc: Callable | None = None):
    """关节摩擦力矩。

    ``Stribeck`` 项为高斯型低速摩擦峰 ``fs * exp(-(dq/v_s)^2) * sign(dq)``，
    与 Coulomb+viscous 合成经典 Stribeck 曲线：``fc + fs`` 是起动（breakaway）
    摩擦幅值，``fc`` 是滑动库仑摩擦，``v_s`` 取 ``symbols.stribeck_velocity``
    （固定模型常数）。``fs`` 对力矩线性，可进入回归矩阵辨识。
    """
    ifunc = ifunc or identity
    fric = zeros(symbols.dof, 1)
    if not symbols.frictionmodel:
        return fric

    asked = set(symbols.frictionmodel)
    unknown = asked - _FRICTION_TERMS
    if unknown:
        raise ValueError(
            f"unknown friction terms {unknown}; use subset of {_FRICTION_TERMS}"
        )

    for i in range(symbols.dof):
        if "viscous" in asked:
            fric[i] += symbols.fv[i] * symbols.dq[i]
        if "Coulomb" in asked:
            fric[i] += symbols.fc[i] * sign(symbols.dq[i])
        if "Stribeck" in asked:
            scaled = symbols.dq[i] / symbols.stribeck_velocity[i]
            fric[i] += symbols.fs[i] * exp(-(scaled**2)) * sign(symbols.dq[i])
        if "offset" in asked:
            fric[i] += symbols.fo[i]
        fric[i] = ifunc(fric[i])
    return fric


def drive_inertia_term(symbols: RobotSymbols, ifunc: Callable | None = None):
    ifunc = ifunc or identity
    out = zeros(symbols.dof, 1)
    if symbols.driveinertiamodel is None:
        return out
    if symbols.driveinertiamodel != "simplified":
        raise ValueError(
            f"unknown driveinertiamodel {symbols.driveinertiamodel!r}; "
            "use None or 'simplified'"
        )
    for i in range(symbols.dof):
        out[i] = ifunc(symbols.Ia[i] * symbols.ddq[i])
    return out
