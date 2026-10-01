"""动力学回归矩阵 H：τ = H π。"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

from sympy import Matrix, zeros

from ..geometry.frames import FrameChain
from ..utils.mathutil import identity
from ..utils.symbols import RobotSymbols
from . import lagrange as lag_mod
from .methods import RneMethod, resolve_rne_method, rne_backward, rne_forward


def regressor(
    symbols: RobotSymbols,
    frames: FrameChain,
    ifunc: Callable | None = None,
    *,
    method: RneMethod | None = None,
    kinematics=None,
):
    """H(q,dq,ddq)，使 τ = H π（π 为 ``symbols.dynparms()``）。

    Park / Khalil / Lagrange 均线性于同一套 barycentric 参数，列构造方式相同。
    """
    ifunc = ifunc or identity
    m = resolve_rne_method(frames, method)
    parms = symbols.dynparms()
    Y = zeros(symbols.dof, len(parms))
    tmp = deepcopy(symbols)

    if m == "lagrange":
        for p, parm in enumerate(parms):
            for i in range(symbols.dof):
                tmp.Le[i] = [1 if x == parm else 0 for x in symbols.Le[i]]
                tmp.l[i] = Matrix(symbols.l[i]).applyfunc(
                    lambda x: 1 if x == parm else 0
                )
                tmp.m[i] = 1 if symbols.m[i] == parm else 0
                tmp.Ia[i] = 1 if symbols.Ia[i] == parm else 0
                tmp.fv[i] = 1 if symbols.fv[i] == parm else 0
                tmp.fc[i] = 1 if symbols.fc[i] == parm else 0
                tmp.fo[i] = 1 if symbols.fo[i] == parm else 0
            Y[:, p] = lag_mod.inverse_dynamics_lagrange(
                tmp, frames, ifunc, kinematics=kinematics
            )
        return Y

    fw = rne_forward(symbols, frames, ifunc, method=m)
    for p, parm in enumerate(parms):
        for i in range(symbols.dof):
            tmp.Le[i] = [1 if x == parm else 0 for x in symbols.Le[i]]
            tmp.l[i] = Matrix(symbols.l[i]).applyfunc(
                lambda x: 1 if x == parm else 0
            )
            tmp.m[i] = 1 if symbols.m[i] == parm else 0
            tmp.Ia[i] = 1 if symbols.Ia[i] == parm else 0
            tmp.fv[i] = 1 if symbols.fv[i] == parm else 0
            tmp.fc[i] = 1 if symbols.fc[i] == parm else 0
            tmp.fo[i] = 1 if symbols.fo[i] == parm else 0
        Y[:, p] = rne_backward(tmp, frames, fw, ifunc, method=m)
    return Y
