"""Park 形式 RNE（旋量 / 伴随）：由 ``S_body`` 与相邻变换逆 ``T_rel_inv`` 递推。"""

from __future__ import annotations

from typing import Callable

from sympy import eye, zeros

from ...utils.lie import Adj, Adjdual, adj, adjdual
from ...utils.mathutil import identity, skew
from ...utils.symbols import RobotSymbols
from ..actuator import drive_inertia_term, friction_force


def rne_park_forward(symbols: RobotSymbols, frames, ifunc: Callable | None = None):
    if frames.convention not in ("standard", "modified", "poe"):
        raise NotImplementedError(
            f"Park RNE for convention={frames.convention!r} not implemented"
        )
    if frames.S_body is None:
        raise ValueError("Park RNE requires body screw axes frames.S_body")
    ifunc = ifunc or identity
    V = list(range(symbols.dof + 1))
    dV = list(range(symbols.dof + 1))
    V[-1] = zeros(6, 1)
    dV[-1] = -zeros(3, 1).col_join(symbols.gravityacc)

    S = frames.S_body
    Tinv = frames.T_rel_inv
    for i in range(symbols.dof):
        V[i] = ifunc(Adj(Tinv[i], V[i - 1]) + S[i] * symbols.dq[i])
        dV[i] = ifunc(
            S[i] * symbols.ddq[i]
            + Adj(Tinv[i], dV[i - 1])
            + adj(Adj(Tinv[i], V[i - 1]), S[i] * symbols.dq[i])
        )
    return V, dV


def rne_park_backward(symbols: RobotSymbols, frames, fw_results, ifunc=None):
    V, dV = fw_results
    ifunc = ifunc or identity
    Tinv = frames.T_rel_inv + [eye(4)]
    F = list(range(symbols.dof + 1))
    F[symbols.dof] = zeros(6, 1)
    tau = zeros(symbols.dof, 1)
    fric = friction_force(symbols)
    idrive = drive_inertia_term(symbols)
    S = frames.S_body

    for i in range(symbols.dof - 1, -1, -1):
        Llm = (symbols.L[i].row_join(skew(symbols.l[i]))).col_join(
            (-skew(symbols.l[i])).row_join(eye(3) * symbols.m[i])
        )
        F[i] = ifunc(
            Adjdual(Tinv[i + 1], F[i + 1])
            + Llm * dV[i]
            - adjdual(V[i], Llm * V[i])
        )
        tau[i] = ifunc((S[i].T * F[i])[0] + fric[i] + idrive[i])
    return tau
