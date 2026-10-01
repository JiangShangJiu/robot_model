"""Khalil 形式 RNE（三维向量递推）：支持标准 DH 与修正 MDH（直接按建系，不互转）。"""

from __future__ import annotations

from typing import Callable

from sympy import Matrix, eye, zeros

from ...utils.mathutil import identity, skew
from ...utils.symbols import RobotSymbols
from ..actuator import drive_inertia_term, friction_force


def rne_khalil_forward(symbols: RobotSymbols, frames, ifunc: Callable | None = None):
    if frames.convention == "modified":
        return _forward_modified(symbols, frames, ifunc)
    if frames.convention == "standard":
        return _forward_standard(symbols, frames, ifunc)
    raise NotImplementedError(
        f"Khalil RNE for convention={frames.convention!r} not implemented"
    )


def rne_khalil_backward(symbols: RobotSymbols, frames, fw_results, ifunc=None):
    if frames.convention == "modified":
        return _backward_modified(symbols, frames, fw_results, ifunc)
    if frames.convention == "standard":
        return _backward_standard(symbols, frames, fw_results, ifunc)
    raise NotImplementedError(
        f"Khalil RNE for convention={frames.convention!r} not implemented"
    )


def _forward_modified(symbols, frames, ifunc=None):
    """MDH：关节轴为当前系 ``z``（过原点）。"""
    ifunc = ifunc or identity
    w = list(range(symbols.dof + 1))
    dw = list(range(symbols.dof + 1))
    dV = list(range(symbols.dof + 1))
    U = list(range(symbols.dof + 1))

    w[-1] = zeros(3, 1)
    dw[-1] = zeros(3, 1)
    dV[-1] = -symbols.gravityacc
    U[-1] = zeros(3, 3)
    z = Matrix([0, 0, 1])

    for i in range(symbols.dof):
        s = frames.sigma[i]
        ns = 1 - s
        w_pj = frames.R_rel[i].T * w[i - 1]
        w[i] = ifunc(w_pj + ns * symbols.dq[i] * z)
        dw[i] = ifunc(
            frames.R_rel[i].T * dw[i - 1]
            + ns
            * (symbols.ddq[i] * z + w_pj.cross(symbols.dq[i] * z).reshape(3, 1))
        )
        dV[i] = ifunc(
            frames.R_rel[i].T * (dV[i - 1] + U[i - 1] * frames.p_rel[i])
            + s
            * (
                symbols.ddq[i] * z
                + 2 * w_pj.cross(symbols.dq[i] * z).reshape(3, 1)
            )
        )
        U[i] = ifunc(skew(dw[i]) + skew(w[i]) ** 2)

    return w, dw, dV, U


def _backward_modified(symbols, frames, fw_results, ifunc=None):
    w, dw, dV, U = fw_results
    ifunc = ifunc or identity
    Rdh = frames.R_rel + [eye(3)]
    pdh = frames.p_rel + [zeros(3, 1)]

    F = list(range(symbols.dof))
    M = list(range(symbols.dof))
    f = list(range(symbols.dof + 1))
    m = list(range(symbols.dof + 1))
    f[symbols.dof] = zeros(3, 1)
    m[symbols.dof] = zeros(3, 1)
    z = Matrix([0, 0, 1])
    tau = zeros(symbols.dof, 1)
    fric = friction_force(symbols)
    idrive = drive_inertia_term(symbols)

    for i in range(symbols.dof - 1, -1, -1):
        s = frames.sigma[i]
        ns = 1 - s
        F[i] = ifunc(symbols.m[i] * dV[i] + U[i] * Matrix(symbols.l[i]))
        M[i] = ifunc(
            symbols.L[i] * dw[i]
            + w[i].cross(symbols.L[i] * w[i]).reshape(3, 1)
            + Matrix(symbols.l[i]).cross(dV[i]).reshape(3, 1)
        )
        f_nj = Rdh[i + 1] * f[i + 1]
        f[i] = ifunc(F[i] + f_nj)
        m[i] = ifunc(
            M[i] + Rdh[i + 1] * m[i + 1] + pdh[i + 1].cross(f_nj).reshape(3, 1)
        )
        tau[i] = ifunc(((s * f[i] + ns * m[i]).T * z)[0] + fric[i] + idrive[i])

    return tau


def _forward_standard(symbols, frames, ifunc=None):
    """标准 DH：关节轴为上一系 ``z``；线加速度用关节计入后的 ``ω`` 搬运 ``p``。"""
    ifunc = ifunc or identity
    w = list(range(symbols.dof + 1))
    dw = list(range(symbols.dof + 1))
    dV = list(range(symbols.dof + 1))
    U = list(range(symbols.dof + 1))

    w[-1] = zeros(3, 1)
    dw[-1] = zeros(3, 1)
    dV[-1] = -symbols.gravityacc
    U[-1] = zeros(3, 3)
    z = Matrix([0, 0, 1])

    for i in range(symbols.dof):
        s = frames.sigma[i]
        ns = 1 - s
        R = frames.R_rel[i]
        w_l = w[i - 1] + ns * symbols.dq[i] * z
        dw_l = dw[i - 1] + ns * (
            symbols.ddq[i] * z
            + w[i - 1].cross(symbols.dq[i] * z).reshape(3, 1)
        )
        U_l = skew(dw_l) + skew(w_l) ** 2
        w[i] = ifunc(R.T * w_l)
        dw[i] = ifunc(R.T * dw_l)
        z_b = R.T * z
        dV[i] = ifunc(
            R.T * (dV[i - 1] + U_l * frames.p_rel[i])
            + s
            * (
                symbols.ddq[i] * z_b
                + 2 * (R.T * w_l).cross(symbols.dq[i] * z_b).reshape(3, 1)
            )
        )
        U[i] = ifunc(skew(dw[i]) + skew(w[i]) ** 2)

    return w, dw, dV, U


def _backward_standard(symbols, frames, fw_results, ifunc=None):
    """力矩投影到上一系关节轴 ``z``。"""
    w, dw, dV, U = fw_results
    ifunc = ifunc or identity
    Rdh = frames.R_rel + [eye(3)]
    pdh = frames.p_rel + [zeros(3, 1)]

    F = list(range(symbols.dof))
    M = list(range(symbols.dof))
    f = list(range(symbols.dof + 1))
    m = list(range(symbols.dof + 1))
    f[symbols.dof] = zeros(3, 1)
    m[symbols.dof] = zeros(3, 1)
    z = Matrix([0, 0, 1])
    tau = zeros(symbols.dof, 1)
    fric = friction_force(symbols)
    idrive = drive_inertia_term(symbols)

    for i in range(symbols.dof - 1, -1, -1):
        s = frames.sigma[i]
        ns = 1 - s
        F[i] = ifunc(symbols.m[i] * dV[i] + U[i] * Matrix(symbols.l[i]))
        M[i] = ifunc(
            symbols.L[i] * dw[i]
            + w[i].cross(symbols.L[i] * w[i]).reshape(3, 1)
            + Matrix(symbols.l[i]).cross(dV[i]).reshape(3, 1)
        )
        f_nj = Rdh[i + 1] * f[i + 1]
        f[i] = ifunc(F[i] + f_nj)
        m[i] = ifunc(
            M[i] + Rdh[i + 1] * m[i + 1] + pdh[i + 1].cross(f_nj).reshape(3, 1)
        )
        f_im1 = frames.R_rel[i] * f[i]
        m_im1 = frames.R_rel[i] * m[i] + frames.p_rel[i].cross(f_im1).reshape(3, 1)
        tau[i] = ifunc(
            ((s * f_im1 + ns * m_im1).T * z)[0] + fric[i] + idrive[i]
        )

    return tau
