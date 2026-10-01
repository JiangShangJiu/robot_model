"""由逆动力学派生的 M / c / C / g 与斜对称检验。"""

from __future__ import annotations

from typing import Callable

from sympy import Matrix, Rational, zeros

from ..geometry.frames import FrameChain
from ..utils.mathutil import identity
from ..utils.symbols import RobotSymbols
from . import lagrange as lag_mod
from .methods import (
    RneMethod,
    inverse_dynamics,
    resolve_rne_method,
    rne_backward,
    rne_forward,
)
from .actuator import friction_force


def gravity_term(
    symbols: RobotSymbols,
    frames: FrameChain,
    ifunc: Callable | None = None,
    *,
    method: RneMethod | None = None,
):
    ifunc = ifunc or identity
    # 几何只依赖 (q, description)，置零 dq/ddq 不影响 frames，直接复用
    tmp = symbols.copy()
    tmp.dq = zeros(tmp.dof, 1)
    tmp.ddq = zeros(tmp.dof, 1)
    tmp.frictionmodel = None
    m = resolve_rne_method(frames, method)
    if m == "lagrange":
        return lag_mod.gravity_term_lagrange(tmp, frames, ifunc)
    return inverse_dynamics(tmp, frames, ifunc, method=m)


def coriolis_term(
    symbols: RobotSymbols,
    frames: FrameChain,
    ifunc: Callable | None = None,
    *,
    method: RneMethod | None = None,
):
    ifunc = ifunc or identity
    tmp = symbols.copy()
    tmp.gravityacc = zeros(3, 1)
    tmp.ddq = zeros(tmp.dof, 1)
    tmp.frictionmodel = None
    m = resolve_rne_method(frames, method)
    if m == "lagrange":
        return lag_mod.coriolis_term_lagrange(tmp, frames, ifunc=ifunc)
    return inverse_dynamics(tmp, frames, ifunc, method=m)


def friction_term(symbols: RobotSymbols, ifunc=None):
    return friction_force(symbols, ifunc)


def coriolis_matrix_from_M(
    M: Matrix,
    q,
    dq,
    ifunc: Callable | None = None,
):
    """由 ``M(q)`` 的 Christoffel 符号构造 ``C(q,dq)``，使 ``c=C\\dot q``，
    且 ``\\dot M-2C`` 斜对称。"""
    ifunc = ifunc or identity
    n = M.shape[0]
    C = zeros(n)
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                c_ijk = Rational(1, 2) * (
                    M[i, j].diff(q[k])
                    + M[i, k].diff(q[j])
                    - M[j, k].diff(q[i])
                )
                s += c_ijk * dq[k]
            C[i, j] = ifunc(s)
    return C


def coriolis_matrix(
    symbols: RobotSymbols,
    frames: FrameChain,
    ifunc: Callable | None = None,
    *,
    method: RneMethod | None = None,
    M: Matrix | None = None,
):
    """科氏矩阵 ``C``（Christoffel）：``c=C\\dot q``，``\\dot M-2C`` 斜对称。"""
    ifunc = ifunc or identity
    if M is None:
        M = inertia_matrix(symbols, frames, ifunc, method=method)
    return coriolis_matrix_from_M(M, symbols.q, symbols.dq, ifunc)


def mdot_minus_2C(
    M: Matrix,
    C: Matrix,
    q,
    dq,
    ifunc: Callable | None = None,
):
    """``S=\\dot M-2C``（应斜对称：``S+S^T=0``）。"""
    ifunc = ifunc or identity
    n = M.shape[0]
    Mdot = zeros(n)
    for i in range(n):
        for j in range(n):
            Mdot[i, j] = sum(M[i, j].diff(q[k]) * dq[k] for k in range(n))
    return ifunc(Mdot - 2 * C)


def skew_symmetry_residual(
    M: Matrix,
    C: Matrix,
    q,
    dq,
    ifunc: Callable | None = None,
):
    """``(\\dot M-2C)+(\\dot M-2C)^T``，理想为零。"""
    S = mdot_minus_2C(M, C, q, dq, ifunc)
    return S + S.T


def inertia_matrix(
    symbols: RobotSymbols,
    frames: FrameChain,
    ifunc: Callable | None = None,
    *,
    method: RneMethod | None = None,
):
    ifunc = ifunc or identity
    m = resolve_rne_method(frames, method)
    if m == "lagrange":
        return lag_mod.inertia_matrix_lagrange(symbols, frames, ifunc)

    # 逐列施加单位 ddq；几何与 ddq 无关，整列循环共用同一份 frames
    M = zeros(symbols.dof, symbols.dof)
    tmp = symbols.copy()
    tmp.gravityacc = zeros(3, 1)
    tmp.frictionmodel = None
    tmp.dq = zeros(tmp.dof, 1)
    for i in range(symbols.dof):
        tmp.ddq = zeros(tmp.dof, 1)
        tmp.ddq[i] = 1
        coli = rne_backward(
            tmp,
            frames,
            rne_forward(tmp, frames, ifunc, method=m),
            ifunc,
            method=m,
        )
        M[:, i] = (M[i, :i].T).col_join(coli[i:, :])
    return M
