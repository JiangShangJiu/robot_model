"""拉格朗日逆动力学：由动能 / 势能生成 ``M,g,c,τ``（DH / MDH / PoE 通用）。

惯性参数与 RNE 一致：连杆原点处 barycentric ``(L, l, m)``，
动能 ``T=½ ωᵀ(R L Rᵀ)ω + ωᵀ((R l)×v_o) + ½ m ‖v_o‖²``（对参数线性）。
"""

from __future__ import annotations

from typing import Callable

from sympy import Matrix, Rational, zeros

from ..geometry.dh import DHParams
from ..geometry.frames import FrameChain
from ..geometry.poe import PoEParams
from ..utils.mathutil import identity, skew
from ..utils.symbols import RobotSymbols
from ..kinematics import Kinematics
from .actuator import drive_inertia_term, friction_force


def _ensure_kinematics(
    symbols: RobotSymbols,
    frames: FrameChain,
    kinematics: Kinematics | None,
) -> Kinematics:
    if kinematics is not None:
        return kinematics
    return Kinematics(symbols, frames, analytical=False)


def inertia_matrix_lagrange(
    symbols: RobotSymbols,
    frames: FrameChain,
    ifunc: Callable | None = None,
    *,
    kinematics: Kinematics | None = None,
):
    """``M(q)``，使 ``T = ½ \\dot qᵀ M \\dot q``（对 ``L,l,m`` 线性）。"""
    ifunc = ifunc or identity
    kin = _ensure_kinematics(symbols, frames, kinematics)
    n = symbols.dof
    M = zeros(n)
    for i in range(n):
        Jp = kin.Jp[i]
        Jo = kin.Jo[i]
        R = frames.R[i]
        L_base = R * symbols.L[i] * R.T
        l_base = R * Matrix(symbols.l[i])
        # T = 1/2 ωᵀ L_b ω + ωᵀ (l_b × v) + 1/2 m vᵀ v
        #   = 1/2 dqᵀ (Joᵀ L_b Jo + m Jpᵀ Jp + Joᵀ skew(l_b) Jp + Jpᵀ skew(l_b)ᵀ Jo) dq
        # ωᵀ (l × v) = ωᵀ skew(l) v → Joᵀ skew(l_b) Jp 的对称化
        sk = skew(l_base)
        M += (
            Jo.T * L_base * Jo
            + symbols.m[i] * (Jp.T * Jp)
            + Jo.T * sk * Jp
            + Jp.T * sk.T * Jo
        )
    return ifunc(M)


def gravity_term_lagrange(
    symbols: RobotSymbols,
    frames: FrameChain,
    ifunc: Callable | None = None,
    *,
    kinematics: Kinematics | None = None,
):
    """``g=∂V/∂q``，``V=-Σ (m g·p + g·(R l))``。"""
    ifunc = ifunc or identity
    _ = kinematics
    V = 0
    gvec = symbols.gravityacc
    for i in range(symbols.dof):
        p = frames.p[i]
        R = frames.R[i]
        V += -symbols.m[i] * (gvec.T * p)[0] - (
            gvec.T * (R * Matrix(symbols.l[i]))
        )[0]
    q = Matrix(symbols.q)
    return ifunc(Matrix([V]).jacobian(q).T)


def coriolis_term_lagrange(
    symbols: RobotSymbols,
    frames: FrameChain,
    M: Matrix | None = None,
    ifunc: Callable | None = None,
    *,
    kinematics: Kinematics | None = None,
):
    """``c(q,dq)=Ṁ dq - ½ ∂(dqᵀ M dq)/∂q``。"""
    ifunc = ifunc or identity
    if M is None:
        M = inertia_matrix_lagrange(
            symbols, frames, ifunc, kinematics=kinematics
        )
    n = symbols.dof
    q = symbols.q
    dq = symbols.dq
    c = zeros(n, 1)
    dM_dq = [M.diff(q[k]) for k in range(n)]
    for i in range(n):
        mdot_dq_i = 0
        for j in range(n):
            Mdot_ij = sum(dM_dq[k][i, j] * dq[k] for k in range(n))
            mdot_dq_i += Mdot_ij * dq[j]
        quad = sum(
            dq[a] * M[a, b] * dq[b] for a in range(n) for b in range(n)
        )
        c[i] = ifunc(mdot_dq_i - Rational(1, 2) * quad.diff(q[i]))
    return c


def inverse_dynamics_lagrange(
    symbols: RobotSymbols,
    frames: FrameChain,
    ifunc: Callable | None = None,
    *,
    kinematics: Kinematics | None = None,
):
    """``τ = M ddq + c(q,dq) + g(q) + 摩擦 + 驱动惯量``。"""
    ifunc = ifunc or identity
    kin = _ensure_kinematics(symbols, frames, kinematics)
    M = inertia_matrix_lagrange(symbols, frames, ifunc, kinematics=kin)
    c = coriolis_term_lagrange(
        symbols, frames, M=M, ifunc=ifunc, kinematics=kin
    )
    g = gravity_term_lagrange(symbols, frames, ifunc, kinematics=kin)
    fric = friction_force(symbols, ifunc)
    idrive = drive_inertia_term(symbols, ifunc)
    return ifunc(M * symbols.ddq + c + g + fric + idrive)


def gravity_term_lagrange_from_description(
    symbols: RobotSymbols,
    description: DHParams | PoEParams,
    ifunc: Callable | None = None,
):
    frames = FrameChain(symbols, description)
    return gravity_term_lagrange(symbols, frames, ifunc)


def coriolis_term_lagrange_from_description(
    symbols: RobotSymbols,
    description: DHParams | PoEParams,
    ifunc: Callable | None = None,
):
    tmp = symbols.copy()
    tmp.gravityacc = zeros(3, 1)
    tmp.ddq = zeros(tmp.dof, 1)
    tmp.frictionmodel = None
    frames = FrameChain(tmp, description)
    M = inertia_matrix_lagrange(tmp, frames, ifunc)
    return coriolis_term_lagrange(tmp, frames, M=M, ifunc=ifunc)


def inertia_matrix_lagrange_from_description(
    symbols: RobotSymbols,
    description: DHParams | PoEParams,
    ifunc: Callable | None = None,
):
    frames = FrameChain(symbols, description)
    return inertia_matrix_lagrange(symbols, frames, ifunc)
