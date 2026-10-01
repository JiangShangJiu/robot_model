"""Franka/Panda：数值 RNE / 拉格朗日对拍（避免 7 轴全符号展开）。"""

from __future__ import annotations

import numpy as np
import sympy

from robot_model import Robot
from robot_model.utils.lie import Adj_np as _Adj
from robot_model.utils.lie import Adjdual_np as _Adjdual
from robot_model.utils.lie import adj_np as _adj
from robot_model.utils.lie import adjdual_np as _adjdual
from robot_model.utils.lie import skew_np as _skew
from robot_model.robots.franka import PANDA_MDH_PARMS


def _tensor_L(Le6: np.ndarray) -> np.ndarray:
    xx, xy, xz, yy, yz, zz = Le6
    return np.array(
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float
    )


def _dyn_params(dof: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.5, 2.0, size=dof)
    l = rng.uniform(-0.05, 0.05, size=(dof, 3))
    Le = np.zeros((dof, 6))
    for i in range(dof):
        Le[i] = [
            0.04 + 0.01 * i,
            0.002,
            -0.001,
            0.03 + 0.01 * i,
            0.001,
            0.025 + 0.01 * i,
        ]
    return m, l, Le


def park_numpy(robot: Robot, q, dq, ddq, m, l, Le, g=None):
    """数值 Park RNE（与符号实现同结构）。"""
    g = np.array([0.0, 0.0, -9.81]) if g is None else np.asarray(g, float)
    n = robot.dof
    qsyms = list(robot.q)
    Tinv_fun = [
        sympy.lambdify(qsyms, robot.frames.T_rel_inv[i], "numpy") for i in range(n)
    ]
    S_fun = [
        sympy.lambdify(qsyms, robot.frames.S_body[i], "numpy") for i in range(n)
    ]
    qq = [float(x) for x in q]
    Tinv = [np.asarray(f(*qq), float).reshape(4, 4) for f in Tinv_fun]
    S = [np.asarray(f(*qq), float).reshape(6) for f in S_fun]

    V = [None] * (n + 1)
    dV = [None] * (n + 1)
    V[-1] = np.zeros(6)
    dV[-1] = np.hstack([np.zeros(3), -g])
    for i in range(n):
        V[i] = _Adj(Tinv[i], V[i - 1]) + S[i] * dq[i]
        dV[i] = (
            S[i] * ddq[i]
            + _Adj(Tinv[i], dV[i - 1])
            + _adj(_Adj(Tinv[i], V[i - 1]), S[i] * dq[i])
        )

    F = [None] * (n + 1)
    F[n] = np.zeros(6)
    Tinv_ext = Tinv + [np.eye(4)]
    tau = np.zeros(n)
    for i in range(n - 1, -1, -1):
        L = _tensor_L(Le[i])
        sk_l = _skew(l[i])
        Llm = np.block([[L, sk_l], [-sk_l, m[i] * np.eye(3)]])
        F[i] = (
            _Adjdual(Tinv_ext[i + 1], F[i + 1])
            + Llm @ dV[i]
            - _adjdual(V[i], Llm @ V[i])
        )
        tau[i] = float(S[i] @ F[i])
    return tau


def khalil_numpy(robot: Robot, q, dq, ddq, m, l, Le, g=None):
    """数值 Khalil RNE（按当前 convention 走 modified / standard）。"""
    g = np.array([0.0, 0.0, -9.81]) if g is None else np.asarray(g, float)
    n = robot.dof
    conv = robot.frames.convention
    qsyms = list(robot.q)
    R_fun = [
        sympy.lambdify(qsyms, robot.frames.R_rel[i], "numpy") for i in range(n)
    ]
    p_fun = [
        sympy.lambdify(qsyms, robot.frames.p_rel[i], "numpy") for i in range(n)
    ]
    qq = [float(x) for x in q]
    Rdh = [np.asarray(f(*qq), float).reshape(3, 3) for f in R_fun]
    pdh = [np.asarray(f(*qq), float).reshape(3) for f in p_fun]
    sigma = list(robot.frames.sigma)
    z = np.array([0.0, 0.0, 1.0])

    w = [None] * (n + 1)
    dw = [None] * (n + 1)
    dV = [None] * (n + 1)
    U = [None] * (n + 1)
    w[-1] = np.zeros(3)
    dw[-1] = np.zeros(3)
    dV[-1] = -g
    U[-1] = np.zeros((3, 3))

    for i in range(n):
        s = sigma[i]
        ns = 1 - s
        if conv == "modified":
            w_pj = Rdh[i].T @ w[i - 1]
            w[i] = w_pj + ns * dq[i] * z
            dw[i] = Rdh[i].T @ dw[i - 1] + ns * (
                ddq[i] * z + np.cross(w_pj, dq[i] * z)
            )
            dV[i] = Rdh[i].T @ (dV[i - 1] + U[i - 1] @ pdh[i]) + s * (
                ddq[i] * z + 2 * np.cross(w_pj, dq[i] * z)
            )
        else:  # standard
            w_l = w[i - 1] + ns * dq[i] * z
            dw_l = dw[i - 1] + ns * (
                ddq[i] * z + np.cross(w[i - 1], dq[i] * z)
            )
            U_l = _skew(dw_l) + _skew(w_l) @ _skew(w_l)
            w[i] = Rdh[i].T @ w_l
            dw[i] = Rdh[i].T @ dw_l
            z_b = Rdh[i].T @ z
            dV[i] = Rdh[i].T @ (dV[i - 1] + U_l @ pdh[i]) + s * (
                ddq[i] * z_b + 2 * np.cross(Rdh[i].T @ w_l, dq[i] * z_b)
            )
        U[i] = _skew(dw[i]) + _skew(w[i]) @ _skew(w[i])

    Rdh_e = Rdh + [np.eye(3)]
    pdh_e = pdh + [np.zeros(3)]
    f = [None] * (n + 1)
    mm = [None] * (n + 1)
    f[n] = np.zeros(3)
    mm[n] = np.zeros(3)
    tau = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = sigma[i]
        ns = 1 - s
        L = _tensor_L(Le[i])
        Fi = m[i] * dV[i] + U[i] @ l[i]
        Mi = L @ dw[i] + np.cross(w[i], L @ w[i]) + np.cross(l[i], dV[i])
        f_nj = Rdh_e[i + 1] @ f[i + 1]
        f[i] = Fi + f_nj
        mm[i] = Mi + Rdh_e[i + 1] @ mm[i + 1] + np.cross(pdh_e[i + 1], f_nj)
        if conv == "modified":
            tau[i] = float((s * f[i] + ns * mm[i]) @ z)
        else:
            fim1 = Rdh[i] @ f[i]
            mim1 = Rdh[i] @ mm[i] + np.cross(pdh[i], fim1)
            tau[i] = float((s * fim1 + ns * mim1) @ z)
    return tau


def lagrange_numpy(robot: Robot, q, dq, ddq, m, l, Le, g=None, eps=1e-7):
    """数值拉格朗日：M 由雅可比组装（barycentric 线性形式），c/g 数值微分。"""
    g = np.array([0.0, 0.0, -9.81]) if g is None else np.asarray(g, float)
    n = robot.dof
    qsyms = list(robot.q)
    kin = robot.kinematics

    Jp_f = [sympy.lambdify(qsyms, kin.Jp[i], "numpy") for i in range(n)]
    Jo_f = [sympy.lambdify(qsyms, kin.Jo[i], "numpy") for i in range(n)]
    R_f = [sympy.lambdify(qsyms, robot.frames.R[i], "numpy") for i in range(n)]
    p_f = [sympy.lambdify(qsyms, robot.frames.p[i], "numpy") for i in range(n)]

    def M_at(qv):
        qq = [float(x) for x in qv]
        M = np.zeros((n, n))
        for i in range(n):
            Jp = np.asarray(Jp_f[i](*qq), float).reshape(3, n)
            Jo = np.asarray(Jo_f[i](*qq), float).reshape(3, n)
            R = np.asarray(R_f[i](*qq), float).reshape(3, 3)
            L_base = R @ _tensor_L(Le[i]) @ R.T
            l_base = R @ l[i]
            sk = _skew(l_base)
            M += (
                Jo.T @ L_base @ Jo
                + m[i] * (Jp.T @ Jp)
                + Jo.T @ sk @ Jp
                + Jp.T @ sk.T @ Jo
            )
        return M

    def V_at(qv):
        qq = [float(x) for x in qv]
        V = 0.0
        for i in range(n):
            R = np.asarray(R_f[i](*qq), float).reshape(3, 3)
            p = np.asarray(p_f[i](*qq), float).reshape(3)
            V += -m[i] * float(g @ p) - float(g @ (R @ l[i]))
        return V

    q = np.asarray(q, float)
    dq = np.asarray(dq, float)
    ddq = np.asarray(ddq, float)
    M = M_at(q)
    grav = np.zeros(n)
    for k in range(n):
        qp, qm = q.copy(), q.copy()
        qp[k] += eps
        qm[k] -= eps
        grav[k] = (V_at(qp) - V_at(qm)) / (2 * eps)

    Mdot = np.zeros((n, n))
    for k in range(n):
        qp, qm = q.copy(), q.copy()
        qp[k] += eps
        qm[k] -= eps
        Mdot += ((M_at(qp) - M_at(qm)) / (2 * eps)) * dq[k]

    quad_grad = np.zeros(n)
    for k in range(n):
        qp, qm = q.copy(), q.copy()
        qp[k] += eps
        qm[k] -= eps
        qp_val = float(dq @ M_at(qp) @ dq)
        qm_val = float(dq @ M_at(qm) @ dq)
        quad_grad[k] = (qp_val - qm_val) / (2 * eps)

    c = Mdot @ dq - 0.5 * quad_grad
    return M @ ddq + c + grav


def test_franka_mdh_park_khalil_lagrange():
    mdh = Robot.from_mdh("franka_panda_arm", PANDA_MDH_PARMS)
    q = np.array([0.2, -0.4, 0.3, -1.0, 0.5, 0.7, -0.3])
    dq = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.2, 0.0])
    ddq = np.array([0.05, 0.1, -0.05, 0.2, 0.0, -0.1, 0.15])
    m, l, Le = _dyn_params(7, seed=0)

    tp = park_numpy(mdh, q, dq, ddq, m, l, Le)
    tk = khalil_numpy(mdh, q, dq, ddq, m, l, Le)
    tl = lagrange_numpy(mdh, q, dq, ddq, m, l, Le)
    assert np.linalg.norm(tp - tk) < 1e-8
    assert np.linalg.norm(tp - tl) < 5e-5


def test_franka_poe_park_lagrange_vs_mdh():
    mdh = Robot.from_mdh("franka_panda_arm", PANDA_MDH_PARMS)
    poe = mdh.to_poe()
    q = np.array([0.1, -0.5, 0.2, -1.0, 0.3, 0.8, -0.4])
    dq = np.array([-0.2, 0.1, 0.0, 0.25, -0.15, 0.05, 0.1])
    ddq = np.array([0.1, -0.05, 0.2, -0.1, 0.05, 0.0, -0.08])
    m, l, Le = _dyn_params(7, seed=1)

    t_mdh = park_numpy(mdh, q, dq, ddq, m, l, Le)
    t_poe = park_numpy(poe, q, dq, ddq, m, l, Le)
    t_lag = lagrange_numpy(poe, q, dq, ddq, m, l, Le)
    assert np.linalg.norm(t_mdh - t_poe) < 1e-8
    assert np.linalg.norm(t_poe - t_lag) < 5e-5


def test_franka_dh_park_khalil():
    dh = Robot.from_mdh("franka_panda_arm", PANDA_MDH_PARMS).to_dh()
    q = np.array([0.2, -0.4, 0.3, -1.0, 0.5, 0.7, -0.3])
    dq = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.2, 0.0])
    ddq = np.array([0.05, 0.1, -0.05, 0.2, 0.0, -0.1, 0.15])
    m, l, Le = _dyn_params(7, seed=2)
    tp = park_numpy(dh, q, dq, ddq, m, l, Le)
    tk = khalil_numpy(dh, q, dq, ddq, m, l, Le)
    assert np.linalg.norm(tp - tk) < 1e-8


if __name__ == "__main__":
    test_franka_mdh_park_khalil_lagrange()
    test_franka_poe_park_lagrange_vs_mdh()
    test_franka_dh_park_khalil()
    print("franka dynamics numeric OK")
