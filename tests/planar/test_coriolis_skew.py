"""M、C 与斜对称关系 ``Ṁ-2C``。"""

from __future__ import annotations

import numpy as np
import sympy

from robot_model import Robot
from robot_model.dynamics import skew_symmetry_residual


def _fill_dyn(robot, seed=0):
    rng = np.random.default_rng(seed)
    sym = robot.symbols
    subs = {}
    for i in range(robot.dof):
        subs[sym.q[i]] = float(rng.uniform(-1, 1))
        subs[sym.dq[i]] = float(rng.uniform(-1, 1))
        subs[sym.m[i]] = float(0.8 + 0.2 * i)
        for j, v in enumerate([0.1, 0.0, 0.0, 0.08, 0.0, 0.06]):
            subs[sym.Le[i][j]] = float(v)
        for j, v in enumerate([0.01, -0.02, 0.03]):
            subs[sym.l[i][j]] = float(v)
        subs[sym.Ia[i]] = 0.0
    return subs


def test_coriolis_matrix_and_skew_symmetry():
    robot = Robot.from_mdh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    dyn = robot.dynamics
    M = dyn.gen_inertiamatrix(method="khalil")
    C = dyn.gen_coriolismatrix(method="khalil")
    c = dyn.gen_coriolisterm(method="khalil")

    # c = C dq
    assert sympy.simplify(c - C * robot.symbols.dq) == sympy.zeros(2, 1)

    # Ṁ-2C 斜对称
    R = skew_symmetry_residual(M, C, robot.symbols.q, robot.symbols.dq)
    assert sympy.simplify(R) == sympy.zeros(2)

    S = dyn.gen_skew_symmetry(method="khalil")
    # xᵀ S x = 0
    x = sympy.Matrix(sympy.symbols("x1 x2", real=True))
    assert sympy.simplify((x.T * S * x)[0]) == 0

    # 数值：τ 的科氏部分
    subs = _fill_dyn(robot)
    for i in range(2):
        subs[robot.symbols.ddq[i]] = 0.0
    g = robot.symbols.gravityacc
    subs[g[0]] = 0
    subs[g[1]] = 0
    subs[g[2]] = 0
    cn = np.array(c.subs(subs).evalf(), float).reshape(-1)
    Cn = np.array(C.subs(subs).evalf(), float)
    dqn = np.array([float(subs[robot.symbols.dq[i]]) for i in range(2)])
    assert np.linalg.norm(cn - Cn @ dqn) < 1e-10


def test_skew_symmetry_park_lagrange():
    robot = Robot.from_dh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    for method in ("park", "lagrange"):
        dyn = robot.dynamics
        M = dyn.gen_inertiamatrix(method=method)
        C = dyn.gen_coriolismatrix(method=method)
        R = skew_symmetry_residual(M, C, robot.symbols.q, robot.symbols.dq)
        assert sympy.simplify(R) == sympy.zeros(2)


if __name__ == "__main__":
    test_coriolis_matrix_and_skew_symmetry()
    test_skew_symmetry_park_lagrange()
    print("coriolis / skew-symmetry OK")
