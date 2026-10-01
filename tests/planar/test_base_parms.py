"""回归矩阵与最小参数集。"""

from __future__ import annotations

import numpy as np
import sympy

from robot_model import Robot


def _planar():
    return Robot.from_mdh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )


def test_regressor_park_khalil_lagrange_match():
    robot = _planar()
    dyn = robot.dynamics
    Hk = dyn.gen_regressor(method="khalil")
    Hp = dyn.gen_regressor(method="park")
    Hl = dyn.gen_regressor(method="lagrange")
    # 数值抽查若干配置
    qfun_k = sympy.lambdify(
        list(robot.q) + list(robot.symbols.dq) + list(robot.symbols.ddq),
        Hk,
        "numpy",
    )
    qfun_p = sympy.lambdify(
        list(robot.q) + list(robot.symbols.dq) + list(robot.symbols.ddq),
        Hp,
        "numpy",
    )
    qfun_l = sympy.lambdify(
        list(robot.q) + list(robot.symbols.dq) + list(robot.symbols.ddq),
        Hl,
        "numpy",
    )
    args = [0.3, -0.5, 0.1, 0.2, 0.05, -0.1]
    Ak = np.asarray(qfun_k(*args), float)
    Ap = np.asarray(qfun_p(*args), float)
    Al = np.asarray(qfun_l(*args), float)
    assert np.linalg.norm(Ak - Ap) < 1e-8
    assert np.linalg.norm(Ak - Al) < 1e-6


def test_base_parms_and_Hb():
    robot = _planar()
    dyn = robot.dynamics
    dyn.gen_regressor(method="khalil")
    base = dyn.calc_base_parms(samples=800, seed=0)
    assert dyn.n_base < dyn.n_dynparms
    assert len(base) == dyn.n_base
    Hb = dyn.gen_base_regressor()
    assert Hb.shape == (robot.dof, dyn.n_base)

    # τ = H π = Hb π_b
    H_fun = sympy.lambdify(
        list(robot.q) + list(robot.symbols.dq) + list(robot.symbols.ddq),
        dyn.H,
        "numpy",
    )
    Hb_fun = sympy.lambdify(
        list(robot.q) + list(robot.symbols.dq) + list(robot.symbols.ddq),
        Hb,
        "numpy",
    )
    args = [0.2, 0.4, -0.1, 0.3, 0.05, 0.1]
    Hn = np.asarray(H_fun(*args), float)
    Hbn = np.asarray(Hb_fun(*args), float)

    rng = np.random.default_rng(1)
    pi = rng.normal(size=dyn.n_dynparms)
    # π_b = (Pb.T + Kd Pd.T) π
    Pb = np.asarray(dyn.Pb, float)
    Pd = np.asarray(dyn.Pd, float)
    Kd = np.asarray(dyn.Kd, float)
    pi_b = (Pb.T + Kd @ Pd.T) @ pi
    assert np.linalg.norm(Hn @ pi - Hbn @ pi_b) < 1e-6


def test_poe_regressor_and_base():
    mdh = _planar()
    poe = mdh.to_poe()
    H = poe.dynamics.gen_regressor(method="park")
    assert H.shape[0] == 2
    poe.dynamics.calc_base_parms(samples=500, seed=2)
    assert poe.dynamics.n_base >= 1


if __name__ == "__main__":
    test_regressor_park_khalil_lagrange_match()
    test_base_parms_and_Hb()
    test_poe_regressor_and_base()
    print("regressor / base parms OK")
