"""Franka 7 轴：数值回归矩阵与最小参数集。"""

from __future__ import annotations

import numpy as np

from robot_model import Robot
from robot_model.robots.franka import PANDA_MDH_PARMS


def test_franka_regressor_numpy_and_base_parms():
    mdh = Robot.from_mdh("franka_panda_arm", PANDA_MDH_PARMS)
    dyn = mdh.dynamics
    assert dyn.n_dynparms == 70  # 7 * (6+3+1)

    q = np.array([0.2, -0.4, 0.3, -1.0, 0.5, 0.7, -0.3])
    dq = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.2, 0.0])
    ddq = np.array([0.05, 0.1, -0.05, 0.2, 0.0, -0.1, 0.15])

    H = dyn.regressor_numpy(q, dq, ddq, method="park")
    assert H.shape == (7, 70)

    Hk = dyn.regressor_numpy(q, dq, ddq, method="khalil")
    assert np.linalg.norm(H - Hk) < 1e-8

    # τ = H π
    rng = np.random.default_rng(0)
    pi = rng.normal(size=70)
    nd = dyn.numeric_dynamics(method="park")
    m = np.zeros(7)
    l = np.zeros((7, 3))
    Le = np.zeros((7, 6))
    for c, (kind, i, j) in enumerate(nd._parm_slots):
        if kind == "m":
            m[i] = pi[c]
        elif kind == "l":
            l[i, j] = pi[c]
        elif kind == "Le":
            Le[i, j] = pi[c]
    tau = nd.invdyn(q, dq, ddq, m, l, Le)
    assert np.linalg.norm(H @ pi - tau) < 1e-8

    base = dyn.calc_base_parms(method="park", samples=150, seed=0, numeric=True)
    assert dyn.n_base < dyn.n_dynparms
    assert dyn.n_base > 0
    assert len(base) == dyn.n_base

    dyn.gen_base_regressor(method="park")
    Pb = np.asarray(dyn.Pb, float)
    Pd = np.asarray(dyn.Pd, float)
    Kd = np.asarray(dyn.Kd, float)
    pi_b = (Pb.T + Kd @ Pd.T) @ pi
    Hb = dyn.Hb_func(q, dq, ddq)
    assert Hb.shape == (7, dyn.n_base)
    assert np.linalg.norm(H @ pi - Hb @ pi_b) < 1e-5


def test_franka_poe_numeric_regressor_base():
    poe = Robot.from_mdh("franka_panda_arm", PANDA_MDH_PARMS).to_poe()
    dyn = poe.dynamics
    q = np.zeros(7)
    dq = np.ones(7) * 0.1
    ddq = np.ones(7) * 0.05
    H = dyn.regressor_numpy(q, dq, ddq, method="park")
    assert H.shape == (7, dyn.n_dynparms)
    dyn.calc_base_parms(method="park", samples=120, seed=1, numeric=True)
    assert dyn.n_base < dyn.n_dynparms


if __name__ == "__main__":
    test_franka_regressor_numpy_and_base_parms()
    test_franka_poe_numeric_regressor_base()
    print("franka regressor / base parms OK")
