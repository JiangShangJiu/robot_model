"""PoE Park、拉格朗日与 RNE 对拍。"""

from __future__ import annotations

import numpy as np

from robot_model import Robot


def _fill(robot, q, dq, ddq):
    sym = robot.symbols
    subs = {}
    for i in range(robot.dof):
        subs[sym.q[i]] = float(q[i])
        subs[sym.dq[i]] = float(dq[i])
        subs[sym.ddq[i]] = float(ddq[i])
        subs[sym.m[i]] = 1.0 + 0.1 * i
        for j, v in enumerate([0.1, 0.0, 0.0, 0.1, 0.0, 0.1]):
            subs[sym.Le[i][j]] = float(v)
        for j, v in enumerate([0.01, 0.02, 0.03]):
            subs[sym.l[i][j]] = float(v)
        subs[sym.Ia[i]] = 0.0
        if sym.fv is not None:
            for attr in ("fv", "fc", "fo"):
                arr = getattr(sym, attr)
                if arr is not None:
                    subs[arr[i]] = 0.0
    g = sym.gravityacc
    subs[g[0]] = 0.0
    subs[g[1]] = 0.0
    subs[g[2]] = -9.81
    return subs


def _num(tau, subs):
    return np.array(tau.subs(subs).evalf(), dtype=float).reshape(-1)


def test_poe_park_matches_mdh():
    mdh = Robot.from_mdh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    poe = mdh.to_poe()
    q, dq, ddq = [0.3, -0.5], [0.1, 0.2], [0.05, -0.1]
    tau_m = mdh.dynamics.gen_invdyn(method="park")
    tau_p = poe.dynamics.gen_invdyn(method="park")
    assert np.linalg.norm(
        _num(tau_m, _fill(mdh, q, dq, ddq)) - _num(tau_p, _fill(poe, q, dq, ddq))
    ) < 1e-10


def test_lagrange_matches_rne_mdh():
    robot = Robot.from_mdh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    tau_r = robot.dynamics.gen_invdyn(method="khalil")
    tau_l = robot.dynamics.gen_invdyn(method="lagrange")
    q, dq, ddq = [0.35, -0.55], [0.1, -0.2], [0.05, 0.15]
    subs = _fill(robot, q, dq, ddq)
    assert np.linalg.norm(_num(tau_r, subs) - _num(tau_l, subs)) < 1e-8


def test_lagrange_matches_rne_dh():
    robot = Robot.from_dh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    tau_r = robot.dynamics.gen_invdyn(method="park")
    tau_l = robot.dynamics.gen_invdyn(method="lagrange")
    q, dq, ddq = [0.35, -0.55], [0.1, -0.2], [0.05, 0.15]
    subs = _fill(robot, q, dq, ddq)
    assert np.linalg.norm(_num(tau_r, subs) - _num(tau_l, subs)) < 1e-8


def test_lagrange_poe_matches_park():
    mdh = Robot.from_mdh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    poe = mdh.to_poe()
    tau_p = poe.dynamics.gen_invdyn(method="park")
    tau_l = poe.dynamics.gen_invdyn(method="lagrange")
    q, dq, ddq = [0.2, 0.4], [-0.1, 0.3], [0.05, -0.08]
    subs = _fill(poe, q, dq, ddq)
    assert np.linalg.norm(_num(tau_p, subs) - _num(tau_l, subs)) < 1e-8


if __name__ == "__main__":
    test_poe_park_matches_mdh()
    test_lagrange_matches_rne_mdh()
    test_lagrange_matches_rne_dh()
    test_lagrange_poe_matches_park()
    print("lagrange / poe dynamics OK")
