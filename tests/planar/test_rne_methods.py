"""DH/MDH 上 Park 与 Khalil 直接对照（不经约定互转）。"""

from __future__ import annotations

import numpy as np
import sympy

from robot_model import Robot


def _subs_state(robot, q, dq, ddq, seed: int = 0):
    rng = np.random.default_rng(seed)
    sym = robot.symbols
    subs = {}
    for i in range(robot.dof):
        subs[sym.q[i]] = float(q[i])
        subs[sym.dq[i]] = float(dq[i])
        subs[sym.ddq[i]] = float(ddq[i])
        subs[sym.m[i]] = float(rng.uniform(0.8, 1.5))
        for j, v in enumerate([0.12, 0.01, -0.02, 0.11, 0.015, 0.09]):
            subs[sym.Le[i][j]] = float(v * (1 + 0.1 * i))
        for j, v in enumerate([0.02, -0.01, 0.03]):
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


def _tau_num(tau, subs):
    return np.array(tau.subs(subs).evalf(), dtype=float).reshape(-1)


def test_mdh_park_matches_khalil():
    robot = Robot.from_mdh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    assert robot.frames.S_body is not None
    tau_k = robot.dynamics.gen_invdyn(method="khalil")
    tau_p = robot.dynamics.gen_invdyn(method="park")
    assert sympy.simplify(tau_k - tau_p) == sympy.zeros(2, 1)

    subs = _subs_state(
        robot,
        [0.35, -0.55],
        [0.1, -0.2],
        [0.05, 0.15],
    )
    assert np.linalg.norm(_tau_num(tau_k, subs) - _tau_num(tau_p, subs)) < 1e-12


def test_dh_park_matches_khalil():
    robot = Robot.from_dh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    tau_p = robot.dynamics.gen_invdyn(method="park")
    tau_k = robot.dynamics.gen_invdyn(method="khalil")
    assert sympy.simplify(tau_p - tau_k) == sympy.zeros(2, 1)

    subs = _subs_state(
        robot,
        [0.35, -0.55],
        [0.1, -0.2],
        [0.05, 0.15],
        seed=1,
    )
    assert np.linalg.norm(_tau_num(tau_p, subs) - _tau_num(tau_k, subs)) < 1e-12


def test_default_methods_unchanged():
    mdh = Robot.from_mdh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    dh = Robot.from_dh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    assert sympy.simplify(mdh.dynamics.gen_invdyn() - mdh.dynamics.gen_invdyn(method="khalil")) == sympy.zeros(2, 1)
    assert sympy.simplify(dh.dynamics.gen_invdyn() - dh.dynamics.gen_invdyn(method="park")) == sympy.zeros(2, 1)


if __name__ == "__main__":
    test_mdh_park_matches_khalil()
    test_dh_park_matches_khalil()
    test_default_methods_unchanged()
    print("rne methods OK")
