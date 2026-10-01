"""解析雅可比（平面 2R）。"""

from __future__ import annotations

import numpy as np
import sympy

from robot_model import Robot
from robot_model.utils.mathutil import rpy_zyx_E


def _lamb(robot, expr, q):
    fun = sympy.lambdify(list(robot.q), expr, "numpy")
    return np.asarray(fun(*q), dtype=float)


def test_analytical_planar_direct_and_relation():
    robot = Robot.from_mdh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    kin = robot.kinematics
    q = np.array([0.35, -0.55])

    Ja = kin.ja_numpy(q)
    Ja_c = kin.ja_from_geometric_numpy(q)
    assert np.linalg.norm(Ja - Ja_c) < 1e-6

    Ja_sym = _lamb(robot, kin.gen_analytical_symbolic(), q)
    assert np.linalg.norm(Ja - Ja_sym) < 1e-5

    Jg = _lamb(robot, kin.J[-1], q)
    phi = _lamb(robot, kin.rpy[-1], q).reshape(3)
    r, p, y = sympy.symbols("roll pitch yaw", real=True)
    E = np.asarray(
        sympy.lambdify((r, p, y), rpy_zyx_E([r, p, y]), "numpy")(*phi),
        dtype=float,
    )
    dq = np.array([0.2, -0.1])
    assert np.linalg.norm(Jg[3:] @ dq - E @ (Ja[3:] @ dq)) < 1e-6


if __name__ == "__main__":
    test_analytical_planar_direct_and_relation()
    print("planar analytical jacobian OK")
