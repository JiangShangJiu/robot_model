"""雅可比：Franka MDH/DH/PoE 末端一致。"""

from __future__ import annotations

import numpy as np
import sympy

from robot_model import Robot
from robot_model.robots.franka import PANDA_MDH_PARMS


def _J_numpy(robot: Robot, q: np.ndarray, link: int = -1) -> np.ndarray:
    idx = robot.dof - 1 if link < 0 else link
    fun = sympy.lambdify(list(robot.q), robot.kinematics.J[idx], "numpy")
    return np.asarray(fun(*q), dtype=float)


def test_mdh_dh_poe_ee_jacobian_match():
    mdh = Robot.from_mdh("panda", PANDA_MDH_PARMS)
    dh = mdh.to_dh()
    poe = mdh.to_poe()
    q = np.array([0.1, -0.5, 0.2, -1.0, 0.3, 0.8, -0.4])

    Jm = _J_numpy(mdh, q)
    Jd = _J_numpy(dh, q)
    Jp = _J_numpy(poe, q)

    assert Jm.shape == (6, 7)
    assert np.linalg.norm(Jm - Jd) < 1e-9
    assert np.linalg.norm(Jm - Jp) < 1e-9


if __name__ == "__main__":
    test_mdh_dh_poe_ee_jacobian_match()
    print("franka jacobian OK")
