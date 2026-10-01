"""雅可比：平面 2R 的 DH/PoE 对拍与差分检验。"""

from __future__ import annotations

import numpy as np
import sympy

from robot_model import Robot


def _J_numpy(robot: Robot, q: np.ndarray, link: int = -1) -> np.ndarray:
    idx = robot.dof - 1 if link < 0 else link
    fun = sympy.lambdify(list(robot.q), robot.kinematics.J[idx], "numpy")
    return np.asarray(fun(*q), dtype=float)


def test_planar_poe_jacobian_matches_dh():
    dh = Robot.from_dh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    poe = dh.to_poe()
    q = np.array([0.4, -0.6])
    assert np.linalg.norm(_J_numpy(dh, q) - _J_numpy(poe, q)) < 1e-12


def test_jacobian_finite_difference_mdh():
    """末端线速度列：∂p/∂q_j ≈ Jp 列。"""
    mdh = Robot.from_mdh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    poe = mdh.to_poe()
    q = np.array([0.3, -0.5], dtype=float)
    eps = 1e-7
    J = _J_numpy(poe, q)
    p0 = poe.fk_numpy(q)[:3, 3]
    for j in range(2):
        dq = np.zeros(2)
        dq[j] = eps
        p1 = poe.fk_numpy(q + dq)[:3, 3]
        jp_num = (p1 - p0) / eps
        assert np.linalg.norm(jp_num - J[:3, j]) < 1e-5


if __name__ == "__main__":
    test_planar_poe_jacobian_matches_dh()
    test_jacobian_finite_difference_mdh()
    print("planar jacobian OK")
