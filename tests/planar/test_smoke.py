"""自包含冒烟测试。"""

from __future__ import annotations

import sympy

from robot_model import Robot


def test_mdh_fk_and_invdyn():
    robot = Robot.from_mdh(
        "planar2r",
        [
            ("0", "1", "0", "q"),
            ("0", "1", "0", "q"),
        ],
    )
    assert robot.dof == 2
    assert robot.convention == "modified"
    assert robot.description.convention == "modified"

    T = robot.forward_kinematics()
    assert T.shape == (4, 4)

    T0 = T.subs({robot.q[0]: 0, robot.q[1]: 0})
    assert T0[0, 3] == 2
    assert T0[1, 3] == 0

    J = robot.kinematics.J[-1]
    assert J.shape == (6, 2)

    tau = robot.dynamics.gen_invdyn()
    assert tau.shape == (2, 1)
    assert all(isinstance(x, sympy.Expr) for x in tau)

    H = robot.dynamics.gen_regressor()
    assert H.shape == (2, len(robot.symbols.dynparms()))


def test_dh_park_path():
    robot = Robot.from_dh(
        "rr",
        [
            ("0", "1", "0", "q"),
            ("0", "1", "0", "q"),
        ],
    )
    assert robot.convention == "standard"
    assert robot.frames.S_body is not None
    tau = robot.dynamics.gen_invdyn()
    assert tau.shape == (2, 1)


if __name__ == "__main__":
    test_mdh_fk_and_invdyn()
    test_dh_park_path()
    print("robot_model smoke OK")
