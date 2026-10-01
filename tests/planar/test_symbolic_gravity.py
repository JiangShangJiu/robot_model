"""符号重力：可走符号路径，数值路径须给出明确报错。"""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from robot_model import Robot

PARMS = [("0", "1", "0", "q"), ("0", "1", "0", "q")]
G = sympy.symbols("g", real=True, positive=True)


def _robot(gravityacc):
    return Robot.from_mdh("planar2r", PARMS, gravityacc=gravityacc)


def test_symbolic_gravity_flows_into_symbolic_terms():
    # 重力需落在运动平面内，否则平面 2R 的重力项恒为 0
    robot = _robot(sympy.Matrix([[0], [-G], [0]]))

    gterm = robot.dynamics.gen_gravityterm()
    assert G in gterm.free_symbols

    H = robot.dynamics.gen_regressor()
    assert G in H.free_symbols
    # g 是常系数，不应混进动力学参数向量 π
    assert G not in set(robot.dynamics.dynparms)


def test_numeric_path_rejects_symbolic_gravity():
    robot = _robot(sympy.Matrix([[0], [-G], [0]]))
    z = np.zeros(2)

    with pytest.raises(TypeError, match="gravityacc"):
        robot.dynamics.regressor_numpy(z, z, z)


def test_numeric_path_accepts_custom_constant_gravity():
    robot = _robot([[0], [-9.81], [0]])
    z = np.zeros(2)

    H = robot.dynamics.regressor_numpy(z, z, z)
    assert H.shape == (2, robot.dynamics.n_dynparms)
    assert np.isfinite(H).all()


if __name__ == "__main__":
    test_symbolic_gravity_flows_into_symbolic_terms()
    test_numeric_path_rejects_symbolic_gravity()
    test_numeric_path_accepts_custom_constant_gravity()
    print("symbolic gravity OK")
