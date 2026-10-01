"""质心速度应等于质心位置对关节坐标的导数，与连杆质量无关。"""

import pytest
import sympy

from robot_model import Robot


@pytest.mark.parametrize('convention', ['dh', 'mdh', 'poe'])
@pytest.mark.parametrize('second_joint', ['revolute', 'prismatic'])
def test_com_jacobian_matches_position_derivative(convention, second_joint):
    second = (sympy.pi / 2, 1, 0, 'q') if second_joint == 'revolute' else (sympy.pi / 2, 1, 'q', 0)
    rows = [(0, 1, 0, 'q'), second]
    robot = (Robot.from_dh('com', rows) if convention == 'dh'
             else Robot.from_mdh('com', rows))
    if convention == 'poe':
        robot = robot.to_poe()
    kin, symbols, frames = robot.kinematics, robot.symbols, robot.frames
    inertial_symbols = set(symbols.m) | set(sympy.flatten(symbols.l))
    for link in range(robot.dof):
        position = frames.p[link] + frames.R[link] * symbols.r[link]
        expected = position.jacobian(symbols.q)
        error = kin.Jcp[link] - expected
        assert all(sympy.trigsimp(value) == 0 for value in error)
        assert not (kin.Jcp[link].free_symbols & inertial_symbols)
        assert kin.Jc[link][:3, :] == kin.Jcp[link]
        assert kin.Jc[link][3:, :] == kin.Jo[link]
