"""DH/MDH 参数表打印与约定互转。"""

from __future__ import annotations

import sympy

from robot_model import Robot


def test_print_dh_table_contains_headers():
    robot = Robot.from_mdh(
        "arm",
        [
            ("0", "0", "0.333", "q"),
            ("-pi/2", "0", "0", "q"),
            ("pi/2", "0", "0.316", "q"),
        ],
    )
    text = robot.dh_table()
    assert "MDH" in text
    assert "α" in text
    assert "θ" in text
    assert "Joint1" in text
    assert "0.333" in text
    assert "0.333000000000000" not in text
    assert "转动关节" in text
    assert robot.convention == "modified"


def test_mdh_standard_roundtrip_when_alpha0_a0_zero():
    robot = Robot.from_mdh(
        "arm",
        [
            ("0", "0", "d1", "q"),
            ("alpha1", "a1", "d2", "q"),
            ("alpha2", "a2", "d3", "q"),
        ],
    )
    dh = robot.to_dh()
    assert dh.convention == "standard"
    back = dh.to_mdh()
    assert back.convention == "modified"
    for i in range(robot.dof):
        for j in range(4):
            assert sympy.simplify(back.description.parms[i][j] - robot.description.parms[i][j]) == 0
        assert back.description.sigma[i] == robot.description.sigma[i]


def test_shift_moves_a_alpha():
    robot = Robot.from_mdh(
        "arm",
        [
            ("0", "0", "d1", "q"),
            ("alpha1", "a1", "0", "q"),
        ],
    )
    dh = robot.to_dh()
    # MDH 第二行的 (α,a) 应落到标准 DH 第一行
    assert dh.description.parms[0][0] == robot.description.parms[1][0]
    assert dh.description.parms[0][1] == robot.description.parms[1][1]
    assert dh.description.parms[0][2] == robot.description.parms[0][2]
    assert dh.description.parms[1][0] == 0
    assert dh.description.parms[1][1] == 0


if __name__ == "__main__":
    test_print_dh_table_contains_headers()
    test_mdh_standard_roundtrip_when_alpha0_a0_zero()
    test_shift_moves_a_alpha()
    print("dh table/convert OK")
