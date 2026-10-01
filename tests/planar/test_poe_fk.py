"""PoE 正运动学：平面 2R。"""

from __future__ import annotations

import numpy as np
import sympy

from robot_model import Robot


def test_planar2r_poe_matches_mdh():
    mdh = Robot.from_mdh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    poe = mdh.to_poe()
    assert poe.convention == "poe"

    q = np.array([0.3, -0.7])
    r = mdh.compare_fk(poe, q)
    assert r["pos_err"] < 1e-12
    assert r["rot_err"] < 1e-12

    dh = Robot.from_dh(
        "planar2r_dh",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    S1 = [0, 0, 1, 0, 0, 0]
    S2 = [0, 0, 1, 0, -1, 0]
    M = sympy.Matrix(
        [[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    hand = Robot.from_poe("planar2r_poe", [S1, S2], M)
    r2 = dh.compare_fk(hand, q)
    assert r2["pos_err"] < 1e-12
    assert r2["rot_err"] < 1e-12
    r3 = dh.compare_fk(dh.to_poe(), q)
    assert r3["pos_err"] < 1e-12
    assert r3["rot_err"] < 1e-12


if __name__ == "__main__":
    test_planar2r_poe_matches_mdh()
    print("planar poe fk OK")
