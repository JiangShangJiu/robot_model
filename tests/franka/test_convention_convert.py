"""DH / MDH / PoE 互转：Franka。"""

from __future__ import annotations

import numpy as np

from robot_model import Robot
from robot_model.robots.franka import PANDA_MDH_PARMS


def _assert_fk(a: Robot, b: Robot, q: np.ndarray, tol: float = 1e-9):
    r = a.compare_fk(b, q)
    assert r["pos_err"] < tol, r["pos_err"]
    assert r["rot_err"] < tol, r["rot_err"]


def test_mdh_poe_mdh_roundtrip():
    mdh = Robot.from_mdh("arm", PANDA_MDH_PARMS)
    back = mdh.to_poe().to_mdh()
    assert back.convention == "modified"
    q = np.array([0.1, -0.5, 0.2, -1.0, 0.3, 0.8, -0.4])
    _assert_fk(mdh, back, q)
    for i in range(7):
        r = mdh.compare_fk(back, q, link=i)
        assert r["pos_err"] < 1e-9, (i, r["pos_err"])
        assert r["rot_err"] < 1e-9, (i, r["rot_err"])


def test_triangle_mdh_dh_poe():
    mdh = Robot.from_mdh("arm", PANDA_MDH_PARMS)
    dh = mdh.to_dh()
    poe = mdh.to_poe()
    dh_from_poe = poe.to_dh()
    mdh_from_poe = poe.to_mdh()
    q = np.array([-0.2, 0.4, -0.3, -0.8, 0.5, 0.6, 0.1])
    for other in (dh, poe, dh_from_poe, mdh_from_poe):
        _assert_fk(mdh, other, q)


if __name__ == "__main__":
    test_mdh_poe_mdh_roundtrip()
    test_triangle_mdh_dh_poe()
    print("franka convention convert OK")
