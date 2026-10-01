"""DH / MDH / PoE 互转：平面 2R。"""

from __future__ import annotations

import numpy as np

from robot_model import Robot


def _assert_fk(a: Robot, b: Robot, q: np.ndarray, tol: float = 1e-9):
    r = a.compare_fk(b, q)
    assert r["pos_err"] < tol, r["pos_err"]
    assert r["rot_err"] < tol, r["rot_err"]


def test_dh_poe_dh_roundtrip_planar():
    dh = Robot.from_dh(
        "planar2r",
        [("0", "1", "0", "q"), ("0", "1", "0", "q")],
    )
    back = dh.to_poe().to_dh()
    assert back.convention == "standard"
    q = np.array([0.3, -0.7])
    _assert_fk(dh, back, q)
    for i in range(2):
        r = dh.compare_fk(back, q, link=i)
        assert r["pos_err"] < 1e-12
        assert r["rot_err"] < 1e-12


if __name__ == "__main__":
    test_dh_poe_dh_roundtrip_planar()
    print("planar convention convert OK")
