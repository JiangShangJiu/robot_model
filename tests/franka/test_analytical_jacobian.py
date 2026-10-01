"""解析雅可比（Franka / Panda）。"""

from __future__ import annotations

import numpy as np

from robot_model import Robot
from robot_model.robots.franka import PANDA_MDH_PARMS


def test_analytical_panda_direct_vs_converted():
    robot = Robot.from_mdh("panda", PANDA_MDH_PARMS)
    kin = robot.kinematics
    q = np.array([0.2, -0.4, 0.3, -1.0, 0.5, 0.7, -0.3])

    Ja = kin.ja_numpy(q)
    Ja_c = kin.ja_from_geometric_numpy(q)
    assert np.linalg.norm(Ja[:3] - Ja_c[:3]) < 1e-6
    assert np.linalg.norm(Ja[3:] - Ja_c[3:]) < 5e-6


def test_analytical_poe_matches_mdh_and_conversion():
    mdh = Robot.from_mdh("panda", PANDA_MDH_PARMS)
    poe = mdh.to_poe()
    q = np.array([0.2, -0.4, 0.3, -1.0, 0.5, 0.7, -0.3])

    Ja_m = mdh.kinematics.ja_numpy(q)
    Ja_p = poe.kinematics.ja_numpy(q)
    Ja_pc = poe.kinematics.ja_from_geometric_numpy(q)
    assert np.linalg.norm(Ja_m - Ja_p) < 1e-6
    assert np.linalg.norm(Ja_p - Ja_pc) < 5e-6


if __name__ == "__main__":
    test_analytical_panda_direct_vs_converted()
    test_analytical_poe_matches_mdh_and_conversion()
    print("franka analytical jacobian OK")
