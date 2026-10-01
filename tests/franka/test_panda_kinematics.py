"""Panda 臂运动学对拍：robot_model MDH FK vs MuJoCo link1–link7（不含手爪）。"""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from robot_model import Robot
from robot_model.robots.franka import (
    PANDA_MDH_BODY_NAMES,
    PANDA_MDH_PARMS,
    panda_xml_path,
)

mujoco = pytest.importorskip("mujoco")

ATOL_POS = 1e-9
ATOL_ROT = 1e-9


def _rotation_error(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.linalg.norm(Ra @ Rb.T - np.eye(3)))


@pytest.fixture(scope="module")
def panda_robot():
    return Robot.from_mdh("franka_panda_arm", PANDA_MDH_PARMS)


@pytest.fixture(scope="module")
def fk_funcs(panda_robot):
    q = list(panda_robot.q)
    return [
        sympy.lambdify(q, panda_robot.frames.T[i], "numpy") for i in range(7)
    ]


@pytest.fixture(scope="module")
def mj_model():
    return mujoco.MjModel.from_xml_path(str(panda_xml_path()))


def _sample_configs(model) -> list[np.ndarray]:
    home = np.array(model.key("home").qpos[:7], dtype=float)
    rng = np.random.default_rng(0)
    return [
        np.zeros(7),
        home,
        rng.uniform(-1.0, 1.0, size=7),
        rng.uniform(-1.2, 1.2, size=7),
    ]


def test_panda_arm_fk_pose_match_mujoco(fk_funcs, mj_model):
    data = mujoco.MjData(mj_model)
    body_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in PANDA_MDH_BODY_NAMES
    ]
    assert all(i >= 0 for i in body_ids)

    for q in _sample_configs(mj_model):
        data.qpos[:] = 0.0
        data.qpos[:7] = q
        mujoco.mj_forward(mj_model, data)
        for i, bid in enumerate(body_ids):
            T = np.asarray(fk_funcs[i](*q), dtype=float)
            pos_err = np.linalg.norm(T[:3, 3] - data.xpos[bid])
            rot_err = _rotation_error(T[:3, :3], data.xmat[bid].reshape(3, 3))
            assert pos_err < ATOL_POS, f"{PANDA_MDH_BODY_NAMES[i]} pos {pos_err}"
            assert rot_err < ATOL_ROT, f"{PANDA_MDH_BODY_NAMES[i]} rot {rot_err}"


def test_fixture_does_not_use_hand_frame():
    assert "hand" not in PANDA_MDH_BODY_NAMES
    assert PANDA_MDH_PARMS[-1][2] in ("0", 0)
