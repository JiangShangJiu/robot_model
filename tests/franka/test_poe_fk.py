"""PoE 正运动学：Franka MDH→PoE，末端 vs MuJoCo。"""

from __future__ import annotations

import numpy as np

from robot_model import Robot
from robot_model.robots.franka import PANDA_MDH_PARMS, panda_xml_path


def pytest_import_mujoco():
    import importlib

    return importlib.import_module("mujoco")


def test_panda_poe_ee_matches_mdh_and_mujoco():
    mujoco = pytest_import_mujoco()
    mdh = Robot.from_mdh("franka_panda_arm", PANDA_MDH_PARMS)
    poe = mdh.to_poe()

    model = mujoco.MjModel.from_xml_path(str(panda_xml_path()))
    data = mujoco.MjData(model)
    link7 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    home = np.array(model.key("home").qpos[:7], dtype=float)
    rng = np.random.default_rng(1)
    cfgs = [
        np.zeros(7),
        home,
        np.full(7, 0.2),
        np.array([0.1, -0.5, 0.2, -1.0, 0.3, 0.8, -0.4]),
        rng.uniform(-1.0, 1.0, size=7),
    ]

    for q in cfgs:
        r = mdh.compare_fk(poe, q)
        assert r["pos_err"] < 1e-9, r["pos_err"]
        assert r["rot_err"] < 1e-9, r["rot_err"]

        data.qpos[:] = 0.0
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        T = poe.fk_numpy(q)
        pos_err = np.linalg.norm(T[:3, 3] - data.xpos[link7])
        rot_err = np.linalg.norm(
            T[:3, :3] @ data.xmat[link7].reshape(3, 3).T - np.eye(3)
        )
        assert pos_err < 1e-9, pos_err
        assert rot_err < 1e-9, rot_err


def test_body_form_poe_matches_space_form():
    """B_i = Ad_{M_ee^{-1}} S_i 送入 frame='body'，应还原同一模型。"""
    import sympy

    from robot_model.utils.mathutil import adjoint_twist, inverse_T

    mdh = Robot.from_mdh("franka_panda_arm", PANDA_MDH_PARMS)
    poe = mdh.to_poe()
    desc = poe.description
    M_ee_inv = inverse_T(sympy.Matrix(desc.M[-1]))
    body_screws = [
        adjoint_twist(M_ee_inv, sympy.Matrix(s)) for s in desc.screws
    ]

    body = Robot.from_poe(
        "franka_panda_body",
        body_screws,
        desc.M,
        frame="body",
        sigma=desc.sigma,
        home_convention=desc.home_convention,
    )

    q = np.array([0.1, -0.5, 0.2, -1.0, 0.3, 0.8, -0.4])
    for i in range(7):
        r = mdh.compare_fk(body, q, link=i)
        assert r["pos_err"] < 1e-9, (i, r["pos_err"])
        assert r["rot_err"] < 1e-9, (i, r["rot_err"])


def test_unknown_frame_rejected():
    import pytest

    with pytest.raises(ValueError, match="space.*body"):
        Robot.from_poe("bad", [[0, 0, 1, 0, 0, 0]], np.eye(4), frame="tool")


if __name__ == "__main__":
    test_panda_poe_ee_matches_mdh_and_mujoco()
    test_body_form_poe_matches_space_form()
    test_unknown_frame_rejected()
    print("franka poe fk OK")
