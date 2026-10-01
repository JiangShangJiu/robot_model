"""空间 3R：MDH 运动学 vs MuJoCo（逐帧位姿、末端 site、几何雅可比）。"""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from robot_model import Robot
from robot_model.simulation.reference import MujocoReference
from robot_model.robots.rrr import (
    RRR_BODY_NAMES,
    RRR_MDH_PARMS,
    RRR_TOOL_OFFSET,
    rrr_xml_path,
)

mujoco = pytest.importorskip("mujoco")

ATOL = 1e-9


@pytest.fixture(scope="module")
def robot():
    return Robot.from_mdh("rrr_arm", RRR_MDH_PARMS)


@pytest.fixture(scope="module")
def mj_model():
    return mujoco.MjModel.from_xml_path(str(rrr_xml_path()))


@pytest.fixture(scope="module")
def ref(mj_model):
    return MujocoReference(mj_model, RRR_BODY_NAMES)


@pytest.fixture(scope="module")
def fk_funcs(robot):
    q = list(robot.q)
    return [sympy.lambdify(q, robot.frames.T[i], "numpy") for i in range(3)]


@pytest.fixture(scope="module")
def jac_funcs(robot):
    q = list(robot.q)
    return [sympy.lambdify(q, robot.kinematics.J[i], "numpy") for i in range(3)]


def _configs() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [
        np.zeros(3),
        np.array([0.3, -0.6, 1.2]),
        rng.uniform(-1.5, 1.5, 3),
        rng.uniform(-2.5, 2.5, 3),
    ]


def test_link_poses_match_mujoco(fk_funcs, ref):
    """MDH 帧 i 与 body link{i} 必须逐帧重合，这是后续动力学对拍的前提。"""
    for q in _configs():
        for i, T_mj in enumerate(ref.body_poses(q)):
            T = np.asarray(fk_funcs[i](*q), dtype=float)
            pos_err = np.linalg.norm(T[:3, 3] - T_mj[:3, 3])
            rot_err = np.linalg.norm(T[:3, :3] @ T_mj[:3, :3].T - np.eye(3))
            assert pos_err < ATOL, f"{RRR_BODY_NAMES[i]} pos {pos_err}"
            assert rot_err < ATOL, f"{RRR_BODY_NAMES[i]} rot {rot_err}"


def test_tool_offset_matches_site(fk_funcs, mj_model, ref):
    sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "tool")
    assert sid >= 0
    offset = np.array(RRR_TOOL_OFFSET)
    for q in _configs():
        ref.body_poses(q)  # 同步 ref.data
        T = np.asarray(fk_funcs[2](*q), dtype=float)
        tool = T[:3, 3] + T[:3, :3] @ offset
        assert np.linalg.norm(tool - ref.data.site_xpos[sid]) < ATOL


def test_tool_sensors_match_forward_kinematics(fk_funcs, jac_funcs, ref):
    """末端 site 的位置 / 速度传感器，对应 FK 与雅可比平移到 site 之后的结果。"""
    offset = np.array(RRR_TOOL_OFFSET)
    rng = np.random.default_rng(4)
    for _ in range(4):
        q, dq = rng.uniform(-1.5, 1.5, 3), rng.uniform(-1.5, 1.5, 3)
        ref.forward(q, dq)
        T = np.asarray(fk_funcs[2](*q), dtype=float)
        r = T[:3, :3] @ offset

        assert np.allclose(T[:3, 3] + r, ref.sensor("tool_pos"), atol=ATOL)

        J = np.asarray(jac_funcs[2](*q), dtype=float)
        v_link, omega = J[:3] @ dq, J[3:] @ dq
        assert np.allclose(
            v_link + np.cross(omega, r), ref.sensor("tool_linvel"), atol=ATOL
        )


def test_subtree_com_sensor_matches_link_masses(fk_funcs, ref):
    """质心传感器对上各连杆一阶矩之和——独立校验惯性参数的提取。"""
    rng = np.random.default_rng(6)
    total = sum(lk.m for lk in ref.links)
    for _ in range(3):
        q = rng.uniform(-1.5, 1.5, 3)
        poses = ref.body_poses(q)
        com = (
            sum(
                lk.m * (T[:3, :3] @ lk.r + T[:3, 3])
                for lk, T in zip(ref.links, poses)
            )
            / total
        )
        assert np.allclose(com, ref.sensor("arm_com"), atol=ATOL)


def test_joint_sensors_echo_state(ref):
    rng = np.random.default_rng(8)
    q, dq = rng.uniform(-1.5, 1.5, 3), rng.uniform(-1.5, 1.5, 3)
    ref.forward(q, dq)
    encoders = [ref.sensor(f"enc{i}")[0] for i in (1, 2, 3)]
    tachos = [ref.sensor(f"tach{i}")[0] for i in (1, 2, 3)]
    assert np.allclose(encoders, q, atol=ATOL)
    assert np.allclose(tachos, dq, atol=ATOL)


def test_geometric_jacobian_matches_mujoco(jac_funcs, mj_model, ref):
    """``J=[Jp;Jo]`` 与 ``mj_jacBody`` 的 (jacp, jacr) 对齐（均在基座系）。"""
    jacp = np.zeros((3, mj_model.nv))
    jacr = np.zeros((3, mj_model.nv))
    for q in _configs():
        ref.body_poses(q)
        for i, bid in enumerate(ref.body_ids):
            mujoco.mj_jacBody(mj_model, ref.data, jacp, jacr, bid)
            J = np.asarray(jac_funcs[i](*q), dtype=float)
            assert np.allclose(J[:3], jacp, atol=ATOL), RRR_BODY_NAMES[i]
            assert np.allclose(J[3:], jacr, atol=ATOL), RRR_BODY_NAMES[i]
