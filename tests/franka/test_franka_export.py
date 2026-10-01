"""Franka 7 轴走完整符号路径：导出代码必须对上 MuJoCo 真值。

这是 3R 之外第二个有**动力学**外部真值的模型。7 轴的意义在于规模：
``tau`` 展开有 450 万个节点，``C`` 有 530 万个——只有导出层不在大表达式上
做 ``subs`` / ``simplify``，这条路才走得通。所以本文件既验数值，也是
「7 轴符号导出仍然可用」这件事的回归测试。

整套符号量生成 + cse 约两分钟，故整模块标记 ``slow``，
日常可用 ``pytest -m "not slow"`` 跳过。
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from robot_model import Robot
from robot_model.robots.franka import (
    PANDA_MDH_BODY_NAMES,
    PANDA_MDH_PARMS,
    panda_xml_path,
)
from robot_model.simulation.reference import MujocoReference, strip_to_rigid_body

mujoco = pytest.importorskip("mujoco")

pytestmark = pytest.mark.slow

ATOL = 1e-9

# 导出哪些量：这一串正是 notebooks/franka/modeling.ipynb 落到
# outputs/franka_symbolic/ 的内容，测试与 notebook 保持同一个集合
KEYS = ("T", "M", "c", "g", "tau", "H", "Hb", "baseparms")


@pytest.fixture(scope="module")
def robot():
    return Robot.from_mdh("franka_panda_arm", PANDA_MDH_PARMS)


@pytest.fixture(scope="module")
def ref():
    model = strip_to_rigid_body(
        mujoco.MjModel.from_xml_path(str(panda_xml_path()))
    )
    return MujocoReference(model, PANDA_MDH_BODY_NAMES)


@pytest.fixture(scope="module")
def generated(robot, tmp_path_factory):
    """生成并 import 一次；后续用例都用生成的代码算数值。

    不用 ``sympy.lambdify``：对 450 万节点的 ``tau`` 直接 lambdify 要几分钟，
    而导出层的 cse 把它压到一千多个中间量，秒级就能编译。
    """
    path = tmp_path_factory.mktemp("export") / "franka_dynamics.py"
    robot.export.python(path, keys=KEYS)
    spec = importlib.util.spec_from_file_location("franka_generated", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _states():
    rng = np.random.default_rng(5)
    return [
        (
            rng.uniform(-1.2, 1.2, 7),
            rng.uniform(-1.0, 1.0, 7),
            rng.uniform(-2.0, 2.0, 7),
        )
        for _ in range(3)
    ]


def test_generated_torque_matches_mj_inverse(generated, ref):
    """符号 tau 与 mj_inverse 逐点对拍——唯一的外部真值检验。"""
    pi = ref.dynparms()
    for q, dq, ddq in _states():
        tau_mj = ref.inverse_dynamics(q, dq, ddq)
        assert np.allclose(generated.tau(q, dq, ddq, pi), tau_mj, atol=ATOL)
        split = (
            generated.M(q, pi) @ ddq
            + generated.c(q, dq, pi)
            + generated.g(q, pi)
        )
        assert np.allclose(split, tau_mj, atol=ATOL)


def test_generated_inertia_matches_mj_fullM(generated, ref):
    pi = ref.dynparms()
    data = mujoco.MjData(ref.model)
    for q, _, _ in _states():
        data.qpos[:] = q
        mujoco.mj_forward(ref.model, data)
        M_mj = np.zeros((7, 7))
        mujoco.mj_fullM(ref.model, M_mj, data.qM)
        assert np.allclose(generated.M(q, pi), M_mj, atol=ATOL)


def test_generated_kinematics_matches_mujoco(generated, ref):
    for q, _, _ in _states():
        assert np.allclose(generated.T(q), ref.body_poses(q)[-1], atol=ATOL)


def test_regressor_forms_reproduce_torque(generated, ref):
    """``tau = H pi = Hb pi_b``：两种回归形式都要落回真值。"""
    pi = ref.dynparms()
    pi_b = generated.baseparms(pi)
    for q, dq, ddq in _states():
        tau_mj = ref.inverse_dynamics(q, dq, ddq)
        assert np.allclose(generated.H(q, dq, ddq) @ pi, tau_mj, atol=ATOL)
        assert np.allclose(generated.Hb(q, dq, ddq) @ pi_b, tau_mj, atol=ATOL)


def test_base_parms_reduce_parameter_count(robot, generated):
    """7 轴 70 个完整参数里只有 43 个可辨识，这个数是模型的固有属性。"""
    dyn = robot.dynamics
    assert dyn.n_dynparms == 70
    assert dyn.n_base == 43
    q, dq, ddq = _states()[0]
    assert generated.H(q, dq, ddq).shape == (7, 70)
    assert generated.Hb(q, dq, ddq).shape == (7, 43)


def test_gravity_vanishes_on_first_joint(generated, ref):
    """关节 1 轴与重力平行，g[0] 恒为 0——拟人臂的固有性质，不是巧合。"""
    pi = ref.dynparms()
    for q, _, _ in _states():
        assert abs(generated.g(q, pi)[0]) < ATOL
