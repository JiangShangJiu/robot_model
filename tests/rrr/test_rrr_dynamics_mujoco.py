"""空间 3R：动力学 vs MuJoCo ``mj_inverse``。

这是本库动力学唯一的**外部**真值对拍。其余动力学测试都是库内互拍
（Park / Khalil / 拉格朗日 / 数值 RNE），只能发现实现之间不一致，
发现不了几套实现按同一个错误约定写成。3 轴规模让全符号链路
（invdyn、回归矩阵 H、最小参数集）都能在秒级跑完，所以这里一并覆盖。
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from robot_model import Robot
from robot_model.simulation.reference import MujocoReference
from robot_model.robots.rrr import (
    RRR_BODY_NAMES,
    RRR_MDH_PARMS,
    rrr_xml_path,
)

mujoco = pytest.importorskip("mujoco")

ATOL = 1e-9


@pytest.fixture(scope="module")
def robot():
    return Robot.from_mdh("rrr_arm", RRR_MDH_PARMS)


@pytest.fixture(scope="module")
def ref():
    model = mujoco.MjModel.from_xml_path(str(rrr_xml_path()))
    return MujocoReference(model, RRR_BODY_NAMES)


@pytest.fixture(scope="module")
def states() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(7)
    zero = np.zeros(3)
    out = [
        (zero.copy(), zero.copy(), zero.copy()),  # 纯重力，零位
        (np.array([0.3, -0.6, 1.2]), zero.copy(), zero.copy()),  # 纯重力
        (rng.uniform(-1.5, 1.5, 3), rng.uniform(-1, 1, 3), zero.copy()),  # +科氏
    ]
    out += [
        (
            rng.uniform(-1.5, 1.5, 3),
            rng.uniform(-1.5, 1.5, 3),
            rng.uniform(-2, 2, 3),
        )
        for _ in range(4)
    ]
    return out


def _lambdify(robot, expr, *, dq=False, ddq=False, parms=False):
    s = robot.symbols
    args = list(s.q)
    if dq:
        args += list(s.dq)
    if ddq:
        args += list(s.ddq)
    if parms:
        args += list(s.dynparms())
    return sympy.lambdify(args, expr, "numpy")


def test_gravity_only_has_no_torque_on_waist(robot, ref):
    """关节 1 轴与重力平行 → tau_1 恒为 0，两边都应如此。"""
    m, l, Le = ref.inertia_args()
    nd = robot.dynamics.numeric_dynamics()
    zero = np.zeros(3)
    for q in (zero, np.array([0.3, -0.6, 1.2]), np.array([-1.0, 0.8, -0.4])):
        assert abs(ref.inverse_dynamics(q, zero, zero)[0]) < ATOL
        assert abs(nd.invdyn(q, zero, zero, m, l, Le)[0]) < ATOL


def test_numeric_invdyn_matches_mujoco(robot, ref, states):
    m, l, Le = ref.inertia_args()
    nd = robot.dynamics.numeric_dynamics()
    for q, dq, ddq in states:
        tau = nd.invdyn(q, dq, ddq, m, l, Le)
        assert np.allclose(tau, ref.inverse_dynamics(q, dq, ddq), atol=ATOL)


def test_symbolic_invdyn_matches_mujoco(robot, ref, states):
    pi = ref.dynparms()
    f = _lambdify(
        robot, robot.dynamics.gen_invdyn(), dq=True, ddq=True, parms=True
    )
    for q, dq, ddq in states:
        tau = np.asarray(
            f(*(list(q) + list(dq) + list(ddq) + list(pi))), dtype=float
        ).ravel()
        assert np.allclose(tau, ref.inverse_dynamics(q, dq, ddq), atol=ATOL)


def test_regressor_times_parms_matches_mujoco(robot, ref, states):
    """``tau = H pi`` 同时校验线性参数化与 dynparms 的排列顺序。"""
    pi = ref.dynparms()
    f = _lambdify(robot, robot.dynamics.gen_regressor(), dq=True, ddq=True)
    for q, dq, ddq in states:
        H = np.asarray(f(*(list(q) + list(dq) + list(ddq))), dtype=float)
        assert np.allclose(H @ pi, ref.inverse_dynamics(q, dq, ddq), atol=ATOL)


def test_mass_coriolis_gravity_split_matches_mujoco(robot, ref, states):
    """``M ddq + c + g`` 必须重建出同一个 tau，校验 derived 那套分解。"""
    pi = ref.dynparms()
    dyn = robot.dynamics
    f_M = _lambdify(robot, dyn.gen_inertiamatrix(), parms=True)
    f_c = _lambdify(robot, dyn.gen_coriolisterm(), dq=True, parms=True)
    f_g = _lambdify(robot, dyn.gen_gravityterm(), parms=True)
    for q, dq, ddq in states:
        M = np.asarray(f_M(*(list(q) + list(pi))), dtype=float)
        c = np.asarray(
            f_c(*(list(q) + list(dq) + list(pi))), dtype=float
        ).ravel()
        g = np.asarray(f_g(*(list(q) + list(pi))), dtype=float).ravel()
        tau = M @ ddq + c + g
        assert np.allclose(tau, ref.inverse_dynamics(q, dq, ddq), atol=ATOL)


def test_base_regressor_matches_mujoco(robot, ref, states):
    """最小参数集：``tau = Hb pi_b``，端到端校验辨识链路。"""
    dyn = robot.dynamics
    dyn.gen_regressor()
    dyn.calc_base_parms()
    Hb = dyn.gen_base_regressor()
    assert Hb is not None
    assert 0 < dyn.n_base < len(list(robot.symbols.dynparms()))

    f_Hb = _lambdify(robot, Hb, dq=True, ddq=True)
    # baseparms 是用原始参数符号写出的 pi_b 表达式，代入真值即得数值
    f_pib = sympy.lambdify(
        list(robot.symbols.dynparms()), sympy.Matrix(dyn.baseparms), "numpy"
    )
    pi_b = np.asarray(f_pib(*ref.dynparms()), dtype=float).ravel()

    for q, dq, ddq in states:
        H = np.asarray(
            f_Hb(*(list(q) + list(dq) + list(ddq))), dtype=float
        )
        assert np.allclose(H @ pi_b, ref.inverse_dynamics(q, dq, ddq), atol=ATOL)
