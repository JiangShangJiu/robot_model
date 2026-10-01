"""带摩擦 / 电机惯量的辨识，以及走完整闭环采集的辨识。

理想模型（`rrr_arm.xml`）刻意不含摩擦，用于纯刚体验证；摩擦在这里
按需注入到 model 字段上，这样两类测试互不干扰，也免得维护第二个 XML。

关节侧参数的对应关系::

    MuJoCo dof_damping      -> fv  粘性摩擦
    MuJoCo dof_frictionloss -> fc  库仑摩擦
    MuJoCo dof_armature     -> Ia  电机等效惯量
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from robot_model import Robot
from robot_model.data import drop_low_speed, sample_trajectory
from robot_model.excitation import search_excitation
from robot_model.identification import identify, parameter_error
from robot_model.simulation import MujocoTrackingSource
from robot_model.simulation.reference import MujocoReference
from robot_model.robots.rrr import (
    RRR_BODY_NAMES,
    RRR_MDH_PARMS,
    rrr_xml_path,
)

mujoco = pytest.importorskip("mujoco")

DAMPING = np.array([0.30, 0.25, 0.15])
ARMATURE = np.array([0.05, 0.04, 0.02])
FRICTIONLOSS = np.array([0.20, 0.15, 0.10])


def _model_with_friction():
    model = mujoco.MjModel.from_xml_path(str(rrr_xml_path()))
    model.dof_damping[:] = DAMPING
    model.dof_armature[:] = ARMATURE
    model.dof_frictionloss[:] = FRICTIONLOSS
    return model


@pytest.fixture(scope="module")
def setup():
    """``(Hb_func, pi_b 真值, MuJoCo 参照, 采样数据)``。"""
    model = _model_with_friction()
    ref = MujocoReference(model, RRR_BODY_NAMES)

    robot = Robot.from_mdh(
        "rrr_friction",
        RRR_MDH_PARMS,
        frictionmodel={"viscous", "Coulomb"},
        driveinertiamodel="simplified",
    )
    dyn = robot.dynamics
    dyn.gen_regressor()
    dyn.calc_base_parms()
    Hb = dyn.gen_base_regressor()

    s = robot.symbols
    f = sympy.lambdify(list(s.q) + list(s.dq) + list(s.ddq), Hb, "numpy")

    def Hb_func(q, dq, ddq):
        return np.asarray(f(*(list(q) + list(dq) + list(ddq))), dtype=float)

    full = []
    for i, link in enumerate(ref.links):
        full += list(link.Le) + list(link.l) + [link.m]
        full += [ARMATURE[i], DAMPING[i], FRICTIONLOSS[i]]
    pi_b = np.asarray(
        sympy.lambdify(list(s.dynparms()), sympy.Matrix(dyn.baseparms), "numpy")(
            *np.array(full)
        ),
        dtype=float,
    ).ravel()

    traj, _ = search_excitation(
        Hb_func, 3, np.random.default_rng(0), trials=20, vel_scale=1.2
    )
    data = sample_trajectory(traj, ref.inverse_dynamics, rate=100)
    return Hb_func, pi_b, model, traj, data


def test_friction_parameters_enter_the_parameter_vector():
    robot = Robot.from_mdh(
        "rrr_friction",
        RRR_MDH_PARMS,
        frictionmodel={"viscous", "Coulomb"},
        driveinertiamodel="simplified",
    )
    names = [str(p) for p in robot.symbols.dynparms()]
    assert len(names) == 3 * (10 + 3)  # 每连杆 10 个惯性 + Ia/fv/fc
    for i in (1, 2, 3):
        assert f"Ia_{i}" in names and f"fv_{i}" in names and f"fc_{i}" in names


def test_inverse_dynamics_matches_mujoco_with_friction():
    """先确认正向模型一致，否则辨识出的偏差无从归因。"""
    model = _model_with_friction()
    ref = MujocoReference(model, RRR_BODY_NAMES)
    robot = Robot.from_mdh(
        "rrr_friction",
        RRR_MDH_PARMS,
        frictionmodel={"viscous", "Coulomb"},
        driveinertiamodel="simplified",
    )
    nd = robot.dynamics.numeric_dynamics()
    m, l, Le = ref.inertia_args()
    rng = np.random.default_rng(2)
    for _ in range(5):
        q = rng.uniform(-1.5, 1.5, 3)
        dq = rng.uniform(0.5, 1.5, 3) * rng.choice([-1, 1], 3)  # 避开过零
        ddq = rng.uniform(-2, 2, 3)
        tau = nd.invdyn(
            q, dq, ddq, m, l, Le, Ia=ARMATURE, fv=DAMPING, fc=FRICTIONLOSS
        )
        assert np.allclose(tau, ref.inverse_dynamics(q, dq, ddq), atol=1e-9)


def test_coulomb_model_diverges_near_zero_velocity():
    """MuJoCo 用约束求解干摩擦，本库用 fc*sign(dq)，低速区必然不一致。

    这不是 bug，是模型适用范围——辨识时要靠剔除低速点规避。
    """
    model = _model_with_friction()
    ref = MujocoReference(model, RRR_BODY_NAMES)
    robot = Robot.from_mdh(
        "rrr_friction",
        RRR_MDH_PARMS,
        frictionmodel={"viscous", "Coulomb"},
        driveinertiamodel="simplified",
    )
    nd = robot.dynamics.numeric_dynamics()
    m, l, Le = ref.inertia_args()
    q, ddq = np.array([0.3, -0.5, 0.8]), np.zeros(3)

    def err(speed):
        dq = np.full(3, speed)
        tau = nd.invdyn(
            q, dq, ddq, m, l, Le, Ia=ARMATURE, fv=DAMPING, fc=FRICTIONLOSS
        )
        return np.max(np.abs(tau - ref.inverse_dynamics(q, dq, ddq)))

    assert err(1.0) < 1e-9  # 正常速度一致
    assert err(1e-4) > 1e-2  # 接近静止时显著偏离


def test_identification_recovers_friction_parameters(setup):
    Hb_func, pi_b, _, _, data = setup
    result = identify(Hb_func, drop_low_speed(data, 0.05))
    err = parameter_error(result.parms, pi_b)
    assert err["rel_norm"] < 1e-9, result.summary()


def test_low_speed_samples_bias_the_estimate(setup):
    """不剔除低速点，库仑模型失配会把参数带偏一个数量级以上。"""
    Hb_func, pi_b, _, _, data = setup
    with_all = parameter_error(identify(Hb_func, data).parms, pi_b)["rel_norm"]
    cleaned = parameter_error(
        identify(Hb_func, drop_low_speed(data, 0.05)).parms, pi_b
    )["rel_norm"]
    assert with_all > 1e-3
    assert cleaned < with_all / 100


def test_tracking_source_follows_trajectory(setup):
    _, _, model, traj, _ = setup
    source = MujocoTrackingSource(model, kp=50.0, kd=2.0)
    assert source._adr is not None, "应当读到 enc/tach/trq 传感器"
    data = source.run(traj, decimate=5)
    q_des, _, _ = traj.evaluate(data.t)
    assert np.abs(data.q - q_des).max() < 1e-2


def test_closed_loop_acquisition_identifies_parameters(setup):
    """完整闭环：控制器跟踪 → 传感器读数 → 微分 → 辨识。

    精度必然不如直接用 mj_inverse（加速度只能靠微分），
    但应当仍在百分之一量级。
    """
    Hb_func, pi_b, model, traj, _ = setup
    source = MujocoTrackingSource(model, kp=50.0, kd=2.0)
    data = source.run(traj, decimate=5)
    result = identify(Hb_func, drop_low_speed(data, 0.05))
    err = parameter_error(result.parms, pi_b)["rel_norm"]
    assert err < 1e-2, result.summary()
