"""Stribeck 摩擦（高斯低速摩擦峰）的建模、回归一致性与辨识。

模型（每关节）::

    tau_f = fv*dq + fc*sign(dq) + fs*exp(-(dq/v_s)^2)*sign(dq) [+ fo]

``fc + fs`` 是起动（breakaway）摩擦幅值，``fc`` 是滑动库仑摩擦；
特征速度 ``v_s``（``stribeck_velocity``）是固定模型常数，非线性地进入
指数项，不参与最小二乘；``fs`` 对力矩线性，随基参数一同辨识。
对象仿真侧对应 ``RealismConfig.low_speed_friction`` / ``friction_velocity``
注入的低速摩擦峰（以 ``tanh(dq/0.005)`` 代替 ``sign`` 平滑过零）。
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from robot_model import Robot
from robot_model.data import sample_trajectory
from robot_model.excitation import search_excitation
from robot_model.identification import identify, parameter_error
from robot_model.identification.recovery import (
    RecoveryConfig,
    _Layout,
    _bounds,
    pack_inertial_parameters,
)
from robot_model.robots.rrr import RRR_MDH_PARMS

STRIBECK_VELOCITY = (0.25, 0.30, 0.35)


def _robot(**kwargs):
    return Robot.from_mdh(
        "rrr_stribeck",
        RRR_MDH_PARMS,
        frictionmodel={"viscous", "Coulomb", "Stribeck"},
        driveinertiamodel="simplified",
        stribeck_velocity=STRIBECK_VELOCITY,
        **kwargs,
    )


def _friction_truth(rng):
    return {
        "Ia": rng.uniform(0.01, 0.08, 3),
        "fv": rng.uniform(0.1, 0.4, 3),
        "fc": rng.uniform(0.05, 0.3, 3),
        "fs": rng.uniform(0.03, 0.12, 3),
    }


def _rigid_truth(rng):
    m = rng.uniform(0.5, 3.0, 3)
    l = rng.normal(0.0, 0.2, (3, 3))
    Le = rng.normal(0.0, 0.02, (3, 6))
    Le[:, [0, 3, 5]] = rng.uniform(0.05, 0.2, (3, 3))
    return m, l, Le


def _full_vector(m, l, Le, extras):
    full = []
    for i in range(3):
        full += list(Le[i]) + list(l[i]) + [m[i]]
        full += [extras["Ia"][i], extras["fv"][i], extras["fc"][i], extras["fs"][i]]
    return np.asarray(full)


def test_stribeck_parameters_enter_the_parameter_vector():
    robot = _robot()
    names = [str(p) for p in robot.symbols.dynparms()]
    assert len(names) == 3 * (10 + 4)  # 每连杆 10 个惯性 + Ia/fv/fc/fs
    for i in (1, 2, 3):
        assert f"fs_{i}" in names
    assert robot.symbols.stribeck_velocity == STRIBECK_VELOCITY


def test_stribeck_velocity_is_validated_and_broadcast():
    scalar = Robot.from_mdh(
        "rrr_vs", RRR_MDH_PARMS, frictionmodel={"Stribeck"}, stribeck_velocity=0.2
    )
    assert scalar.symbols.stribeck_velocity == (0.2, 0.2, 0.2)
    default = Robot.from_mdh("rrr_vs0", RRR_MDH_PARMS, frictionmodel={"Stribeck"})
    assert default.symbols.stribeck_velocity == (0.1, 0.1, 0.1)
    for bad in (0.0, -0.1, np.nan, (0.1, 0.2)):
        with pytest.raises(ValueError):
            Robot.from_mdh(
                "rrr_bad", RRR_MDH_PARMS,
                frictionmodel={"Stribeck"}, stribeck_velocity=bad,
            )


def test_unknown_friction_term_is_rejected():
    robot = Robot.from_mdh("rrr_bad_term", RRR_MDH_PARMS, frictionmodel={"stribeck"})
    with pytest.raises(ValueError, match="unknown friction terms"):
        robot.dynamics.gen_invdyn()


def test_symbolic_and_numeric_regressors_match_inverse_dynamics():
    robot = _robot()
    s = robot.symbols
    dyn = robot.dynamics
    H = dyn.gen_regressor()
    f_sym = sympy.lambdify(list(s.q) + list(s.dq) + list(s.ddq), H, "numpy")
    nd = dyn.numeric_dynamics()

    rng = np.random.default_rng(3)
    m, l, Le = _rigid_truth(rng)
    extras = _friction_truth(rng)
    pi = _full_vector(m, l, Le, extras)
    # 覆盖 Stribeck 峰内外与过零附近的速度
    for speed_scale in (0.02, 0.2, 1.0):
        q = rng.uniform(-1.5, 1.5, 3)
        dq = rng.normal(0.0, speed_scale, 3)
        ddq = rng.normal(0.0, 2.0, 3)
        Hs = np.asarray(f_sym(*q, *dq, *ddq), dtype=float)
        Hn = nd.regressor(q, dq, ddq)
        tau = nd.invdyn(q, dq, ddq, m, l, Le, **extras)
        np.testing.assert_allclose(Hs, Hn, rtol=0, atol=1e-10)
        np.testing.assert_allclose(Hs @ pi, tau, rtol=0, atol=1e-10)


def test_identification_recovers_stribeck_friction():
    """理想采样下，含 Stribeck 项的基参数可被最小二乘精确恢复。"""
    robot = _robot()
    dyn = robot.dynamics
    dyn.gen_regressor()
    dyn.calc_base_parms()
    dyn.gen_base_regressor()
    Hb_func = dyn.Hb_func
    assert dyn.n_base == 25  # 22 个刚体/摩擦组合基参数 + 3 个独立 fs

    rng = np.random.default_rng(7)
    m, l, Le = _rigid_truth(rng)
    extras = _friction_truth(rng)
    full = _full_vector(m, l, Le, extras)
    s = robot.symbols
    pi_b = np.asarray(
        sympy.lambdify(list(s.dynparms()), sympy.Matrix(dyn.baseparms), "numpy")(*full),
        dtype=float,
    ).ravel()

    nd = dyn.numeric_dynamics()
    traj, _ = search_excitation(Hb_func, 3, np.random.default_rng(1), trials=10,
                                vel_scale=1.0)
    data = sample_trajectory(
        traj, lambda q, dq, ddq: nd.invdyn(q, dq, ddq, m, l, Le, **extras), rate=100,
    )
    result = identify(Hb_func, data)
    assert result.rank == dyn.n_base
    err = parameter_error(result.parms, pi_b)
    assert err["rel_norm"] < 1e-9, result.summary()


def test_coulomb_only_model_is_biased_on_stribeck_data():
    """同一份含 Stribeck 峰的数据，不建模 fs 的回归会留下系统性残差。"""
    stribeck = _robot()
    plain = Robot.from_mdh(
        "rrr_plain", RRR_MDH_PARMS,
        frictionmodel={"viscous", "Coulomb"}, driveinertiamodel="simplified",
    )
    for robot in (stribeck, plain):
        robot.dynamics.gen_regressor()
        robot.dynamics.calc_base_parms()
        robot.dynamics.gen_base_regressor()

    rng = np.random.default_rng(11)
    m, l, Le = _rigid_truth(rng)
    extras = _friction_truth(rng)
    nd = stribeck.dynamics.numeric_dynamics()
    traj, _ = search_excitation(stribeck.dynamics.Hb_func, 3,
                                np.random.default_rng(2), trials=10, vel_scale=1.0)
    data = sample_trajectory(
        traj, lambda q, dq, ddq: nd.invdyn(q, dq, ddq, m, l, Le, **extras), rate=100,
    )
    with_stribeck = identify(stribeck.dynamics.Hb_func, data)
    without = identify(plain.dynamics.Hb_func, data)
    assert with_stribeck.residual_rms < 1e-9
    assert without.residual_rms > 100 * max(with_stribeck.residual_rms, 1e-12)


def test_recovery_layout_and_bounds_accept_stribeck_extras():
    robot = _robot()
    robot.dynamics.gen_regressor()
    robot.dynamics.calc_base_parms()
    layout = _Layout(robot)
    assert layout.extra_kinds == ["Ia", "fv", "fc", "fs"] * 3
    config = RecoveryConfig(friction_upper=2.5)
    lower, upper = _bounds(layout, config)
    extra_lower = lower[10 * 3:]
    extra_upper = upper[10 * 3:]
    for j, kind in enumerate(layout.extra_kinds):
        assert extra_lower[j] == 0.0
        assert extra_upper[j] == (config.drive_inertia_upper if kind == "Ia" else 2.5)

    rng = np.random.default_rng(13)
    m, l, Le = _rigid_truth(rng)
    extras = _friction_truth(rng)
    packed = pack_inertial_parameters(
        robot, m, [l[i] / m[i] for i in range(3)],
        [np.diag([0.02, 0.03, 0.04])] * 3,
        extras={f"fs_{i + 1}": extras["fs"][i] for i in range(3)},
    )
    named = dict(zip(map(str, robot.symbols.dynparms()), packed))
    for i in range(3):
        assert named[f"fs_{i + 1}"] == extras["fs"][i]
        assert named[f"fv_{i + 1}"] == 0.0
