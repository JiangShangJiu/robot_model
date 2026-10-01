"""Franka 闭环实机近似：模型前提、配置自洽与端到端辨识。

力矩 motor 模型与 position servo 模型必须区分开：后者不能用于实机近似闭环。
"""
import numpy as np
import pytest

from robot_model.experiments.franka import default_config, setup
from robot_model.robots import (
    PANDA_JOINT_LOWER,
    PANDA_JOINT_UPPER,
    PANDA_MDH_BODY_NAMES,
    panda_motor_xml_path,
    panda_xml_path,
)


@pytest.fixture
def models():
    mujoco = pytest.importorskip("mujoco")
    return (mujoco.MjModel.from_xml_path(str(panda_motor_xml_path())),
            mujoco.MjModel.from_xml_path(str(panda_xml_path())))


def test_motor_model_matches_servo_model_geometry_and_inertia(models):
    """两个无夹爪模型只应在执行器上不同；惯性或几何不同就是另一台机器人。"""
    motor, servo = models
    assert motor.nbody == servo.nbody == len(PANDA_MDH_BODY_NAMES) + 2
    for field in ("body_mass", "body_ipos", "body_inertia", "body_pos", "body_quat",
                  "jnt_pos", "jnt_axis", "jnt_range"):
        assert np.allclose(getattr(motor, field), getattr(servo, field)), field


def test_realistic_source_requires_torque_motors(models):
    from robot_model.simulation.realistic import _validate_motor_model

    motor, servo = models
    _validate_motor_model(motor)
    with pytest.raises(ValueError, match="力矩 motor"):
        _validate_motor_model(servo)


def test_joint_limit_constants_track_the_mjcf(models):
    """限位常量是手写的，必须与 MJCF 一致。"""
    motor, _ = models
    assert np.allclose(motor.jnt_range[:, 0], PANDA_JOINT_LOWER)
    assert np.allclose(motor.jnt_range[:, 1], PANDA_JOINT_UPPER)


def test_zero_is_not_a_usable_trajectory_center():
    """关节 4 的区间完全在负半轴，关节 6 的中点在 1.87；轨迹中心必须由限位算。"""
    assert PANDA_JOINT_UPPER[3] < 0
    midpoints = (np.array(PANDA_JOINT_LOWER) + np.array(PANDA_JOINT_UPPER)) / 2
    assert abs(midpoints[3]) > 1.0 and abs(midpoints[5]) > 1.0


def test_default_config_is_consistent_with_setup():
    config, description = default_config(), setup()
    assert description.dof == 7
    assert config.joint_lower == PANDA_JOINT_LOWER
    assert config.joint_upper == PANDA_JOINT_UPPER
    for name in ("kp", "kd", "velocity_limit", "acceleration_limit"):
        assert len(getattr(config, name)) == 7, name
    assert len(config.actuator.torque_limit) == 7
    # 展示实验固定使用随机搜索，提速后也不自动改变激励设计策略
    assert description.search_refine is False


def test_tuned_gains_track_without_saturation_or_contact():
    """阶段 2 的调参结论：规格级包络下零饱和、零接触、跟踪 RMS 远低于门槛。"""
    pytest.importorskip("mujoco")
    import mujoco

    from robot_model.excitation import generate_excitation_candidates
    from robot_model.simulation import RealisticMujocoSource

    config, description = default_config(), setup()
    model = mujoco.MjModel.from_xml_path(str(panda_motor_xml_path()))
    trajectory = generate_excitation_candidates(
        7, np.random.default_rng(11), trials=1,
        limits=(np.array(PANDA_JOINT_LOWER), np.array(PANDA_JOINT_UPPER)),
        velocity_limits=config.velocity_limit,
        acceleration_limits=config.acceleration_limit,
        vel_scale=description.candidate_vel_scale,
    )[0]
    diagnostics = RealisticMujocoSource(model, config).run(
        trajectory, periods=1, seed=303).diagnostics
    assert max(diagnostics["tracking_rms_rad"]) < 0.02
    assert max(diagnostics["saturation_fraction"]) == 0
    assert diagnostics["contact_fraction"] == 0
    assert max(diagnostics["joint_limit_fraction"]) == 0
    assert not any(diagnostics["speed_limit_exceeded"])
    assert not any(diagnostics["acceleration_limit_exceeded"])


@pytest.mark.slow
def test_identification_beats_nominal_on_independent_trajectory(tmp_path):
    """端到端：62 个基参数满秩，辨识模型在留出轨迹上优于名义模型。"""
    pytest.importorskip("mujoco")
    pytest.importorskip("scipy")
    from robot_model.experiments.franka import run_experiment

    report = run_experiment(default_config(), trials=4, periods=1,
                            screen=True, optimize=False, output_dir=tmp_path)
    assert report["status"] == "completed"
    assert report["robot"] == "panda_realistic"
    assert report["experiment"]["closed_loop_screening"]
    assert report["screening"]["accepted_count"] > 0
    assert report["screening"]["selected_candidate_id"] is not None
    assert report["fit"]["rank"] == 62
    assert report["fit"]["n_parms"] == 62
    assert report["fit"]["identifiable"]
    fitted = report["validation_rms_Nm"]["measured_fitted"]
    nominal = report["validation_rms_Nm"]["measured_nominal"]
    assert all(f < n for f, n in zip(fitted, nominal))
    for phase in ("train", "validation"):
        assert max(report[phase]["saturation_fraction"]) == 0
        assert report[phase]["contact_fraction"] == 0
        assert max(report[phase]["joint_limit_fraction"]) == 0
