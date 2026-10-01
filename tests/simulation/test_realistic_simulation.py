"""实机近似的因果性、执行器约束、真值隔离与闭环回归。"""
from dataclasses import replace
import json
import numpy as np
import pytest
from robot_model.data import MotionData, align_measurements
from robot_model.excitation import FourierTrajectory
from robot_model.simulation import (
    ActuatorConfig, OnlineSensorConfig, RealismConfig, RealisticMujocoSource,
)
from robot_model.simulation.hardware import DelayLine, OnlineSensors, TorqueActuator
from robot_model.experiments.rrr import default_config, run_experiment
from robot_model.robots.rrr import rrr_xml_path


def clean_sensors(**kwargs):
    return replace(OnlineSensorConfig(
        encoder_bits=None, position_noise_std=0, position_bias=0,
        encoder_delay_steps=0, torque_noise_std=0, torque_gain=1,
        torque_bias=0, torque_deadband=0, torque_delay_steps=0,
    ), **kwargs)


def test_delay_is_causal_and_has_exact_step_count():
    line = DelayLine(2, [0])
    assert [line.push([i]).item() for i in (1, 2, 3, 4)] == [0, 0, 1, 2]


def test_online_velocity_uses_past_encoder_and_channels_delay_independently():
    cfg = clean_sensors(encoder_delay_steps=1, torque_delay_steps=2)
    sensors = OnlineSensors(cfg, 1, .004, [0], np.random.default_rng(0))
    q, dq, tau = [], [], []
    for k in range(5):
        a, b = sensors.feedback([.004*k])
        q.append(a.item())
        dq.append(b.item())
        tau.append(sensors.torque(np.array([k+1])).item())
    assert np.allclose(q, [0, 0, .004, .008, .012])
    assert tau == [0, 0, 1, 2, 3]
    assert dq[0] == dq[1] == 0
    assert 0 < dq[2] < dq[3] < dq[4] < 1


def test_motor_delay_lag_limits_and_rate_are_physical():
    motor = TorqueActuator(ActuatorConfig(torque_limit=2, time_constant_s=.01,
                                          slew_rate=50, gain=1, delay_steps=2), 1, .001)
    out = np.array([motor.step([20]).item() for _ in range(200)])
    assert np.all(out[:2] == 0)
    assert 0 < out[2] <= .05
    assert np.max(np.abs(np.diff(out))) <= .05 + 1e-12
    assert out[-1] <= 2 and out[-1] > 1.99
    assert motor.saturated.item()
    reverse = np.array([motor.step([-20]).item() for _ in range(200)])
    assert np.max(np.abs(np.diff(reverse))) <= .05 + 1e-12
    assert reverse[-1] >= -2 and reverse[-1] < -1.99


def test_first_order_response_matches_analytic_solution():
    motor = TorqueActuator(ActuatorConfig(torque_limit=10, time_constant_s=.02,
                                          slew_rate=1e6, gain=1, delay_steps=0), 1, .001)
    out = np.array([motor.step([3]).item() for _ in range(100)])
    expected = 3 * (1-np.exp(-np.arange(1, 101)*.001/.02))
    assert np.allclose(out, expected, atol=1e-12)


def test_continuous_trajectory_bounds_hold_between_coarse_samples():
    traj = FourierTrajectory.random(3, np.random.default_rng(5), vel_scale=20)
    lo, hi = [-1, -.5, -2], [1, .5, 2]
    traj = traj.constrained(lower=lo, upper=hi, velocity=[.3, .5, .7], acceleration=1.2)
    q, dq, ddq = traj.evaluate(np.linspace(0, traj.period, 10001))
    assert np.all(q >= lo) and np.all(q <= hi)
    assert np.all(np.max(np.abs(dq), axis=0) <= [.3, .5, .7])
    assert np.max(np.abs(ddq)) <= 1.2


def test_delay_compensation_uses_measured_data_only():
    t = np.arange(20)*.01
    q = (np.arange(20)-2).reshape(-1, 1)
    tau = (np.arange(20)-4).reshape(-1, 1)
    data = MotionData(t, q, q, q, tau)
    aligned = align_measurements(data, encoder_delay_steps=2, torque_delay_steps=4)
    assert np.array_equal(aligned.q, aligned.tau)
    assert np.array_equal(aligned.q.ravel(), np.arange(16))
    assert np.array_equal(aligned.t, t[:16])


@pytest.fixture
def model():
    mj = pytest.importorskip('mujoco')
    return mj.MjModel.from_xml_path(str(rrr_xml_path()))


@pytest.fixture
def short_trajectory():
    return FourierTrajectory.random(3, np.random.default_rng(10), base_freq=2,
                                    n_harmonics=2, vel_scale=.15)


def test_plant_is_separate_and_feedforward_ignores_true_parameters(model):
    mass = model.body_mass.copy()
    source = RealisticMujocoSource(model, default_config())
    assert np.array_equal(model.body_mass, mass)
    assert not np.allclose(source.model.body_mass, mass)
    args = ([.2, -.4, .3], [.1, .2, -.1], [1., -.2, .3])
    ff = source.feedforward(*args)
    source.model.body_mass[:] *= 2
    assert np.array_equal(source.feedforward(*args), ff)


def test_runs_reset_clock_noise_delays_and_motor_state(model, short_trajectory):
    source = RealisticMujocoSource(model, default_config())
    a = source.run(short_trajectory, warmup_periods=0, seed=42)
    b = source.run(short_trajectory, warmup_periods=0, seed=42)
    for field in ('t', 'q', 'dq', 'ddq', 'tau'):
        assert np.array_equal(getattr(a.measured, field), getattr(b.measured, field))
    assert a.measured.t[0] == 0
    assert np.allclose(np.diff(a.measured.t), .004)
    assert np.allclose(a.truth.q[0], short_trajectory.evaluate(0)[0])
    assert np.all(a.truth.dq[0] == 0)
    c = source.run(short_trajectory, warmup_periods=0, seed=43)
    assert not np.array_equal(a.truth.q, c.truth.q)


def test_feedback_bias_changes_actual_motion(model, short_trajectory):
    config = replace(default_config(), sensors=clean_sensors())
    normal = RealisticMujocoSource(model, config).run(short_trajectory, warmup_periods=0)
    biased = RealisticMujocoSource(model, replace(config, sensors=clean_sensors(position_bias=.02))).run(
        short_trajectory, warmup_periods=0)
    assert not np.allclose(normal.command, biased.command)
    assert np.max(np.abs(normal.truth.q-biased.truth.q)) > .001


def test_recorded_torque_is_applied_not_command_and_truth_is_time_aligned(model, short_trajectory):
    mj = pytest.importorskip('mujoco')
    cfg = replace(default_config(), sensors=clean_sensors(), low_speed_friction=0,
                  disturbance_amplitude=0, frictionloss=0)
    source = RealisticMujocoSource(model, cfg)
    run = source.run(short_trajectory, warmup_periods=0)
    assert np.array_equal(run.measured.tau, run.truth.tau)
    assert not np.allclose(run.command, run.truth.tau)
    assert np.all(np.abs(run.truth.tau) <= np.asarray(cfg.actuator.torque_limit)+1e-12)
    data = mj.MjData(source.model)
    for i in (0, 10, 50):
        data.qpos[:] = run.truth.q[i]
        data.qvel[:] = run.truth.dq[i]
        data.qacc[:] = run.truth.ddq[i]
        mj.mj_inverse(source.model, data)
        assert np.allclose(data.qfrc_inverse, run.truth.tau[i], atol=1e-8)


@pytest.mark.parametrize('change', [
    {'control_dt': .0015}, {'physics_dt': -1}, {'mass_relative_error': 1.1},
    {'joint_lower': [-1]*3}, {'damping': -1}, {'kp': [1, 2]},
])
def test_bad_configs_fail_before_simulation(model, change):
    with pytest.raises(ValueError):
        RealisticMujocoSource(model, replace(RealismConfig(), **change))


def test_unsupported_transmission_fails_explicitly(model):
    model.actuator_gear[0, 0] = 100
    with pytest.raises(ValueError, match='gear=1'):
        RealisticMujocoSource(model)


def test_config_roundtrip():
    config = default_config(17)
    reloaded = RealismConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert json.dumps(config.to_dict(), sort_keys=True) == json.dumps(reloaded.to_dict(), sort_keys=True)


def test_independent_validation_improves_over_nominal_and_saves_reproducible_data(tmp_path):
    pytest.importorskip('mujoco')
    pytest.importorskip('scipy')
    report = run_experiment(default_config(), trials=4, periods=1,
                            optimize=False, output_dir=tmp_path)
    assert report['fit']['identifiable']
    scores = report['validation_rms_Nm']
    assert np.linalg.norm(scores['truth_fitted']) < np.linalg.norm(scores['truth_nominal'])
    assert np.max(report['validation']['tracking_rms_rad']) < .1
    assert not any(report['validation']['joint_limit_fraction'])
    train = np.load(tmp_path / 'train.npz')
    validation = np.load(tmp_path / 'validation.npz')
    assert 'truth_tau' in train and 'tau' in train
    assert not np.array_equal(train['desired_q'], validation['desired_q'])
    saved = json.loads((tmp_path / 'report.json').read_text())
    assert saved['config']['seed'] == 0
