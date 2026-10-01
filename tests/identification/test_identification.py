"""辨识基础件：轨迹的解析导数、最小二乘求解器。不依赖仿真器。"""

from __future__ import annotations

import numpy as np
import pytest

from robot_model.data import (
    MotionData,
    add_noise,
    differentiate,
    drop_low_speed,
    estimate_derivatives,
    filter_measurements,
    sample_trajectory,
    zero_phase_lowpass,
)
from robot_model.excitation import FourierTrajectory
from robot_model.identification import (
    identify,
    parameter_error,
    predict_torque,
    stack_regressor,
)
from robot_model.simulation import SensorModel


@pytest.fixture
def traj() -> FourierTrajectory:
    return FourierTrajectory.random(
        3, np.random.default_rng(0), n_harmonics=4, base_freq=0.2
    )


def test_velocity_and_acceleration_are_exact_derivatives(traj):
    """dq/ddq 是解析导数，写错了这里就会暴露（中心差分比对）。"""
    t0, h = 1.234, 1e-6
    _, dq, ddq = traj.evaluate(t0)
    q_plus, dq_plus, _ = traj.evaluate(t0 + h)
    q_minus, dq_minus, _ = traj.evaluate(t0 - h)
    assert np.allclose(dq, (q_plus - q_minus) / (2 * h), atol=1e-7)
    assert np.allclose(ddq, (dq_plus - dq_minus) / (2 * h), atol=1e-7)


def test_trajectory_is_periodic(traj):
    q0, dq0, ddq0 = traj.evaluate(0.3)
    q1, dq1, ddq1 = traj.evaluate(0.3 + traj.period)
    assert np.allclose(q0, q1)
    assert np.allclose(dq0, dq1)
    assert np.allclose(ddq0, ddq1)


def test_timestamps_cover_one_period_without_duplicate_endpoint(traj):
    t = traj.timestamps(periods=2, rate=50.0)
    assert t.size == int(round(2 * traj.period * 50.0))
    assert t[0] == 0.0
    assert t[-1] < 2 * traj.period


def test_scaling_leaves_offset_untouched(traj):
    scaled = traj.scaled([0.5, 0.5, 0.5])
    assert np.allclose(scaled.q0, traj.q0)
    _, dq, _ = traj.evaluate(0.7)
    _, dq_s, _ = scaled.evaluate(0.7)
    assert np.allclose(dq_s, 0.5 * dq)


def _linear_problem(n_samples=40, dof=3, n_parms=5, seed=0):
    """构造一个已知参数的线性问题：tau = Hb(q,dq,ddq) pi。"""
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(dof, n_parms, 3 * dof))

    def regressor(q, dq, ddq):
        x = np.concatenate([q, dq, ddq])
        return basis @ x

    q = rng.normal(size=(n_samples, dof))
    dq = rng.normal(size=(n_samples, dof))
    ddq = rng.normal(size=(n_samples, dof))
    parms = rng.normal(size=n_parms)
    tau = np.array(
        [regressor(q[i], dq[i], ddq[i]) @ parms for i in range(n_samples)]
    )
    data = MotionData(t=np.arange(n_samples) * 0.01, q=q, dq=dq, ddq=ddq, tau=tau)
    return regressor, data, parms


def test_identify_recovers_parameters_exactly_without_noise():
    regressor, data, parms = _linear_problem()
    result = identify(regressor, data)
    assert np.allclose(result.parms, parms, atol=1e-10)
    assert result.residual_rms < 1e-12
    assert result.rank == parms.size


def test_stacked_regressor_shape():
    regressor, data, parms = _linear_problem()
    W, y = stack_regressor(regressor, data)
    assert W.shape == (data.n_samples * data.dof, parms.size)
    assert y.size == data.n_samples * data.dof


def test_predict_torque_reproduces_observations():
    regressor, data, parms = _linear_problem()
    assert np.allclose(predict_torque(regressor, data, parms), data.tau)


def test_noise_degrades_estimate_gracefully():
    regressor, data, parms = _linear_problem(n_samples=400)
    rng = np.random.default_rng(1)
    errors = []
    for std in (0.0, 0.01, 0.1):
        noisy = add_noise(data, rng, tau_std=std)
        est = identify(regressor, noisy).parms
        errors.append(parameter_error(est, parms)["rel_norm"])
    assert errors[0] < 1e-12
    assert errors[0] < errors[1] < errors[2]  # 误差随噪声单调上升


def test_weights_must_match_dof():
    regressor, data, _ = _linear_problem()
    with pytest.raises(ValueError):
        identify(regressor, data, weights=[1.0, 1.0])


def test_parameter_error_rejects_mismatched_length():
    with pytest.raises(ValueError):
        parameter_error(np.zeros(3), np.zeros(4))


def test_sample_trajectory_uses_supplied_torque_source(traj):
    def tau_func(q, dq, ddq):
        return q + dq + ddq

    data = sample_trajectory(traj, tau_func, rate=20.0)
    assert data.tau.shape == data.q.shape
    assert np.allclose(data.tau, data.q + data.dq + data.ddq)


def test_drop_low_speed_keeps_only_fast_samples():
    dq = np.array([[1.0, 1.0], [0.01, 1.0], [1.0, 1.0], [0.5, 0.001]])
    n = dq.shape[0]
    data = MotionData(
        t=np.arange(n) * 0.01,
        q=np.zeros((n, 2)),
        dq=dq,
        ddq=np.zeros((n, 2)),
        tau=np.zeros((n, 2)),
    )
    kept = drop_low_speed(data, 0.05)
    assert len(kept) == 2  # 只有第 0、2 行两个关节都够快
    assert np.allclose(kept.dq, 1.0)


def test_drop_low_speed_rejects_too_strict_threshold():
    n = 3
    data = MotionData(
        t=np.arange(n) * 0.01,
        q=np.zeros((n, 2)),
        dq=np.full((n, 2), 0.1),
        ddq=np.zeros((n, 2)),
        tau=np.zeros((n, 2)),
    )
    assert len(drop_low_speed(data, 0.0)) == n  # 阈值 0 视为不筛
    with pytest.raises(ValueError):
        drop_low_speed(data, 10.0)


class TestFiltering:
    """滤波与微分。scipy 缺席时整体跳过。"""

    @pytest.fixture(autouse=True)
    def _need_scipy(self):
        pytest.importorskip("scipy")

    @staticmethod
    def _signal(fs=200.0, seconds=4.0, freq=0.5):
        t = np.arange(int(fs * seconds)) / fs
        x = np.sin(2 * np.pi * freq * t)
        return t, x.reshape(-1, 1), freq

    def test_lowpass_preserves_inband_signal_without_phase_shift(self):
        """零相位是关键：有相移的话 q、dq、tau 会在时间上错位。"""
        t, x, _ = self._signal()
        y = zero_phase_lowpass(x, fs=200.0, cutoff=20.0)
        edge = 40  # 边界瞬态不比
        assert np.allclose(y[edge:-edge], x[edge:-edge], atol=1e-3)

    def test_lowpass_attenuates_noise(self):
        t, x, _ = self._signal()
        rng = np.random.default_rng(0)
        noisy = x + rng.normal(0, 0.1, size=x.shape)
        y = zero_phase_lowpass(noisy, fs=200.0, cutoff=5.0)
        edge = 40
        before = np.std((noisy - x)[edge:-edge])
        after = np.std((y - x)[edge:-edge])
        assert after < before / 5

    def test_cutoff_must_be_below_nyquist(self):
        _, x, _ = self._signal()
        with pytest.raises(ValueError):
            zero_phase_lowpass(x, fs=200.0, cutoff=150.0)

    def test_differentiate_matches_analytic_derivative(self):
        t, x, freq = self._signal()
        dt = t[1] - t[0]
        expected = (2 * np.pi * freq) * np.cos(2 * np.pi * freq * t)
        got = differentiate(x, dt).ravel()
        assert np.allclose(got[2:-2], expected[2:-2], atol=1e-3)

    def test_filtering_before_differentiating_beats_raw(self):
        """带噪声时先滤波再微分应当明显更准，这是整条预处理链的理由。"""
        t, x, freq = self._signal()
        dt = t[1] - t[0]
        rng = np.random.default_rng(1)
        noisy = x + rng.normal(0, 1e-3, size=x.shape)
        truth = -((2 * np.pi * freq) ** 2) * np.sin(2 * np.pi * freq * t)

        edge = 60
        raw = estimate_derivatives(
            differentiate(noisy, dt), dt=dt, cutoff=None
        ).ravel()
        filtered = estimate_derivatives(
            differentiate(noisy, dt), dt=dt, cutoff=5.0
        ).ravel()
        err_raw = np.std((raw - truth)[edge:-edge])
        err_filtered = np.std((filtered - truth)[edge:-edge])
        assert err_filtered < err_raw / 10

    def test_filter_measurements_recomputes_acceleration(self):
        t, x, _ = self._signal()
        n = t.size
        data = MotionData(
            t=t,
            q=np.tile(x, (1, 2)),
            dq=np.tile(x, (1, 2)),
            ddq=np.zeros((n, 2)),
            tau=np.zeros((n, 2)),
        )
        out = filter_measurements(data, cutoff=20.0)
        assert not np.allclose(out.ddq, 0.0)  # 由 dq 重算，不再是零
        assert np.allclose(out.t, data.t)


class TestSensorModel:
    """测量失真的纯逻辑部分。"""

    @staticmethod
    def _data(n=50, dof=2):
        t = np.arange(n) * 0.01
        ramp = np.linspace(0.0, 1.0, n).reshape(-1, 1)
        block = np.tile(ramp, (1, dof))
        return MotionData(t=t, q=block, dq=block, ddq=block, tau=block)

    def test_default_model_is_transparent(self):
        data = self._data()
        out = SensorModel().apply(data)
        assert np.allclose(out.q, data.q)
        assert np.allclose(out.tau, data.tau)
        assert len(out) == len(data)

    def test_quantization_snaps_to_lsb(self):
        model = SensorModel(encoder_bits=10)  # span=2pi -> lsb≈6.1e-3
        lsb = model.encoder_lsb
        q = np.array([[0.0, lsb * 1.4, -lsb * 2.6]])
        got = model.quantize(q)
        assert np.allclose(got / lsb, np.round(q / lsb))
        assert np.max(np.abs(got - q)) <= lsb / 2 + 1e-15

    def test_higher_resolution_means_smaller_error(self):
        q = np.linspace(-1.0, 1.0, 200).reshape(-1, 1)
        coarse = SensorModel(encoder_bits=8).quantize(q)
        fine = SensorModel(encoder_bits=16).quantize(q)
        assert np.max(np.abs(fine - q)) < np.max(np.abs(coarse - q))

    def test_torque_gain_scales_measurement(self):
        model = SensorModel(torque_gain=1.02)
        assert np.allclose(model.measure_torque(np.array([1.0, -2.0])), [1.02, -2.04])

    def test_torque_deadband_zeroes_small_values(self):
        model = SensorModel(torque_deadband=0.1)
        got = model.measure_torque(np.array([0.05, -0.05, 0.5]))
        assert np.allclose(got, [0.0, 0.0, 0.5])

    def test_delay_shifts_and_trims(self):
        """延迟后样本数减少，且测量值确实取自更早时刻。"""
        data = self._data(n=20)
        out = SensorModel(encoder_delay=3).apply(data)
        assert len(out) == len(data) - 3
        assert np.allclose(out.t, data.t[3:])
        assert np.allclose(out.q[0], data.q[0])  # t=3 处读到 t=0 的值

    def test_channels_shift_independently(self):
        """分通道延迟是这个模型存在的意义：同步时移无害，错位才致命。"""
        data = self._data(n=20)
        out = SensorModel(encoder_delay=1, torque_delay=4).apply(data)
        assert len(out) == len(data) - 4
        assert np.allclose(out.q[0], data.q[3])  # 编码器只滞后 1 步
        assert np.allclose(out.tau[0], data.tau[0])  # 力矩滞后 4 步

    def test_velocity_from_encoder_uses_differentiated_position(self):
        data = self._data(n=60)
        out = SensorModel(
            encoder_bits=16, velocity_from_encoder=True
        ).apply(data)
        # q 是斜坡，差分后应接近常数斜率，而非原样复制 dq
        slope = (data.q[-1, 0] - data.q[0, 0]) / (data.t[-1] - data.t[0])
        assert np.allclose(out.dq[5:-5], slope, atol=1e-2)


def test_subset_selects_rows(traj):
    data = sample_trajectory(traj, lambda q, dq, ddq: q, rate=20.0)
    half = data.subset(np.arange(0, len(data), 2))
    assert len(half) == (len(data) + 1) // 2
    assert np.allclose(half.q[1], data.q[2])
