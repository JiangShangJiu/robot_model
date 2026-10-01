"""约束优化、多场景最差评分与非整数周期平均的行为测试。"""
from types import SimpleNamespace
import json
import numpy as np
import pytest

from robot_model.data import MotionData, average_periods
from robot_model.excitation import (
    ExcitationCriteria, FourierTrajectory, NoFeasibleExcitation,
    optimize_excitation, screen_robust_excitation,
)


def periodic_data(period=.3713, dt=.002, cycles=8, noise=.03, seed=0):
    t = .013+np.arange(int(np.floor(cycles*period/dt)))*dt
    clean = np.sin(2*np.pi*t/period)[:, None]
    rng = np.random.default_rng(seed)
    x = clean+rng.normal(0, noise, clean.shape)
    return MotionData(t, x, x, x, x), clean


def test_average_noninteger_period_reduces_noise_without_phase_shift():
    data, _ = periodic_data(cycles=20)
    average = average_periods(data, .3713, cycles=20)
    expected = np.sin(2*np.pi*average.data.t/.3713)[:, None]
    error = np.sqrt(np.mean((average.data.q-expected)**2))
    assert error < .03/3
    assert average.cycles == 20
    assert np.max(average.data.t) < data.t[0]+.3713
    assert average.data.t[-1]+19*.3713 <= data.t[-1]+1e-10
    assert np.all(average.standard_deviation['q'] > 0)
    assert average.summary()['cycles'] == 20


def test_average_preserves_bias_and_reports_cycle_scatter():
    data, _ = periodic_data(noise=0)
    shifted = MotionData(data.t, data.q+2, data.dq, data.ddq, data.tau+2)
    a = average_periods(data, .3713, cycles=8)
    b = average_periods(shifted, .3713, cycles=8)
    assert np.allclose(b.data.q-a.data.q, 2)
    assert np.allclose(b.data.tau-a.data.tau, 2)


def test_average_rejects_missing_cycles_and_nonuniform_samples():
    data, _ = periodic_data(cycles=2)
    with pytest.raises(ValueError):
        average_periods(data, .3713, cycles=3)
    with pytest.raises(ValueError):
        average_periods(data, .3713, cycles=1)
    with pytest.raises(ValueError, match='连续'):
        average_periods(data.subset(np.r_[0:10, 15:len(data)]), .3713, cycles=2)
    inferred = average_periods(data, .3713)
    assert inferred.cycles == 2


def regressor(q, dq, ddq):
    return np.array([[q[0], dq[0]]])


def test_slsqp_improves_condition_and_holds_continuous_bounds():
    pytest.importorskip('scipy')
    # 两个参数列幅值不平衡，调整二阶谐波可改善。
    start = FourierTrajectory(np.zeros(1), np.array([[.3, .01]]), np.array([[.15, .01]]), .1)
    result = optimize_excitation(regressor, start, limits=([-1], [1]),
                                 velocity_limits=1., acceleration_limits=2., maxiter=35)
    assert result.report['adopted']
    assert result.report['after']['condition_number'] < result.report['before']['condition_number']*.9
    q, dq, ddq = result.trajectory.evaluate(np.linspace(0, 10, 20001))
    assert np.max(np.abs(q)) <= .9+1e-8
    assert np.max(np.abs(dq)) <= .9+1e-8
    assert np.max(np.abs(ddq)) <= 1.8+1e-8
    assert result.report['constraint_min_slack'] >= -1e-9
    assert result.trajectory.base_freq == start.base_freq
    assert np.array_equal(start.a, [[.3,.01]])
    json.dumps(result.report, allow_nan=False)


def test_optimizer_never_adopts_worse_dense_grid_result():
    pytest.importorskip('scipy')
    start = FourierTrajectory(np.zeros(1), np.array([[.1]]), np.array([[.1]]), .2)
    result = optimize_excitation(regressor, start, limits=([-1],[1]),
                                 velocity_limits=1., acceleration_limits=3., maxiter=1)
    assert result.report['after']['condition_number'] <= result.report['before']['condition_number']
    assert isinstance(result.report['converged'], bool)


def fake_run(scale=1., exceeds=False):
    rng = np.random.default_rng(2)
    q = rng.normal(size=(100,1))*scale
    dq = rng.normal(size=(100,1))
    data = MotionData(np.arange(100)*.01, q, dq, q*0, q*0)
    d = dict(tracking_rms_rad=[.01], tracking_max_rad=[.02], saturation_fraction=[0.],
             slew_limit_fraction=[0.], joint_limit_fraction=[0.], contact_fraction=0.,
             max_speed_rad_s=[.1], max_acceleration_rad_s2=[.1],
             speed_limit_exceeded=[False], acceleration_limit_exceeded=[exceeds])
    return SimpleNamespace(measured=data, diagnostics=d)


def test_robust_ranks_worst_case_and_preserves_primary_object_data():
    candidates = [FourierTrajectory(np.zeros(1), np.ones((1,1)), np.ones((1,1))*.2, f) for f in (.2,.3)]
    primary = {}
    def first(t):
        primary[t.base_freq] = fake_run(1 if t.base_freq == .2 else 3)
        return primary[t.base_freq]
    def second(t):
        return fake_run(50 if t.base_freq == .2 else 4)
    selection = screen_robust_excitation(regressor, candidates, [('a', first), ('b', second)],
                  lambda d,t:d, criteria=ExcitationCriteria(min_samples=20))
    assert selection.report['selected_candidate_id'] == 1
    assert selection.run is primary[.3]
    records = selection.report['candidates']
    assert records[0]['measured']['condition_number'] > records[1]['measured']['condition_number']


def test_one_failed_scenario_rejects_candidate_without_averaging_it_away():
    traj = FourierTrajectory(np.zeros(1), np.ones((1,1)), np.ones((1,1))*.2, .2)
    with pytest.raises(NoFeasibleExcitation) as caught:
        screen_robust_excitation(regressor, [traj],
            [('a',lambda t:fake_run()), ('b',lambda t:fake_run(exceeds=True))],
            lambda d,t:d, criteria=ExcitationCriteria(min_samples=20))
    assert caught.value.report['candidates'][0]['passed_scenarios'] == 1
    assert 'b:' in caught.value.report['candidates'][0]['reasons'][0]


def test_period_averaging_does_not_count_repeats_as_dropped_samples():
    from robot_model.excitation import screen_excitation
    traj = FourierTrajectory(np.zeros(1), np.ones((1,1)), np.ones((1,1))*.2, .2)
    run = fake_run()
    selected = screen_excitation(regressor, [traj], lambda t:run,
        lambda d:d.subset(slice(0,20)), criteria=ExcitationCriteria(min_samples=15), retention_multiplier=4)
    assert selected.report['candidates'][0]['retained_fraction'] == .8
    assert selected.report['candidates'][0]['measured']['samples'] == 20


def test_enhanced_pipeline_saves_primary_fit_and_scenario_audit(tmp_path):
    pytest.importorskip('mujoco')
    from robot_model.experiments.rrr import default_config, run_experiment
    report = run_experiment(default_config(), trials=2, periods=2, optimize=True,
        optimization_starts=1, optimization_maxiter=2, robust_seeds=[0, 1],
        periodic_average=True, output_dir=tmp_path)
    screening = report['screening']
    chosen = screening['candidates'][screening['selected_candidate_id']]
    assert chosen['accepted'] and chosen['passed_scenarios'] == 2
    assert report['fit']['rank'] == report['fit']['n_parms']
    assert report['period_average']['train']['cycles'] == 2
    assert len(report['optimization']) == 1
    with np.load(tmp_path/'train-processed.npz') as processed:
        assert len(processed['t']) == report['fit']['samples_after_preprocessing']
        assert len(processed['t']) == chosen['scenarios'][0]['measured']['samples']
    with np.load(tmp_path/'train-averaged.npz') as averaged:
        assert averaged['std_tau'].shape == averaged['tau'].shape
    saved = json.loads((tmp_path/'report.json').read_text())
    assert saved['enhancements']['robust_seeds'] == [0, 1]
    assert json.loads((tmp_path/'optimization.json').read_text()) == report['optimization']
