"""筛选必须依据可执行性与激励信息，不能被低残差、欠定 SVD 或噪声误导。"""
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from robot_model.data import MotionData
from robot_model.excitation import (
    ExcitationCriteria, FourierTrajectory, NoFeasibleExcitation,
    generate_excitation_candidates, regressor_quality, screen_excitation,
)


def regressor(q, dq, ddq):
    return np.array([[q[0], dq[0]]])


def trajectory(freq=.2):
    return FourierTrajectory(np.zeros(1), np.array([[1.]]), np.array([[.3]]), freq)


def data(scale=1., count=100):
    rng = np.random.default_rng(0)
    q = scale*rng.normal(size=(count, 1))
    dq = rng.normal(size=(count, 1))
    return MotionData(np.arange(count)*.01, q, dq, np.zeros_like(q), np.ones_like(q))


def run(measured=None, **changes):
    diagnostics = dict(
        tracking_rms_rad=[.01], tracking_max_rad=[.02], saturation_fraction=[0.],
        slew_limit_fraction=[0.], joint_limit_fraction=[0.], contact_fraction=0.,
        max_speed_rad_s=[.1], max_acceleration_rad_s2=[.1],
        speed_limit_exceeded=[False], acceleration_limit_exceeded=[False],
    )
    diagnostics.update(changes)
    return SimpleNamespace(measured=data() if measured is None else measured,
                           diagnostics=diagnostics)


CRITERIA = ExcitationCriteria(min_samples=20)


def select(candidates, runner, preprocess=lambda x: x, criteria=CRITERIA):
    return screen_excitation(regressor, candidates, runner, preprocess, criteria=criteria)


def test_best_measured_condition_wins_and_selected_run_is_reused():
    runs = [run(data(20)), run(data(1))]
    calls = []
    def runner(traj):
        calls.append(traj)
        return runs[len(calls)-1]
    result = select([trajectory(), trajectory(.3)], runner)
    assert len(calls) == 2
    assert result.report['selected_candidate_id'] == 1
    assert result.run is runs[1]
    assert result.data is runs[1].measured
    assert result.report['accepted_count'] == 2
    json.dumps(result.report, allow_nan=False)


@pytest.mark.parametrize('change,reason', [
    ({'tracking_rms_rad': [.2]}, '跟踪 RMS'),
    ({'tracking_max_rad': [.3]}, '最大跟踪'),
    ({'saturation_fraction': [.2]}, '饱和率'),
    ({'slew_limit_fraction': [.2]}, '变化率'),
    ({'joint_limit_fraction': [.001]}, '关节限位'),
    ({'contact_fraction': .001}, '接触'),
    ({'speed_limit_exceeded': [True]}, '实际速度'),
    ({'acceleration_limit_exceeded': [True]}, '实际加速度'),
    ({'tracking_rms_rad': [float('nan')]}, '非有限'),
])
def test_failed_dynamics_are_rejected_even_with_excellent_regressor(change, reason):
    with pytest.raises(NoFeasibleExcitation) as caught:
        select([trajectory()], lambda t: run(**change))
    report = caught.value.report
    assert report['selected_candidate_id'] is None
    assert any(reason in s for s in report['candidates'][0]['reasons'])
    json.dumps(report, allow_nan=False)


def test_underdetermined_matrix_does_not_report_condition_one():
    quality = regressor_quality(lambda q, dq, ddq: np.ones((1, 2)), data(count=1))
    assert quality['rank'] == 1
    assert quality['condition_number'] == np.inf
    assert quality['min_singular_value'] == 0


def test_analytical_rank_failure_is_rejected_before_running_simulation():
    calls = []
    with pytest.raises(NoFeasibleExcitation) as caught:
        screen_excitation(lambda q, dq, ddq: np.zeros((1, 2)), [trajectory()],
                          lambda t: calls.append(t), lambda d: d)
    assert not calls
    assert caught.value.report['candidates'][0]['preview']['condition_number'] is None


def test_rank_loss_after_preprocessing_is_rejected():
    base = data()
    deficient = MotionData(base.t, base.q, base.q, base.ddq, base.tau)
    with pytest.raises(NoFeasibleExcitation) as caught:
        select([trajectory()], lambda t: run(), lambda d: deficient)
    assert '实测回归矩阵秩不足' in caught.value.report['candidates'][0]['reasons']


def test_too_few_retained_samples_is_not_a_success():
    with pytest.raises(NoFeasibleExcitation) as caught:
        select([trajectory()], lambda t: run(), lambda d: d.subset(slice(0, 10)))
    reasons = caught.value.report['candidates'][0]['reasons']
    assert '有效采样点不足' in reasons
    assert '有效样本保留比例过低' in reasons


def test_weak_but_well_conditioned_excitation_is_rejected():
    base = data()
    weak = MotionData(base.t, base.q*1e-8, base.dq*1e-8, base.ddq, base.tau)
    with pytest.raises(NoFeasibleExcitation) as caught:
        select([trajectory()], lambda t: run(weak))
    record = caught.value.report['candidates'][0]
    assert record['measured']['condition_number'] < 2
    assert '实测激励强度不足' in record['reasons']


def test_minimum_singular_value_objective_and_torque_independence():
    base = data()
    strong = MotionData(base.t, base.q*2, base.dq*2, base.ddq, base.tau*1000)
    result = select([trajectory(), trajectory(.3)],
                    lambda t: run(base if t.base_freq == .2 else strong),
                    criteria=replace(CRITERIA, objective='min_singular_value'))
    assert result.report['selected_candidate_id'] == 1
    # 扭矩换成任何有限数都不改变回归矩阵评分；不按拟合残差挑训练集。
    assert regressor_quality(regressor, base) == regressor_quality(
        regressor, MotionData(base.t, base.q, base.dq, base.ddq, base.tau*999))


def test_simulation_failure_does_not_prevent_other_candidates():
    def runner(t):
        if t.base_freq == .2:
            raise RuntimeError('solver reset')
        return run()
    result = select([trajectory(), trajectory(.3)], runner)
    assert result.report['selected_candidate_id'] == 1
    assert 'solver reset' in result.report['candidates'][0]['reasons'][0]


def test_generation_is_reproducible_bounded_and_serializable():
    args = dict(trials=5, limits=([-1, -.5], [1, .5]),
                velocity_limits=[.4, .6], acceleration_limits=1.2)
    a = generate_excitation_candidates(2, np.random.default_rng(20), **args)
    b = generate_excitation_candidates(2, np.random.default_rng(20), **args)
    for first, second in zip(a, b):
        assert np.array_equal(first.a, second.a)
        assert first.base_freq == second.base_freq
        q, dq, ddq = first.evaluate(np.linspace(0, first.period, 5000))
        assert np.all(np.abs(q) <= [1, .5])
        assert np.all(np.abs(dq) <= [.4, .6])
        assert np.max(np.abs(ddq)) <= 1.2


@pytest.mark.parametrize('kwargs', [dict(min_samples=0), dict(max_condition_number=np.inf),
                                     dict(min_retained_fraction=2), dict(objective='unknown')])
def test_invalid_criteria_rejected(kwargs):
    with pytest.raises(ValueError):
        ExcitationCriteria(**kwargs)


def test_all_rejected_cli_saves_failure_report_and_returns_nonzero(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip('mujoco')
    pytest.importorskip('scipy')
    root = Path(__file__).resolve().parents[2]
    config = tmp_path / 'reject.json'
    config.write_text(json.dumps({'max_tracking_rms_rad': 1e-12}))
    out = tmp_path / 'output'
    out.mkdir()
    # 旧的成功报告不能被误认为本次筛选成功。
    (out / 'report.json').write_text('{"status": "completed"}')
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run(
        [sys.executable, '-m', 'robot_model.cli.identify_rrr', '--realistic',
         '--trials', '1', '--periods', '1', '--screening-config', str(config),
         '--output-dir', str(out)], cwd=root, env=env, capture_output=True, text=True,
        timeout=60,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert '没有合格激励轨迹' in result.stderr
    report = json.loads((out / 'screening.json').read_text())
    assert report['accepted_count'] == 0
    assert report['selected_candidate_id'] is None
    assert not (out / 'train.npz').exists()
    assert json.loads((out / 'report.json').read_text())['status'] == 'screening_failed'
