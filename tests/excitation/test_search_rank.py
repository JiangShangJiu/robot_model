"""随机激励搜索只能返回列满秩、条件数有限的候选。"""

import numpy as np
import pytest

from robot_model.excitation import FourierTrajectory, search_excitation


def _five_parameters(q, dq, ddq):
    return np.array([[1.0, q[0], dq[0], ddq[0], q[0]**2]])


def test_underdetermined_regressor_is_rejected_despite_small_condition_number():
    with pytest.raises(ValueError, match="列满秩.*增加采样率"):
        search_excitation(
            _five_parameters, 1, np.random.default_rng(0), trials=2,
            n_harmonics=2, base_freq=0.1, rate=0.2,
        )


def test_column_dependent_regressor_is_rejected_with_sufficient_samples():
    def dependent(q, dq, ddq):
        return np.array([[1.0, q[0], 2*q[0]]])

    with pytest.raises(ValueError, match="列满秩.*列相关性"):
        search_excitation(
            dependent, 1, np.random.default_rng(1), trials=3,
            n_harmonics=2, base_freq=0.5, rate=20,
        )


def test_search_returns_best_full_rank_candidate_and_actual_condition_number():
    options = dict(n_harmonics=2, base_freq=0.5, vel_scale=0.4)
    rng = np.random.default_rng(7)
    candidates = [FourierTrajectory.random(1, rng, **options) for _ in range(4)]
    conditions = []
    for candidate in candidates:
        t = candidate.timestamps(rate=20)
        q, dq, ddq = candidate.evaluate(t)
        W = np.vstack([
            _five_parameters(q[i], dq[i], ddq[i]) for i in range(len(t))
        ])
        assert np.linalg.matrix_rank(W) == W.shape[1]
        conditions.append(np.linalg.cond(W))
    selected, condition = search_excitation(
        _five_parameters, 1, np.random.default_rng(7),
        trials=4, rate=20, **options,
    )
    expected = candidates[int(np.argmin(conditions))]
    np.testing.assert_array_equal(selected.a, expected.a)
    np.testing.assert_array_equal(selected.b, expected.b)
    assert condition == pytest.approx(min(conditions))
    assert np.isfinite(condition)


def test_too_few_time_samples_has_actionable_error():
    with pytest.raises(ValueError, match="采样点不足.*提高采样率"):
        search_excitation(
            _five_parameters, 1, np.random.default_rng(0), trials=1,
            base_freq=1.0, rate=1,
        )
