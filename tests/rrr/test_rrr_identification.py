"""辨识闭环：MuJoCo 出数据 → 最小二乘 → 恢复 MJCF 里写死的真参数。

这是整条辨识链路唯一的端到端验证。前面的测试都是正向的
（给参数算力矩），这里是反向的：只看运动和力矩，反推参数。
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from robot_model import Robot
from robot_model.data import add_noise, sample_trajectory
from robot_model.excitation import FourierTrajectory, search_excitation
from robot_model.identification import (
    DynamicsDecomposer,
    identify,
    parameter_error,
    predict_torque,
    stack_regressor,
)
from robot_model.simulation.reference import MujocoReference
from robot_model.robots.rrr import (
    RRR_BODY_NAMES,
    RRR_MDH_PARMS,
    rrr_xml_path,
)

mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def setup():
    """``(Hb_func, pi_b 真值, MuJoCo 参照)``。"""
    robot = Robot.from_mdh("rrr_arm", RRR_MDH_PARMS)
    dyn = robot.dynamics
    dyn.gen_regressor()
    dyn.calc_base_parms()
    Hb = dyn.gen_base_regressor()

    s = robot.symbols
    f = sympy.lambdify(list(s.q) + list(s.dq) + list(s.ddq), Hb, "numpy")

    def Hb_func(q, dq, ddq):
        return np.asarray(f(*(list(q) + list(dq) + list(ddq))), dtype=float)

    ref = MujocoReference(
        mujoco.MjModel.from_xml_path(str(rrr_xml_path())), RRR_BODY_NAMES
    )
    pi_b = np.asarray(
        sympy.lambdify(list(s.dynparms()), sympy.Matrix(dyn.baseparms), "numpy")(
            *ref.dynparms()
        ),
        dtype=float,
    ).ravel()
    return Hb_func, pi_b, ref


@pytest.fixture(scope="module")
def excited(setup):
    Hb_func, _, ref = setup
    traj, cond = search_excitation(
        Hb_func, 3, np.random.default_rng(0), trials=20, vel_scale=1.2
    )
    data = sample_trajectory(traj, ref.inverse_dynamics, rate=100)
    return traj, cond, data


def test_noise_free_identification_recovers_true_parameters(setup, excited):
    Hb_func, pi_b, _ = setup
    _, _, data = excited
    result = identify(Hb_func, data)
    err = parameter_error(result.parms, pi_b)
    assert err["rel_norm"] < 1e-9, result.summary()
    assert result.residual_rms < 1e-9
    assert result.rank == pi_b.size


def test_excitation_search_beats_an_arbitrary_trajectory(setup, excited):
    """随机搜索应当明显压低条件数，否则这个函数没有存在价值。"""
    Hb_func, _, _ = setup
    _, cond, _ = excited
    arbitrary = FourierTrajectory.random(
        3, np.random.default_rng(3), vel_scale=0.06, base_freq=0.03
    )
    t = arbitrary.timestamps(rate=50)
    q, dq, ddq = arbitrary.evaluate(t)
    W = np.vstack([Hb_func(q[i], dq[i], ddq[i]) for i in range(t.size)])
    assert cond < np.linalg.cond(W)


def test_parameter_error_grows_with_noise(setup, excited):
    Hb_func, pi_b, _ = setup
    _, _, data = excited
    errs = []
    for std in (0.01, 0.05, 0.2):
        noisy = add_noise(data, np.random.default_rng(1), tau_std=std)
        est = identify(Hb_func, noisy).parms
        errs.append(parameter_error(est, pi_b)["rel_norm"])
    assert errs[0] < errs[1] < errs[2]
    # 条件数约 30 量级，0.05 N·m 噪声不该把参数打坏到 10% 以上
    assert errs[1] < 0.1


def test_ill_conditioned_trajectory_amplifies_noise(setup):
    """低激励轨迹下同样的噪声会把参数打飞——这就是要优化轨迹的理由。"""
    Hb_func, pi_b, ref = setup
    poor = FourierTrajectory.random(
        3, np.random.default_rng(3), vel_scale=0.06, base_freq=0.03
    )
    data = sample_trajectory(poor, ref.inverse_dynamics, rate=100)
    noisy = add_noise(data, np.random.default_rng(1), tau_std=0.05)
    err = parameter_error(identify(Hb_func, noisy).parms, pi_b)["rel_norm"]
    assert err > 1.0  # 相对误差超过 100%，完全不可用


def test_low_prediction_residual_does_not_imply_correct_parameters(setup):
    """辨识里的经典陷阱：病态方向上参数大错，力矩却几乎拟合得上。

    所以验收辨识结果不能只看残差，必须同时看条件数。
    """
    Hb_func, pi_b, ref = setup
    poor = FourierTrajectory.random(
        3, np.random.default_rng(3), vel_scale=0.06, base_freq=0.03
    )
    data = sample_trajectory(poor, ref.inverse_dynamics, rate=100)
    noisy = add_noise(data, np.random.default_rng(1), tau_std=0.05)

    train = np.arange(0, len(data), 2)
    val = np.arange(1, len(data), 2)
    result = identify(Hb_func, noisy.subset(train))

    pred = predict_torque(Hb_func, data.subset(val), result.parms)
    residual = np.sqrt(np.mean((pred - data.subset(val).tau) ** 2))
    err = parameter_error(result.parms, pi_b)["rel_norm"]

    assert residual < 0.05  # 预测看着很准
    assert err > 1.0  # 参数却完全不对
    assert result.condition_number > 1e3  # 唯一的预警信号


def test_full_regressor_is_rank_deficient(setup, excited):
    """完整 H 列相关，必须在最小参数集上辨识——否则解不唯一。"""
    robot = Robot.from_mdh("rrr_arm", RRR_MDH_PARMS)
    H = robot.dynamics.gen_regressor()
    s = robot.symbols
    f = sympy.lambdify(list(s.q) + list(s.dq) + list(s.ddq), H, "numpy")
    _, _, data = excited
    W = np.vstack(
        [
            np.asarray(
                f(*(list(data.q[i]) + list(data.dq[i]) + list(data.ddq[i]))),
                dtype=float,
            )
            for i in range(0, len(data), 20)
        ]
    )
    n_parms = len(list(s.dynparms()))
    assert W.shape[1] == n_parms
    assert np.linalg.matrix_rank(W) < n_parms


def test_weighted_least_squares_runs(setup, excited):
    Hb_func, pi_b, _ = setup
    _, _, data = excited
    result = identify(Hb_func, data, weights=[1.0, 2.0, 4.0])
    assert parameter_error(result.parms, pi_b)["rel_norm"] < 1e-8


def test_stacked_regressor_matches_observations(setup, excited):
    Hb_func, pi_b, _ = setup
    _, _, data = excited
    W, y = stack_regressor(Hb_func, data)
    assert np.allclose(W @ pi_b, y, atol=1e-9)


class TestDecomposition:
    """辨识结果 → M / c / g。基参数不可逆，但力矩映射完全确定。"""

    @staticmethod
    def _decomposer(setup, excited):
        Hb_func, _, _ = setup
        _, _, data = excited
        return DynamicsDecomposer(Hb_func, identify(Hb_func, data).parms, 3)

    def test_gravity_matches_static_inverse_dynamics(self, setup, excited):
        _, _, ref = setup
        dec = self._decomposer(setup, excited)
        q = np.array([0.3, -0.6, 0.9])
        zero = np.zeros(3)
        assert np.allclose(
            dec.gravity(q), ref.inverse_dynamics(q, zero, zero), atol=1e-9
        )

    def test_inertia_matches_mujoco_mass_matrix(self, setup, excited):
        _, _, ref = setup
        dec = self._decomposer(setup, excited)
        q = np.array([0.3, -0.6, 0.9])

        d = mujoco.MjData(ref.model)
        d.qpos[:] = q
        mujoco.mj_forward(ref.model, d)
        M_mj = np.zeros((ref.model.nv, ref.model.nv))
        mujoco.mj_fullM(ref.model, M_mj, d.qM)

        assert np.allclose(dec.inertia(q), M_mj, atol=1e-9)

    def test_inertia_is_symmetric(self, setup, excited):
        """分解逐列独立求得，对称性不是构造出来的，可当正确性判据。"""
        dec = self._decomposer(setup, excited)
        terms = dec.at(np.array([0.2, 0.7, -0.4]), np.array([0.5, -0.4, 0.7]))
        assert terms.symmetry_error() < 1e-9

    def test_recombination_reproduces_torque(self, setup, excited):
        """M ddq + c + g 必须还原 mj_inverse，否则三项分配错了。"""
        _, _, ref = setup
        dec = self._decomposer(setup, excited)
        rng = np.random.default_rng(7)
        for _ in range(20):
            q, dq, ddq = (rng.uniform(-1.5, 1.5, 3) for _ in range(3))
            terms = dec.at(q, dq)
            assert np.allclose(
                terms.torque(ddq), ref.inverse_dynamics(q, dq, ddq), atol=1e-9
            )

    def test_parameter_dimension_is_checked(self, setup, excited):
        Hb_func, _, _ = setup
        bad = DynamicsDecomposer(Hb_func, np.zeros(3), 3)
        with pytest.raises(ValueError, match="参数维数"):
            bad.gravity(np.zeros(3))
