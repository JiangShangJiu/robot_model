"""基础闭环采样的通道同步、实际力矩和重复运行隔离。"""

import numpy as np
import pytest

from robot_model.excitation import FourierTrajectory
from robot_model.robots.rrr import rrr_xml_path
from robot_model.simulation import MujocoTrackingSource

mujoco = pytest.importorskip("mujoco")


def _trajectory():
    return FourierTrajectory.random(
        3, np.random.default_rng(3), n_harmonics=2,
        base_freq=1.0, vel_scale=0.1,
    )


@pytest.mark.parametrize("decimate", [1, 5])
def test_sensor_and_state_fallback_sample_the_same_instant(decimate):
    model = mujoco.MjModel.from_xml_path(str(rrr_xml_path()))
    trajectory = _trajectory()
    with_sensors = MujocoTrackingSource(model, joint_sensors=True).run(
        trajectory, decimate=decimate,
    )
    fallback = MujocoTrackingSource(model, joint_sensors=False).run(
        trajectory, decimate=decimate,
    )
    q0, dq0, _ = trajectory.evaluate(0.0)
    assert fallback.t[0] == 0.0
    np.testing.assert_allclose(fallback.q[0], q0, atol=1e-14)
    np.testing.assert_allclose(fallback.dq[0], dq0, atol=1e-14)
    for field in ("t", "q", "dq", "ddq", "tau"):
        np.testing.assert_allclose(
            getattr(fallback, field), getattr(with_sensors, field), atol=1e-12,
        )


def test_fallback_records_applied_torque_after_actuator_clipping():
    model = mujoco.MjModel.from_xml_path(str(rrr_xml_path()))
    model.actuator_ctrllimited[:] = True
    model.actuator_ctrlrange[:] = [-0.02, 0.02]
    trajectory = _trajectory()
    with_sensors = MujocoTrackingSource(model, joint_sensors=True).run(trajectory)
    fallback = MujocoTrackingSource(model, joint_sensors=False).run(trajectory)
    assert np.max(np.abs(fallback.tau)) <= 0.02 + 1e-12
    # A nonzero saturated sample proves this is testing the actual clipping path.
    assert np.any(np.isclose(np.abs(with_sensors.tau), 0.02))
    np.testing.assert_allclose(fallback.tau, with_sensors.tau, atol=1e-12)


def test_repeated_runs_reset_time_and_actuator_state():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.002"/>'
        '<worldbody><body><joint name="j" axis="0 0 1"/>'
        '<geom type="capsule" fromto="0 0 0 .5 0 0" size=".05"/>'
        '</body></worldbody><actuator>'
        '<general joint="j" dyntype="filter" dynprm=".1"/>'
        '</actuator></mujoco>'
    )
    trajectory = FourierTrajectory([0.0], [[0.1]], [[0.1]], 1.0)
    source = MujocoTrackingSource(model, kp=1.0, kd=0.1)
    first = source.run(trajectory)
    first_end_time = source.data.time
    assert np.linalg.norm(source.data.act) > 0.01
    second = source.run(trajectory)
    assert source.data.time == pytest.approx(first_end_time)
    for field in ("t", "q", "dq", "ddq", "tau"):
        np.testing.assert_allclose(
            getattr(first, field), getattr(second, field), atol=1e-12,
        )
