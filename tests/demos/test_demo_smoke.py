"""三个演示的端到端冒烟测试：产物齐全且核心断言成立。

演示 1 含两套符号模型推导，标记为 slow；演示 2/3 秒级，进快速套件。
"""

import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")


def load_metrics(path):
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


class TestHardwareIdDemo:
    @pytest.mark.slow
    def test_end_to_end(self, tmp_path):
        from robot_model.demos.hardware_id import HardwareIdDemoConfig, run_demo

        out = tmp_path / "hardware_id"
        run_demo(out, HardwareIdDemoConfig(trials=2))
        metrics = load_metrics(out)
        for name in ("torque_prediction.png", "residual_vs_speed.png",
                     "tracking_before_after.png"):
            assert (out / name).stat().st_size > 0
        for friction_model in ("coulomb_viscous", "stribeck"):
            fit = metrics["identification"][friction_model]
            assert fit["rank"] == fit["n_parms"], "基参数必须满秩"
            rmse = fit["open_loop_rmse_Nm"]
            assert np.all(np.asarray(rmse["identified_valid"])
                          < np.asarray(rmse["nominal_valid"])), "辨识模型必须优于名义模型"
        tracking = metrics["tracking_rms_rad"]
        assert np.all(np.asarray(tracking["after_id_identified"])
                      < np.asarray(tracking["before_id_nominal"])), "辨识后跟踪必须改善"

    def test_payload_model_adds_point_mass(self):
        from robot_model.demos.hardware_id import payload_plant_model
        from robot_model.robots.rrr import rrr_xml_path

        nominal = mujoco.MjModel.from_xml_path(str(rrr_xml_path()))
        plant = payload_plant_model(rrr_xml_path(), 0.6, (0.22, 0.0, 0.0))
        assert plant.nv == nominal.nv == 3
        assert plant.nbody == nominal.nbody + 1
        assert np.isclose(plant.body_mass.sum(), nominal.body_mass.sum() + 0.6)

    def test_regressor_feedforward_matches_nominal_inverse_dynamics(self):
        """名义参数投影经回归矩阵的前馈 = 名义模型 mj_inverse。"""
        from robot_model.demos.hardware_id import RegressorFeedforwardSource
        from robot_model.experiments.realistic import identification_robot, parameters
        from robot_model.experiments.rrr import default_config, setup

        rrr = setup()
        robot = identification_robot(rrr, "coulomb_viscous")
        dyn = robot.dynamics
        dyn.calc_base_parms(samples=200)
        dyn.gen_base_regressor()
        nominal = mujoco.MjModel.from_xml_path(str(rrr.xml_path))
        parms = parameters(robot, nominal, rrr.body_names)
        source = RegressorFeedforwardSource(
            nominal, default_config(0), dyn.Hb_func, parms)
        rng = np.random.default_rng(3)
        for _ in range(3):
            q, dq, ddq = rng.uniform(-1, 1, (3, 3))
            expected = RegressorFeedforwardSource.__mro__[1].feedforward(source, q, dq, ddq)
            assert np.allclose(source.feedforward(q, dq, ddq), expected, atol=1e-8)


class TestContactSafetyDemo:
    def test_guarded_vs_unguarded(self, tmp_path):
        from robot_model.demos.contact_safety import ContactDemoConfig, run_demo

        out = tmp_path / "contact"
        metrics = run_demo(out, ContactDemoConfig())
        for name in ("force_tracking.png", "tank_and_contact.png"):
            assert (out / name).stat().st_size > 0
        without, with_tank = metrics["without_tank"], metrics["with_tank"]
        assert without["contact_loss_events_after_touch"] >= 10, "无护栏必须出现持续弹跳"
        assert with_tank["contact_loss_events_after_touch"] <= 5
        assert with_tank["force_rmse_N"] < 0.3 * without["force_rmse_N"]
        assert with_tank["force_peak_N"] < without["force_peak_N"]
        assert with_tank["scale_active_fraction"] > 0.05, "护栏必须实际介入"

    def test_gravity_compensation_ignores_contact(self):
        """重力补偿在无接触副本上计算：有无穿透结果一致。"""
        from robot_model.demos.contact_safety import contact_scene_path
        from robot_model.demos.task_space import TaskSpaceArm

        model = mujoco.MjModel.from_xml_path(str(contact_scene_path()))
        arm = TaskSpaceArm(model)
        free = arm.gravity_torque(np.array([0.0, -0.6, 1.2]))
        pressed = arm.gravity_torque(np.array([0.0, -0.35, 0.75]))  # 工具在墙内
        assert np.all(np.isfinite(free)) and np.all(np.isfinite(pressed))
        # 第二个位形若把接触力算进来会出现数十 N·m 的额外项
        d = mujoco.MjData(model)
        d.qpos[:] = [0.0, -0.35, 0.75]
        mujoco.mj_forward(model, d)
        if d.ncon > 0:
            assert np.linalg.norm(pressed) < 20.0


class TestVlaBridgeDemo:
    def test_scenarios_contrast(self, tmp_path):
        from robot_model.demos.vla_bridge import VlaBridgeDemoConfig, run_demo

        out = tmp_path / "vla"
        config = VlaBridgeDemoConfig()
        metrics = run_demo(out, config)
        for name in ("paths.png", "timeline.png"):
            assert (out / name).stat().st_size > 0
        scenarios = metrics["scenarios"]
        ideal, naive, guarded = (scenarios[key] for key in
                                 ("ideal_link", "degraded_naive", "degraded_guarded"))
        assert ideal["tracking_rmse_m"] < 0.005
        assert naive["tracking_rmse_m"] > 3 * ideal["tracking_rmse_m"], "退化链路必须显著劣化"
        assert naive["ee_speed_max_m_s"] > 2 * config.setpoint_speed_max_m_s, "裸桥接必须出现超速"
        # 护栏保证：速度钳位生效（留 20% 动态裕量）、保持被触发、无越界
        assert guarded["ee_speed_max_m_s"] < 1.2 * config.setpoint_speed_max_m_s
        assert guarded["hold_fraction"] > 0.05
        assert guarded["outside_workspace_fraction"] == 0.0
        assert guarded["dropped_chunks"] == naive["dropped_chunks"], "同种子下扰动必须一致"
