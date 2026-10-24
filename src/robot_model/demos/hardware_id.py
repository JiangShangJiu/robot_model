"""演示 1：仿真"实机"辨识闭环——辨识前后模型与控制器的可量化改进。

流程：激励轨迹搜索 → 闭环采集（噪声、量化、延迟、Stribeck 摩擦、可选
末端负载失配）→ 基参数辨识 → 在独立保留轨迹上对比：

1. 开环力矩预测：名义模型 vs 辨识模型（coulomb_viscous 与 stribeck 两种
   摩擦结构），按关节与整体 RMSE 报告，并给出含低速段的整段误差。
2. 模型前馈控制器：同一对象、同一反馈增益，仅把前馈换成辨识参数，
   比较跟踪 RMSE——回答"辨识到底换来了多少控制性能"。

对象真值只用于负载设定与文档说明；辨识与控制只读测量链输出。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..excitation import search_excitation
from ..experiments.pipeline import prepare_measurements, torque_rms
from ..experiments.realistic import DEFAULT_MIN_SPEED, identification_robot, parameters
from ..experiments.rrr import default_config, setup
from ..identification import identify, predict_torque
from ..simulation import RealisticMujocoSource
from .common import ensure_output_dir, rounded, save_metrics, use_headless_matplotlib


@dataclass(frozen=True)
class HardwareIdDemoConfig:
    """演示假设；``payload_mass_kg=0`` 可关闭负载失配。"""

    seed: int = 0
    trials: int = 8
    periods: int = 2
    payload_mass_kg: float = 0.6
    payload_offset_m: tuple = (0.22, 0.0, 0.0)  # link3 帧，即工具端
    base_parms_samples: int = 400
    # 控制对比用较低反馈增益：前馈质量的差异在弱反馈下才看得清。
    feedback_scale: float = 0.4

    def to_dict(self):
        return asdict(self)


def payload_plant_model(xml_path, mass_kg: float, offset_m):
    """把点质量负载焊接到 link3 末端，作为对象侧的结构性失配。

    名义/前馈模型不含该负载；辨识应当把它吸收进基参数。
    """
    import mujoco

    if mass_kg <= 0:
        raise ValueError("负载质量必须为正；无负载时直接不构造该模型")
    spec = mujoco.MjSpec.from_file(str(xml_path))
    body = spec.body("link3")
    if body is None:
        raise ValueError("模型缺少 link3，无法挂载负载")
    payload = body.add_body(name="demo_payload", pos=list(offset_m))
    payload.mass = float(mass_kg)
    payload.ipos = [0.0, 0.0, 0.0]
    # 近似为半径 3 cm 的实心球，惯量远小于其平移贡献
    inertia = 0.4 * mass_kg * 0.03**2
    payload.inertia = [inertia] * 3
    payload.explicitinertial = True
    return spec.compile()


class RegressorFeedforwardSource(RealisticMujocoSource):
    """前馈改用 ``Hb(qd,dqd,ddqd) @ parms``，测量链与对象完全不变。

    用名义参数投影即可复现"辨识前"的控制器，用辨识参数得到"辨识后"；
    两者只差一个参数向量，对照公平。
    """

    def __init__(self, nominal_model, config, regressor_func, parms, *, plant_model=None):
        super().__init__(nominal_model, config, plant_model=plant_model)
        self._regressor_func = regressor_func
        self._parms = np.asarray(parms, float).copy()

    def feedforward(self, q, dq, ddq):
        return np.asarray(self._regressor_func(q, dq, ddq), float) @ self._parms


def _identification_models(realism, demo):
    """两种摩擦结构的辨识模型；stribeck 额外建模低速摩擦峰。"""
    models = {}
    for friction_model in ("coulomb_viscous", "stribeck"):
        robot = identification_robot(setup(), friction_model,
                                     stribeck_velocity=realism.friction_velocity)
        dyn = robot.dynamics
        dyn.calc_base_parms(samples=demo.base_parms_samples)
        dyn.gen_base_regressor()
        models[friction_model] = robot
    return models


def _prepare(data, realism, min_speed):
    s = realism.sensors
    return prepare_measurements(
        data, encoder_delay_steps=s.encoder_delay_steps,
        torque_delay_steps=s.torque_delay_steps, min_speed=min_speed,
    )


def run_demo(output_dir, demo: HardwareIdDemoConfig | None = None) -> dict:
    import mujoco

    demo = demo or HardwareIdDemoConfig()
    out = ensure_output_dir(output_dir)
    rrr = setup()
    realism = default_config(demo.seed)
    nominal = mujoco.MjModel.from_xml_path(str(rrr.xml_path))
    plant = (payload_plant_model(rrr.xml_path, demo.payload_mass_kg, demo.payload_offset_m)
             if demo.payload_mass_kg > 0 else None)
    source = RealisticMujocoSource(nominal, realism, plant_model=plant)

    print("构建辨识模型（coulomb_viscous 与 stribeck 两种摩擦结构）…", flush=True)
    models = _identification_models(realism, demo)
    Hb = {name: robot.dynamics.Hb_func for name, robot in models.items()}

    limits = (realism.joint_lower, realism.joint_upper)
    rng = np.random.default_rng(demo.seed + 101)
    print("搜索训练与保留验证激励轨迹…", flush=True)
    train_traj, train_cond = search_excitation(
        Hb["stribeck"], rrr.dof, rng, trials=demo.trials,
        base_freq=rrr.train_base_freq, n_harmonics=rrr.train_harmonics,
        vel_scale=rrr.train_vel_scale, limits=limits,
        velocity_limits=realism.velocity_limit,
        acceleration_limits=realism.acceleration_limit)
    heldout_traj, heldout_cond = search_excitation(
        Hb["stribeck"], rrr.dof, np.random.default_rng(demo.seed + 202),
        trials=demo.trials, base_freq=rrr.validation_base_freq,
        n_harmonics=rrr.validation_harmonics, vel_scale=rrr.validation_vel_scale,
        limits=limits, velocity_limits=realism.velocity_limit,
        acceleration_limits=realism.acceleration_limit)

    print("闭环采集训练数据（含测量噪声/延迟/摩擦/负载失配）…", flush=True)
    train = source.run(train_traj, periods=demo.periods, seed=demo.seed + 303)
    print("闭环采集保留验证数据…", flush=True)
    heldout = source.run(heldout_traj, periods=demo.periods, seed=demo.seed + 404)

    results, predictions = {}, {}
    heldout_full = _prepare(heldout.measured, realism, min_speed=0.0)
    for name, robot in models.items():
        train_data = _prepare(train.measured, realism, DEFAULT_MIN_SPEED[name])
        result = identify(Hb[name], train_data)
        nominal_parms = parameters(robot, source.nominal_model, rrr.body_names)
        valid = _prepare(heldout.measured, realism, DEFAULT_MIN_SPEED[name])
        results[name] = {
            "train_samples": len(train_data),
            "rank": result.rank, "n_parms": result.n_parms,
            "condition_number": result.condition_number,
            "train_residual_rms_Nm": result.residual_rms,
            "heldout_valid_samples": len(valid),
            "open_loop_rmse_Nm": {
                "identified_valid": torque_rms(predict_torque(Hb[name], valid, result.parms), valid.tau),
                "nominal_valid": torque_rms(predict_torque(Hb[name], valid, nominal_parms), valid.tau),
                "identified_full": torque_rms(predict_torque(Hb[name], heldout_full, result.parms), heldout_full.tau),
                "nominal_full": torque_rms(predict_torque(Hb[name], heldout_full, nominal_parms), heldout_full.tau),
            },
        }
        predictions[name] = {
            "parms": result.parms,
            "nominal_parms": nominal_parms,
            "full_prediction": predict_torque(Hb[name], heldout_full, result.parms),
            "nominal_prediction": predict_torque(Hb[name], heldout_full, nominal_parms),
        }
        print(f"[{name}] 基参数 {result.rank}/{result.n_parms}，"
              f"训练残差 RMS {result.residual_rms:.4f} N·m", flush=True)

    print("控制对照：同一对象上换前馈参数，重跑保留轨迹…", flush=True)
    control_cfg = _low_gain(realism, demo.feedback_scale)
    tracking = {}
    for label, parms in (("before_id_nominal", predictions["stribeck"]["nominal_parms"]),
                         ("after_id_identified", predictions["stribeck"]["parms"])):
        ctrl_source = RegressorFeedforwardSource(
            nominal, control_cfg, Hb["stribeck"], parms, plant_model=plant)
        run = ctrl_source.run(heldout_traj, periods=demo.periods, seed=demo.seed + 505)
        tracking[label] = {
            "tracking_rms_rad": run.diagnostics["tracking_rms_rad"],
            "tracking_max_rad": run.diagnostics["tracking_max_rad"],
            "saturation_fraction": run.diagnostics["saturation_fraction"],
            "_error": np.asarray(run.truth.q) - np.asarray(run.desired_q),
            "_t": run.measured.t,
        }

    before = np.asarray(tracking["before_id_nominal"]["tracking_rms_rad"])
    after = np.asarray(tracking["after_id_identified"]["tracking_rms_rad"])
    metrics = {
        "demo": "hardware_id_closed_loop",
        "settings": {**demo.to_dict(),
                     "friction_plant": "viscous+Coulomb+Stribeck peak",
                     "trajectory_condition_number": {"train": train_cond, "heldout": heldout_cond},
                     "control_feedback_gains": {"kp": list(np.asarray(control_cfg.kp, float).reshape(-1)),
                                                "kd": list(np.asarray(control_cfg.kd, float).reshape(-1))}},
        "identification": results,
        "tracking_rms_rad": {
            "before_id_nominal": before.tolist(),
            "after_id_identified": after.tolist(),
            "improvement_percent": (100 * (1 - after / before)).tolist(),
        },
        "notes": [
            "对象含 Stribeck 低速摩擦峰与末端负载失配；名义前馈两者皆未建模。",
            "coulomb_viscous 剔除 |dq|<0.08 rad/s 样本，stribeck 剔除 |dq|<0.02。",
            "full 指含低速段的整段保留数据；低速区是摩擦模型的主要失配区。",
        ],
    }
    save_metrics(out / "metrics.json", rounded(metrics))
    _plot_torque_prediction(out, heldout_full, predictions)
    _plot_residual_vs_speed(out, heldout_full, predictions)
    _plot_tracking(out, tracking)
    print(f"产物已写入 {out.resolve()}")
    return metrics


def _low_gain(realism, scale):
    from dataclasses import replace

    kp = scale * np.asarray(realism.kp, float).reshape(-1)
    kd = scale * np.asarray(realism.kd, float).reshape(-1)
    return replace(realism, kp=tuple(kp), kd=tuple(kd))


def _plot_torque_prediction(out, data, predictions):
    plt = use_headless_matplotlib()
    dof = data.dof
    fig, axes = plt.subplots(dof, 1, figsize=(10, 1.9 * dof + 0.6), sharex=True)
    pred = predictions["stribeck"]
    for i, axis in enumerate(np.atleast_1d(axes)):
        axis.plot(data.t, data.tau[:, i], lw=1.2, color="0.25", label="measured")
        axis.plot(data.t, pred["full_prediction"][:, i], lw=1.0, ls="--",
                  color="#55a868", label="identified (stribeck)")
        axis.plot(data.t, pred["nominal_prediction"][:, i], lw=0.9, ls=":",
                  color="#c44e52", label="nominal model")
        axis.set_ylabel(f"tau{i + 1} [N m]", fontsize=9)
    np.atleast_1d(axes)[0].set_title(
        "Held-out trajectory: torque prediction before vs after identification")
    np.atleast_1d(axes)[0].legend(ncol=3, fontsize=8, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(out / "torque_prediction.png", dpi=130)
    plt.close(fig)


def _plot_residual_vs_speed(out, data, predictions):
    """低速区失配可视化：摩擦建模差异集中在 |dq| 小的样本。"""
    plt = use_headless_matplotlib()
    fig, axes = plt.subplots(1, data.dof, figsize=(4.0 * data.dof, 3.4), sharey=True)
    series = (("nominal", predictions["stribeck"]["nominal_prediction"], "#c44e52"),
              ("coulomb_viscous", predictions["coulomb_viscous"]["full_prediction"], "#dd8452"),
              ("stribeck", predictions["stribeck"]["full_prediction"], "#4c72b0"))
    for i, axis in enumerate(np.atleast_1d(axes)):
        speed = np.abs(data.dq[:, i])
        edges = np.linspace(0.0, speed.max() + 1e-9, 25)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bins = np.digitize(speed, edges) - 1
        for label, prediction, color in series:
            residual = np.abs(prediction[:, i] - data.tau[:, i])
            axis.scatter(speed, residual, s=3, alpha=0.15, color=color)
            median = np.array([np.median(residual[bins == b]) if np.any(bins == b)
                               else np.nan for b in range(len(centers))])
            axis.plot(centers, median, lw=1.8, color=color,
                      label=f"{label} (binned median)" if i == 0 else None)
        axis.axvline(0.08, color="0.4", lw=0.8, ls="--")
        axis.set_yscale("log")
        axis.set_xlabel(f"|dq{i + 1}| [rad/s]")
        if i == 0:
            axis.set_ylabel("|torque residual| [N m]")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Residual vs joint speed on held-out data "
                 "(dashed line: 0.08 rad/s low-speed cutoff)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "residual_vs_speed.png", dpi=130)
    plt.close(fig)


def _plot_tracking(out, tracking):
    plt = use_headless_matplotlib()
    before, after = tracking["before_id_nominal"], tracking["after_id_identified"]
    error_b, error_a = before["_error"], after["_error"]
    dof = error_b.shape[1]
    fig, axes = plt.subplots(dof, 1, figsize=(10, 1.9 * dof + 0.6), sharex=True)
    for i, axis in enumerate(np.atleast_1d(axes)):
        axis.plot(before["_t"], error_b[:, i], lw=0.9, color="#c44e52",
                  label="before ID (nominal feedforward)")
        axis.plot(after["_t"], error_a[:, i], lw=0.9, color="#4c72b0",
                  label="after ID (identified feedforward)")
        axis.set_ylabel(f"q{i + 1} error [rad]", fontsize=9)
        axis.text(0.99, 0.95,
                  f"RMS {before['tracking_rms_rad'][i]:.4f} -> {after['tracking_rms_rad'][i]:.4f} rad",
                  transform=axis.transAxes, ha="right", va="top", fontsize=8)
    np.atleast_1d(axes)[0].set_title(
        "Low-gain tracking on held-out trajectory: same plant, same feedback, only feedforward changes")
    np.atleast_1d(axes)[0].legend(ncol=2, fontsize=8, loc="lower right")
    np.atleast_1d(axes)[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(out / "tracking_before_after.png", dpi=130)
    plt.close(fig)
