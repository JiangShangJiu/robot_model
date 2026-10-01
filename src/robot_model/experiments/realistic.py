"""与机器人无关的实机近似辨识实验编排。

对象真值参数只参与最后评分；反馈和最小二乘不读取这些参数。
筛选的可执行性检查使用仿真状态诊断，激励信息评分使用预处理后的测量数据。

每台机器人用一个 :class:`RealisticSetup` 描述差异，并在 ``experiments/<机器人>.py``
里提供 ``default_config()`` 与 ``run_experiment()``；本模块不含任何机器人专有常量。
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from robot_model import Robot
from robot_model.data import align_measurements, average_periods
from robot_model.excitation import (
    generate_excitation_candidates,
    NoFeasibleExcitation,
    search_excitation,
    screen_excitation,
    screen_robust_excitation,
)
from robot_model.experiments.pipeline import (
    prepare_measurements,
    refine_excitation_candidates,
    torque_rms,
)
from robot_model.identification import (
    RecoveryConfig,
    identify,
    pack_inertial_parameters,
    predict_torque,
    recover_inertial_parameters,
)
from robot_model.simulation import RealisticMujocoSource
from robot_model.simulation.reference import MujocoReference
from robot_model.utils.validation import integer


@dataclass(frozen=True)
class RealisticSetup:
    """一台机器人接入实机近似实验所需的全部差异。

    ``xml_path`` 指向的 MJCF 必须是固定基座、每关节一个 ``gear=1`` 力矩 motor；
    ``body_names`` 按关节顺序对应 MDH 连杆帧，用于读取名义参数和离线评分参数。
    激励相关的基频、谐波数与幅值比例是各机器人的实验假设。
    """

    name: str
    mdh_parms: list
    body_names: list
    xml_path: Path
    assumptions: str
    base_parms_samples: int = 400
    candidate_vel_scale: float = 1.0
    validation_base_freq: float = 0.19
    validation_harmonics: int = 4
    validation_vel_scale: float = 0.85
    train_base_freq: float = 0.15
    train_harmonics: int = 4
    train_vel_scale: float = 1.0
    # 控制 search_excitation 内部对最优随机候选的 SLSQP 精炼。
    # 高自由度下可关闭以限制搜索预算；与 run_experiment 的训练候选 optimize 独立。
    # 验证轨迹只需在训练筛选前固定且激励充分，与是否精炼无关。
    search_refine: bool = True

    @property
    def dof(self) -> int:
        return len(self.mdh_parms)


def parameters(robot, model, body_names):
    """仅评分使用：从 MuJoCo 模型读取惯性、摩擦和 armature。"""
    ref = MujocoReference(model, body_names)
    full = []
    for i, link in enumerate(ref.links):
        full.extend([*link.Le, *link.l, link.m, model.dof_armature[i],
                     model.dof_damping[i], model.dof_frictionloss[i]])
    dyn = robot.dynamics
    return (np.asarray(dyn.Pb, float).T + np.asarray(dyn.Kd, float) @ np.asarray(dyn.Pd, float).T) @ full


def nominal_recovery_parameters(robot, nominal_model, body_names):
    """恢复初值仅来自名义模型，包含其自身的电机惯量和摩擦。"""
    ref = MujocoReference(nominal_model, body_names)
    symbols = robot.symbols
    extras = {}
    for i in range(robot.dof):
        extras[str(symbols.Ia[i])] = float(nominal_model.dof_armature[i])
        extras[str(symbols.fv[i])] = float(nominal_model.dof_damping[i])
        extras[str(symbols.fc[i])] = float(nominal_model.dof_frictionloss[i])
    return pack_inertial_parameters(
        robot, [link.m for link in ref.links], [link.r for link in ref.links],
        [link.I for link in ref.links], extras=extras,
    )


def prepare(data, config, cutoff=5.0, *, period=None, cycles=None):
    """测量预处理：读取 RealismConfig 中的传感器延迟。"""
    s = config.sensors
    return prepare_measurements(
        data,
        encoder_delay_steps=s.encoder_delay_steps,
        torque_delay_steps=s.torque_delay_steps,
        cutoff=cutoff,
        period=period,
        cycles=cycles,
    )


def run_experiment(setup, config, *, trials=12, periods=2, output_dir=None,
                   screen=True, screening_criteria=None, optimize=True,
                   optimization_starts=2, optimization_maxiter=20,
                   robust_seeds=None, periodic_average=False,
                   recover=False, recovery_config=None):
    import mujoco

    integer(trials, 'trials', 1)
    integer(periods, 'periods', 1)
    if recovery_config is not None and not recover:
        raise ValueError('recovery_config 需要 recover=True')
    if recovery_config is not None and not isinstance(recovery_config, RecoveryConfig):
        raise TypeError('recovery_config 必须是 RecoveryConfig')
    if recover and recovery_config is None:
        recovery_config = RecoveryConfig()
    if periodic_average and periods < 2:
        raise ValueError('周期平均至少需要 --periods 2')
    if not screen:
        if robust_seeds is not None:
            raise ValueError('多种子筛选需要启用闭环筛选')
        optimize = False
    if robust_seeds is not None:
        robust_seeds = [integer(s, 'robust seed') for s in robust_seeds]
        if len(robust_seeds) < 2 or len(set(robust_seeds)) != len(robust_seeds):
            raise ValueError('robust_seeds 至少包含两个不同种子')
        if robust_seeds[0] != config.seed:
            raise ValueError('robust_seeds 第一个种子须等于 --seed/配置 seed，以保持训练对象一致')
    dof = setup.dof
    nominal = mujoco.MjModel.from_xml_path(str(setup.xml_path))
    source = RealisticMujocoSource(nominal, config)
    def preprocess(data, trajectory):
        return prepare(data, config, period=trajectory.period if periodic_average else None,
                       cycles=periods if periodic_average else None)
    robot = Robot.from_mdh(setup.name, setup.mdh_parms,
                           frictionmodel={"viscous", "Coulomb"}, driveinertiamodel="simplified")
    dyn = robot.dynamics
    dyn.calc_base_parms(samples=setup.base_parms_samples)
    dyn.gen_base_regressor()
    Hb = dyn.Hb_func
    print(f"实机近似模式：基参数 {dyn.n_base}，物理 {1/config.physics_dt:g} Hz，控制/采样 {1/config.control_dt:g} Hz")
    print("参数为仿真假设；测量反馈进入闭环，真值只用于离线评分。")

    limits = None if config.joint_lower is None else (config.joint_lower, config.joint_upper)
    # 在训练之前固定验证轨迹，不根据辨识效果反复挑选验证集。
    validation_traj, val_cond = search_excitation(
        Hb, dof, np.random.default_rng(config.seed+202), trials=trials,
        base_freq=setup.validation_base_freq, n_harmonics=setup.validation_harmonics,
        vel_scale=setup.validation_vel_scale, limits=limits,
        velocity_limits=config.velocity_limit, acceleration_limits=config.acceleration_limit,
        refine=setup.search_refine,
    )
    screening_report = None
    optimization_report = []
    if screen:
        candidates = generate_excitation_candidates(
            dof, np.random.default_rng(config.seed+101), trials=trials, limits=limits,
            vel_scale=setup.candidate_vel_scale,
            velocity_limits=config.velocity_limit, acceleration_limits=config.acceleration_limit,
        )
        if optimize:
            candidates, optimization_report = refine_excitation_candidates(
                Hb, candidates, limits=limits,
                velocity_limits=config.velocity_limit,
                acceleration_limits=config.acceleration_limit,
                starts=optimization_starts, maxiter=optimization_maxiter,
                progress=lambda msg: print(msg, flush=True),
            )
        scenarios = []
        if robust_seeds is not None:
            for seed in robust_seeds:
                scenario_source = source if seed == config.seed else RealisticMujocoSource(nominal, replace(config, seed=seed))
                scenarios.append((f'model_seed={seed},noise_seed={seed+303}',
                                  lambda traj, src=scenario_source, s=seed: src.run(traj, periods=periods, seed=s+303)))

        def progress(record):
            detail = (f"通过，实测 cond={record['measured']['condition_number']:.3g}"
                      if record['accepted'] else "淘汰：" + "；".join(record['reasons']))
            print(f"候选 {record['candidate_id']+1}/{len(candidates)}：{detail}", flush=True)

        def save_screening(report):
            report["simulation"] = {"seed": config.seed+303, "recorded_periods": periods,
                                    "warmup_periods": 1, "generation_seed": config.seed+101}
            report['optimization'] = optimization_report
            report['periodic_average'] = periodic_average
            report['robust_seeds'] = robust_seeds
            if output_dir is not None:
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                (out / "screening.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                (out / "screening-config.json").write_text(
                    json.dumps(report["criteria"], ensure_ascii=False, indent=2), encoding="utf-8")
                (out / "config.json").write_text(
                    json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
                (out / 'optimization.json').write_text(
                    json.dumps(optimization_report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

        print("闭环筛选激励轨迹（每条均预热，使用相同场景集合与采样规则）…", flush=True)
        try:
            if scenarios:
                selection = screen_robust_excitation(
                    Hb, candidates, scenarios, preprocess,
                    criteria=screening_criteria, progress=progress,
                    retention_multiplier=periods if periodic_average else 1)
            elif periodic_average:
                current = [None]
                def runner(trajectory):
                    current[0] = trajectory
                    return source.run(trajectory, periods=periods, seed=config.seed+303)
                selection = screen_excitation(
                    Hb, candidates, runner, lambda data: preprocess(data, current[0]),
                    criteria=screening_criteria, progress=progress, retention_multiplier=periods)
            else:
                selection = screen_excitation(
                    Hb, candidates,
                    lambda trajectory: source.run(trajectory, periods=periods, seed=config.seed+303),
                    lambda data: prepare(data, config),
                    criteria=screening_criteria, progress=progress)
        except NoFeasibleExcitation as exc:
            save_screening(exc.report)
            if output_dir is not None:
                failure = {
                    "status": "screening_failed", "config": config.to_dict(),
                    "screening": exc.report,
                    "message": "未产生新的训练/验证数据；目录中若有旧 NPZ，它们不属于本次失败运行。",
                }
                (Path(output_dir) / "report.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            raise
        screening_report = selection.report
        save_screening(screening_report)
        train_traj, train, train_data = selection.trajectory, selection.run, selection.data
        chosen = screening_report["selected_candidate_id"]
        cond = screening_report["candidates"][chosen]["preview"]["condition_number"]
        print(f"选中候选 {chosen+1}，{screening_report['accepted_count']}/{len(candidates)} 条通过筛选。")
    else:
        train_traj, cond = search_excitation(
            Hb, dof, np.random.default_rng(config.seed+101), trials=trials,
            base_freq=setup.train_base_freq, n_harmonics=setup.train_harmonics,
            vel_scale=setup.train_vel_scale, limits=limits,
            velocity_limits=config.velocity_limit, acceleration_limits=config.acceleration_limit,
            refine=setup.search_refine,
        )
        print("采集训练轨迹（含一个预热周期）…", flush=True)
        train = source.run(train_traj, periods=periods, seed=config.seed+303)
        train_data = preprocess(train.measured, train_traj)
    print("采集独立验证轨迹…", flush=True)
    validation = source.run(validation_traj, periods=periods, seed=config.seed+404)
    val_data = preprocess(validation.measured, validation_traj)
    result = identify(Hb, train_data)
    recovered = None
    recovery_payload = None
    if recover:
        print("由辨识基参数恢复逐连杆惯性参数（使用名义模型初值）…", flush=True)
        nominal_full = nominal_recovery_parameters(robot, source.nominal_model, setup.body_names)
        recovered = recover_inertial_parameters(
            robot, result.parms, nominal_parameters=nominal_full, config=recovery_config,
        )
        recovery_payload = recovered.to_dict()
        recovery_payload['prior_source'] = 'source.nominal_model (unperturbed MJCF)'
        recovery_payload['nominal_parameters'] = nominal_full.tolist()
    nominal_parms = parameters(robot, source.nominal_model, setup.body_names)
    plant_parms = parameters(robot, source.model, setup.body_names)

    # truth 仅在辨识结束后评分，不影响预处理、估计或反馈。
    truth = validation.truth
    diagnostics = {
        "status": "inertial_recovery_failed" if recovered is not None and not recovered.success else "completed",
        "robot": setup.name,
        "config": config.to_dict(),
        "experiment": {"trials": trials, "recorded_periods": periods, "warmup_periods": 1,
                       "closed_loop_screening": screen},
        "enhancements": {'optimization': optimize, 'optimization_starts': optimization_starts,
                         'optimization_maxiter': optimization_maxiter, 'robust_seeds': robust_seeds,
                         'periodic_average': periodic_average, 'inertial_recovery': recover},
        "optimization": optimization_report,
        "screening": screening_report,
        "mujoco_version": mujoco.__version__,
        "assumptions": setup.assumptions,
        "preprocessing": "已知整数通道延迟补偿，5 Hz 零相位低通，去两端 0.3 s，剔除任一关节 |dq|<0.08",
        "trajectory_condition_number": {"train": cond, "validation": val_cond},
        "train": train.diagnostics,
        "validation": validation.diagnostics,
        "fit": {
            "rank": result.rank, "n_parms": result.n_parms,
            "identifiable": result.rank == result.n_parms,
            "condition_number": result.condition_number,
            "residual_rms_Nm": result.residual_rms,
            "samples_after_preprocessing": len(train_data),
            "parameters": result.parms.tolist(),
            "base_parameter_expressions": [str(p) for p in dyn.baseparms],
            "relative_error_to_rigid_friction_component": float(np.linalg.norm(result.parms-plant_parms)/np.linalg.norm(plant_parms)),
        },
        "validation_rms_Nm": {
            "measured_fitted": torque_rms(predict_torque(Hb, val_data, result.parms), val_data.tau),
            "measured_nominal": torque_rms(predict_torque(Hb, val_data, nominal_parms), val_data.tau),
            "truth_fitted": torque_rms(predict_torque(Hb, truth, result.parms), truth.tau),
            "truth_nominal": torque_rms(predict_torque(Hb, truth, nominal_parms), truth.tau),
        },
    }
    if recovered is not None:
        scores = {
            'measured_recovered': torque_rms(predict_torque(Hb, val_data, recovered.base_parameters), val_data.tau),
            'truth_recovered': torque_rms(predict_torque(Hb, truth, recovered.base_parameters), truth.tau),
        }
        diagnostics['validation_rms_Nm'].update(scores)
        recovery_payload['validation_rms_Nm'] = scores
        diagnostics['inertial_recovery'] = recovery_payload
    averages = {}
    if periodic_average:
        for key, run, traj in (('train', train, train_traj), ('validation', validation, validation_traj)):
            aligned = align_measurements(run.measured,
                encoder_delay_steps=config.sensors.encoder_delay_steps,
                torque_delay_steps=config.sensors.torque_delay_steps)
            averages[key] = average_periods(aligned, traj.period, cycles=periods)
        diagnostics['period_average'] = {key: value.summary() for key, value in averages.items()}
        diagnostics['preprocessing'] = '已知整数延迟补偿 → 按相位重复周期平均 → 5 Hz 零相位滤波 → 去两端0.3s → 去低速'
    warnings = []
    if recovered is not None and not recovered.success:
        warnings.append("惯性参数恢复未通过收敛、物理可行性或基参数匹配检查；候选参数仅供诊断，原辨识基参数保留")
    if result.rank < result.n_parms:
        warnings.append("训练回归矩阵秩不足，参数不可唯一辨识")
    for title, run in (("训练", train), ("验证", validation)):
        d = run.diagnostics
        if d["contact_fraction"] > 0 or any(d["joint_limit_fraction"]):
            warnings.append(f"{title}存在接触/限位约束，当前自由空间回归模型不包含这些外力")
        if max(d["saturation_fraction"]) > .01:
            warnings.append(f"{title}执行器饱和超过 1%，请检查激励幅度和控制增益")
        if any(d["speed_limit_exceeded"]) or any(d["acceleration_limit_exceeded"]):
            warnings.append(f"{title}实际速度或加速度超过配置的轨迹设计上限")
    diagnostics["warnings"] = warnings
    print(result.summary())
    if recovered is not None:
        print(f"惯性参数恢复状态：{'通过' if recovered.success else '未通过'}；详见 inertial_recovery 报告。")
    for warning in warnings:
        print(f"注意：{warning}")
    for title, run in (("训练", train), ("验证", validation)):
        print(f"{title}跟踪 RMS(rad): {np.round(run.diagnostics['tracking_rms_rad'], 5)}")
        print(f"{title}饱和率: {np.round(run.diagnostics['saturation_fraction'], 4)}；限位率: {run.diagnostics['joint_limit_fraction']}")
    print("独立验证每关节力矩 RMS(N·m):")
    for key, value in diagnostics["validation_rms_Nm"].items():
        print(f"  {key}: {np.round(value, 5)}")
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train.save(out / "train.npz")
        validation.save(out / "validation.npz")
        for key, average in averages.items():
            np.savez_compressed(out / f'{key}-averaged.npz',
                **{field: getattr(average.data, field) for field in ('t','q','dq','ddq','tau')},
                **{f'std_{field}': value for field, value in average.standard_deviation.items()})
        for key, data in (('train', train_data), ('validation', val_data)):
            np.savez_compressed(out / f'{key}-processed.npz',
                **{field: getattr(data, field) for field in ('t','q','dq','ddq','tau')})
        np.savez_compressed(out / "trajectories.npz", train_q0=train_traj.q0,
                            train_a=train_traj.a, train_b=train_traj.b, train_freq=train_traj.base_freq,
                            validation_q0=validation_traj.q0, validation_a=validation_traj.a,
                            validation_b=validation_traj.b, validation_freq=validation_traj.base_freq)
        (out / "report.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "config.json").write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if recovered is not None:
            (out / "inertial-parameters.json").write_text(
                json.dumps(recovery_payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            (out / "recovery-config.json").write_text(
                json.dumps(recovery_config.to_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"数据、配置和报告已保存到 {out.resolve()}")
    return diagnostics
