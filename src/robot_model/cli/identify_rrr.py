"""RRR 三轴臂动力学参数辨识全流程演示。

两条路径并排对比：
  1. 理想：解析轨迹 + ``mj_inverse``（验算法，无噪声可到 1e-15）
  2. 闭环：PD 跟踪 + 传感器 + 数值微分（贴近真机，约 1e-3）

用法（先在仓库根目录 pip install -e ".[sim,ident]"）::

    python -m robot_model.cli.identify_rrr
    python -m robot_model.cli.identify_rrr --noise
    python -m robot_model.cli.identify_rrr --servo
    python -m robot_model.cli.identify_rrr --realistic --output-dir ./rrr_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sympy


from robot_model import Robot
from robot_model.data import add_noise, drop_low_speed, filter_measurements, sample_trajectory
from robot_model.excitation import search_excitation
from robot_model.identification import identify, parameter_error
from robot_model.simulation import TYPICAL_SERVO, MujocoTrackingSource
from robot_model.simulation.reference import MujocoReference
from robot_model.robots.rrr import (
    RRR_BODY_NAMES,
    RRR_MDH_PARMS,
    rrr_xml_path,
)


def _build_regressor():
    robot = Robot.from_mdh("rrr_arm", RRR_MDH_PARMS)
    dyn = robot.dynamics
    dyn.gen_regressor()
    dyn.calc_base_parms()
    Hb = dyn.gen_base_regressor()

    s = robot.symbols
    f = sympy.lambdify(list(s.q) + list(s.dq) + list(s.ddq), Hb, "numpy")

    def Hb_func(q, dq, ddq):
        return np.asarray(f(*(list(q) + list(dq) + list(ddq))), dtype=float)

    return robot, Hb_func


def _true_base_parms(robot, ref: MujocoReference) -> np.ndarray:
    s = robot.symbols
    return np.asarray(
        sympy.lambdify(
            list(s.dynparms()), sympy.Matrix(robot.dynamics.baseparms), "numpy"
        )(*ref.dynparms()),
        dtype=float,
    ).ravel()


def _report(title: str, result, pi_b: np.ndarray) -> None:
    err = parameter_error(result.parms, pi_b)
    print(f"\n=== {title} ===")
    print(result.summary())
    print(
        f"相对误差 {err['rel_norm']:.3e}，"
        f"最大分量误差 {err['abs_max']:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RRR 臂辨识演示")
    parser.add_argument(
        "--noise",
        action="store_true",
        help="在理想路径上加 0.05 N·m 力矩噪声",
    )
    parser.add_argument(
        "--servo",
        action="store_true",
        help="闭环路径使用 TYPICAL_SERVO 测量链失真",
    )
    parser.add_argument("--realistic", action="store_true", help="启用对象/控制器分离的实机近似闭环")
    parser.add_argument("--config", type=Path, help="实机近似配置 JSON（由上次运行生成）")
    parser.add_argument("--seed", type=int, default=None, help="覆盖实机近似随机种子")
    parser.add_argument("--trials", type=int, default=12, help="实机近似激励搜索候选数")
    parser.add_argument("--periods", type=int, default=2, help="实机近似每条轨迹的记录周期数")
    parser.add_argument("--output-dir", type=Path, help="保存实机近似配置、NPZ 数据与 JSON 报告")
    parser.add_argument("--no-screen-excitation", action="store_true", help="关闭实机近似模式默认的闭环激励筛选，仅按解析条件数搜索")
    parser.add_argument("--screening-config", type=Path, help="激励筛选阈值 JSON")
    parser.add_argument('--optimize-excitation', action=argparse.BooleanOptionalAction,
                        default=True, help='对最佳解析候选做 SLSQP 约束优化（默认开启，可用 --no-optimize-excitation 关闭）')
    parser.add_argument('--optimization-starts', type=int, default=2, help='优化初值数量')
    parser.add_argument('--optimization-maxiter', type=int, default=20, help='每个初值的局部优化迭代上限')
    parser.add_argument('--robust-seeds', type=int, nargs='+', help='模型与噪声场景种子，至少两个，首个须等于配置 seed')
    parser.add_argument('--average-periods', action='store_true', help='启用重复周期相位平均，至少记录两周期')
    parser.add_argument('--recover-inertia', action='store_true', help='辨识后以名义模型为初值，恢复物理可行的逐连杆质量、质心和惯性张量')
    parser.add_argument('--recovery-config', type=Path, help='惯性参数恢复约束 JSON，需要 --recover-inertia')
    args = parser.parse_args()
    if args.recovery_config and not args.recover_inertia:
        parser.error('--recovery-config 需要 --recover-inertia')
    if (args.recover_inertia or args.recovery_config) and not args.realistic:
        parser.error('惯性参数恢复仅用于 --realistic 模式')
    import mujoco

    if args.realistic:
        if args.noise or args.servo:
            parser.error("--realistic 已包含在线测量误差，不与 --noise/--servo 叠加；请用 --config 调整")
        import json
        from dataclasses import replace
        from robot_model.simulation import RealismConfig
        from robot_model.identification import RecoveryConfig
        from robot_model.excitation import ExcitationCriteria, NoFeasibleExcitation
        from robot_model.experiments.rrr import default_config, run_experiment

        config = (RealismConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
                  if args.config else default_config())
        if args.seed is not None:
            config = replace(config, seed=args.seed)
        if args.no_screen_excitation and args.screening_config:
            parser.error("--no-screen-excitation 不能与 --screening-config 同时使用")
        criteria = (ExcitationCriteria(**json.loads(args.screening_config.read_text(encoding="utf-8")))
                    if args.screening_config else None)
        try:
            recovery_config = (RecoveryConfig(**json.loads(args.recovery_config.read_text(encoding="utf-8")))
                               if args.recovery_config else None)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(f"无效的惯性参数恢复配置：{exc}")
        try:
            report = run_experiment(config, trials=args.trials, periods=args.periods, output_dir=args.output_dir,
                           screen=not args.no_screen_excitation, screening_criteria=criteria,
                           optimize=args.optimize_excitation, optimization_starts=args.optimization_starts,
                           optimization_maxiter=args.optimization_maxiter,
                           robust_seeds=args.robust_seeds, periodic_average=args.average_periods,
                           recover=args.recover_inertia, recovery_config=recovery_config)
        except NoFeasibleExcitation as exc:
            location = (f"；报告：{args.output_dir.resolve() / 'screening.json'}" if args.output_dir else "")
            parser.exit(2, f"{exc}{location}\n")
        if report['status'] == 'inertial_recovery_failed':
            location = (f"；报告：{args.output_dir.resolve() / 'inertial-parameters.json'}" if args.output_dir else "")
            parser.exit(3, f"惯性参数恢复未通过验收，原辨识基参数保留{location}\n")
        return
    if (args.config or args.output_dir or args.seed is not None or args.no_screen_excitation
            or args.screening_config or args.robust_seeds or args.average_periods
            or args.optimization_starts != 2 or args.optimization_maxiter != 20):
        parser.error("仿真配置、筛选配置和输出参数仅用于 --realistic 模式")

    rng = np.random.default_rng(0)
    robot, Hb_func = _build_regressor()
    model = mujoco.MjModel.from_xml_path(str(rrr_xml_path()))
    ref = MujocoReference(model, RRR_BODY_NAMES)
    pi_b = _true_base_parms(robot, ref)

    print(f"基参数维数 {pi_b.size}，搜索激励轨迹…")
    traj, cond = search_excitation(
        Hb_func, 3, rng, trials=20, vel_scale=1.2
    )
    print(f"选中轨迹 cond(W) = {cond:.3g}")

    # --- 路径 1：理想 ---
    data = sample_trajectory(traj, ref.inverse_dynamics, rate=100)
    if args.noise:
        data = add_noise(data, rng, tau_std=0.05)
    _report(
        "理想路径（mj_inverse" + (" + 噪声" if args.noise else "") + ")",
        identify(Hb_func, data),
        pi_b,
    )

    # --- 路径 2：闭环采集 ---
    source = MujocoTrackingSource(model, kp=50.0, kd=2.0)
    sensor_model = TYPICAL_SERVO if args.servo else None
    tracked = source.run(
        traj,
        periods=1,
        decimate=5,
        cutoff=10.0,
        sensor_model=sensor_model,
    )
    tracked = filter_measurements(tracked, cutoff=5.0)
    tracked = drop_low_speed(tracked, 0.05)
    _report(
        "闭环路径（跟踪 + 传感器 + 微分"
        + (" + TYPICAL_SERVO" if args.servo else "")
        + ")",
        identify(Hb_func, tracked),
        pi_b,
    )


if __name__ == "__main__":
    main()
