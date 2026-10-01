"""实机近似辨识实验的通用入口：对象/控制器分离的闭环采集与辨识。

用法（先在仓库根目录 pip install -e ".[sim,ident]"）::

    robot-identify --robot rrr --trials 6 --periods 2 --output-dir ./run
    robot-identify --robot franka --trials 12 --periods 2 \
        --config configs/franka/realistic.json --output-dir ./run

各机器人的实验假设在 ``experiments/<机器人>.py``，流程编排共用
``experiments/realistic.py``。``identify_rrr`` 的理想/噪声/伺服对照演示不在这里。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

ROBOTS = ("rrr", "franka")


def _load_robot(name):
    from importlib import import_module

    module = import_module(f"robot_model.experiments.{name}")
    return module.default_config, module.run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="实机近似闭环辨识实验")
    parser.add_argument("--robot", choices=ROBOTS, required=True, help="实验对象")
    parser.add_argument("--config", type=Path, help="仿真配置 JSON（由上次运行生成）")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    parser.add_argument("--trials", type=int, default=12, help="激励候选数")
    parser.add_argument("--periods", type=int, default=2, help="每条轨迹的记录周期数")
    parser.add_argument("--output-dir", type=Path, help="保存配置、NPZ 数据与 JSON 报告")
    parser.add_argument("--no-screen-excitation", action="store_true",
                        help="关闭默认的闭环激励筛选，仅按解析条件数搜索")
    parser.add_argument("--screening-config", type=Path, help="激励筛选阈值 JSON")
    parser.add_argument("--optimize-excitation", action=argparse.BooleanOptionalAction, default=None,
                        help="对最佳解析候选做 SLSQP 约束优化（RRR 默认开启，Franka 默认关闭；可显式覆盖）")
    parser.add_argument("--optimization-starts", type=int, default=2, help="优化初值数量")
    parser.add_argument("--optimization-maxiter", type=int, default=20, help="每个初值的局部优化迭代上限")
    parser.add_argument("--robust-seeds", type=int, nargs="+",
                        help="模型与噪声场景种子，至少两个，首个须等于配置 seed")
    parser.add_argument("--average-periods", action="store_true",
                        help="启用重复周期相位平均，至少记录两周期")
    parser.add_argument("--recover-inertia", action="store_true",
                        help="辨识后以名义模型为初值，恢复物理可行的逐连杆质量、质心和惯性张量")
    parser.add_argument("--recovery-config", type=Path, help="惯性参数恢复约束 JSON，需要 --recover-inertia")
    args = parser.parse_args()
    optimize = (args.robot == "rrr" if args.optimize_excitation is None
                else args.optimize_excitation)
    if args.recovery_config and not args.recover_inertia:
        parser.error("--recovery-config 需要 --recover-inertia")
    if args.no_screen_excitation and args.screening_config:
        parser.error("--no-screen-excitation 不能与 --screening-config 同时使用")

    from robot_model.excitation import ExcitationCriteria, NoFeasibleExcitation
    from robot_model.identification import RecoveryConfig
    from robot_model.simulation import RealismConfig

    default_config, run_experiment = _load_robot(args.robot)
    config = (RealismConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
              if args.config else default_config())
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    criteria = (ExcitationCriteria(**json.loads(args.screening_config.read_text(encoding="utf-8")))
                if args.screening_config else None)
    try:
        recovery_config = (RecoveryConfig(**json.loads(args.recovery_config.read_text(encoding="utf-8")))
                           if args.recovery_config else None)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(f"无效的惯性参数恢复配置：{exc}")

    try:
        report = run_experiment(
            config, trials=args.trials, periods=args.periods, output_dir=args.output_dir,
            screen=not args.no_screen_excitation, screening_criteria=criteria,
            optimize=optimize, optimization_starts=args.optimization_starts,
            optimization_maxiter=args.optimization_maxiter,
            robust_seeds=args.robust_seeds, periodic_average=args.average_periods,
            recover=args.recover_inertia, recovery_config=recovery_config,
        )
    except NoFeasibleExcitation as exc:
        location = (f"；报告：{args.output_dir.resolve() / 'screening.json'}" if args.output_dir else "")
        parser.exit(2, f"{exc}{location}\n")
    if report["status"] == "inertial_recovery_failed":
        location = (f"；报告：{args.output_dir.resolve() / 'inertial-parameters.json'}"
                    if args.output_dir else "")
        parser.exit(3, f"惯性参数恢复未通过验收，原辨识基参数保留{location}\n")


if __name__ == "__main__":
    main()
