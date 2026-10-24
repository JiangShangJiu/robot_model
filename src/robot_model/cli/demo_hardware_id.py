"""演示 1 入口：仿真实机辨识闭环与辨识前后对照。

用法（仓库根目录）::

    robot-demo-hardware-id --output-dir artifacts/hardware_id
    # 或 PYTHONPATH=src python -m robot_model.cli.demo_hardware_id
"""

from __future__ import annotations

import argparse

from robot_model.demos.hardware_id import HardwareIdDemoConfig, run_demo


def main(argv=None) -> None:
    defaults = HardwareIdDemoConfig()  # CLI 默认值始终取自 dataclass，避免漂移
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="artifacts/hardware_id",
                        help="产物目录（metrics.json 与 PNG）")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--trials", type=int, default=defaults.trials,
                        help="激励轨迹随机搜索条数")
    parser.add_argument("--periods", type=int, default=defaults.periods, help="记录周期数")
    parser.add_argument("--payload-mass", type=float, default=defaults.payload_mass_kg,
                        help="对象侧末端负载质量 kg；0 关闭负载失配")
    parser.add_argument("--feedback-scale", type=float, default=defaults.feedback_scale,
                        help="控制对照的反馈增益缩放（弱反馈放大前馈差异）")
    args = parser.parse_args(argv)
    run_demo(args.output_dir, HardwareIdDemoConfig(
        seed=args.seed, trials=args.trials, periods=args.periods,
        payload_mass_kg=args.payload_mass, feedback_scale=args.feedback_scale,
    ))


if __name__ == "__main__":
    main()
