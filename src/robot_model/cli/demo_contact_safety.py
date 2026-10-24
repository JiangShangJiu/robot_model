"""演示 2 入口：接触力控失稳回放（无/有能量罐）。

用法（仓库根目录）::

    robot-demo-contact --output-dir artifacts/contact_safety
    # 或 PYTHONPATH=src python -m robot_model.cli.demo_contact_safety
"""

from __future__ import annotations

import argparse

from robot_model.demos.contact_safety import ContactDemoConfig, run_demo


def main(argv=None) -> None:
    defaults = ContactDemoConfig()  # CLI 默认值始终取自 dataclass，避免漂移
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="artifacts/contact_safety")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--force-target", type=float, default=defaults.force_target_N,
                        help="法向目标力 N")
    parser.add_argument("--force-delay-ms", type=int, default=defaults.force_delay_steps,
                        help="力测量延迟毫秒（1 kHz 控制）")
    parser.add_argument("--ki-force", type=float, default=defaults.ki_force,
                        help="导纳力积分增益 m/(N·s)；默认故意调高以展示失稳")
    parser.add_argument("--duration", type=float, default=defaults.duration_s,
                        help="每种设置的仿真时长 s")
    args = parser.parse_args(argv)
    run_demo(args.output_dir, ContactDemoConfig(
        seed=args.seed, force_target_N=args.force_target,
        force_delay_steps=args.force_delay_ms, ki_force=args.ki_force,
        duration_s=args.duration,
    ))


if __name__ == "__main__":
    main()
