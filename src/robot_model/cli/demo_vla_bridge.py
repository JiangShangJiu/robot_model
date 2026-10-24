"""演示 3 入口：VLA 动作块 → 底层阻抗环桥接（延迟/丢包 + 护栏）。

用法（仓库根目录）::

    robot-demo-vla --output-dir artifacts/vla_bridge
    # 或 PYTHONPATH=src python -m robot_model.cli.demo_vla_bridge
"""

from __future__ import annotations

import argparse

from robot_model.demos.vla_bridge import VlaBridgeDemoConfig, run_demo


def main(argv=None) -> None:
    defaults = VlaBridgeDemoConfig()  # CLI 默认值始终取自 dataclass，避免漂移
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="artifacts/vla_bridge")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--latency", type=float, default=defaults.latency_s,
                        help="动作块推理+通信延迟 s（退化场景）")
    parser.add_argument("--dropout", type=float, default=defaults.dropout_prob,
                        help="动作块随机丢失概率（退化场景）")
    parser.add_argument("--duration", type=float, default=defaults.duration_s,
                        help="每个场景的仿真时长 s")
    args = parser.parse_args(argv)
    run_demo(args.output_dir, VlaBridgeDemoConfig(
        seed=args.seed, latency_s=args.latency, dropout_prob=args.dropout,
        duration_s=args.duration,
    ))


if __name__ == "__main__":
    main()
