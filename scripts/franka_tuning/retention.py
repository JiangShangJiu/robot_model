"""阶段 5 的一次性诊断：找到能同时满足跟踪与样本保留率的激励幅度。

保留率由 drop_low_speed(|dq|<0.08 任一关节即丢弃) 决定，7 轴比 3 轴难得多。
用法：PYTHONPATH=src python scripts/franka_tuning/retention.py
"""

import sys

import mujoco
import numpy as np

from robot_model.excitation import generate_excitation_candidates
from robot_model.experiments.franka import default_config, prepare
from robot_model.robots import PANDA_JOINT_LOWER, PANDA_JOINT_UPPER, panda_motor_xml_path
from robot_model.simulation import RealisticMujocoSource

# 直接运行此脚本时，Python 会把脚本所在目录加入搜索路径。
from sweep import check


def main():
    model = mujoco.MjModel.from_xml_path(str(panda_motor_xml_path()))
    limits = (np.array(PANDA_JOINT_LOWER), np.array(PANDA_JOINT_UPPER))
    config = default_config()
    source = RealisticMujocoSource(model, config)
    print(f"{'vel_scale':>9} {'cand':>4} {'retain':>7} {'kept':>5} {'maxRMS':>7} {'max|dq|':>8}  判定")
    for vel_scale in (0.4, 0.6, 0.8, 1.0, 1.2):
        trajectories = generate_excitation_candidates(
            7, np.random.default_rng(11), trials=3, limits=limits,
            velocity_limits=config.velocity_limit,
            acceleration_limits=config.acceleration_limit,
            vel_scale=vel_scale,
        )
        for k, trajectory in enumerate(trajectories):
            run = source.run(trajectory, periods=1, seed=303)
            try:
                kept = len(prepare(run.measured, config))
            except ValueError:
                kept = 0
            retention = kept / len(run.measured)
            d = run.diagnostics
            reasons = check(d, config)
            print(f"{vel_scale:>9} {k:>4} {retention:>7.3f} {kept:>5} "
                  f"{max(d['tracking_rms_rad']):>7.4f} {max(d['max_speed_rad_s']):>8.2f}  "
                  f"{'稳定' if not reasons else '，'.join(reasons)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
