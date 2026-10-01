"""阶段 2 的一次性调参脚本：验证 Franka 闭环控制参数的稳定性。

记录当时如何选择 default_config() 的控制增益，供复查历史调参过程；
不属于库代码，也不进自动化测试。用法（仓库根目录）：

    PYTHONPATH=src python scripts/franka_tuning/sweep.py
"""

import sys
from itertools import product

import mujoco
import numpy as np

from robot_model.excitation import generate_excitation_candidates
from robot_model.robots import panda_motor_xml_path
from robot_model.simulation import ActuatorConfig, RealismConfig, RealisticMujocoSource

# 阶段 2 的验收门槛
MAX_TRACKING_RMS = 0.02
MAX_TRACKING_ERROR = 0.08

GAIN_SETS = {
    "A": ((120.,) * 4 + (40., 40., 15.), (12.,) * 4 + (4., 4., 1.5)),
    "A2x": ((240.,) * 4 + (80., 80., 30.), (24.,) * 4 + (8., 8., 3.)),
    "Ahalf": ((60.,) * 4 + (20., 20., 7.5), (6.,) * 4 + (2., 2., 0.75)),
}
VEL_SCALES = (0.3, 0.4, 0.5)
SEEDS = (0, 1)


def make_config(model, kp, kd, seed):
    return RealismConfig(
        seed=seed,
        physics_dt=0.001,
        control_dt=0.004,
        kp=kp,
        kd=kd,
        damping=(1.0,) * 7,
        frictionloss=(0.30,) * 7,
        armature=(0.10,) * 7,
        joint_lower=tuple(model.jnt_range[:, 0]),
        joint_upper=tuple(model.jnt_range[:, 1]),
        velocity_limit=(2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5),
        acceleration_limit=(4.0,) * 7,
        actuator=ActuatorConfig(
            torque_limit=(87., 87., 87., 87., 12., 12., 12.),
            slew_rate=(1000.,) * 7,
        ),
    )


def check(diagnostics, config):
    """返回不满足阶段 2 门槛的原因列表。"""
    d, reasons = diagnostics, []
    if max(d["tracking_rms_rad"]) > MAX_TRACKING_RMS:
        reasons.append(f"跟踪RMS {max(d['tracking_rms_rad']):.4f}")
    if max(d["tracking_max_rad"]) > MAX_TRACKING_ERROR:
        reasons.append(f"最大跟踪误差 {max(d['tracking_max_rad']):.4f}")
    if max(d["saturation_fraction"]) > 0:
        reasons.append(f"饱和 {max(d['saturation_fraction']):.3f}")
    if max(d["slew_limit_fraction"]) > 0:
        reasons.append(f"变化率受限 {max(d['slew_limit_fraction']):.3f}")
    if d["contact_fraction"] > 0:
        reasons.append(f"接触 {d['contact_fraction']:.3f}")
    if max(d["joint_limit_fraction"]) > 0:
        reasons.append(f"限位 {max(d['joint_limit_fraction']):.3f}")
    if any(d["speed_limit_exceeded"]):
        reasons.append("速度超设计上限")
    if any(d["acceleration_limit_exceeded"]):
        reasons.append("加速度超设计上限")
    return reasons


def main():
    model = mujoco.MjModel.from_xml_path(str(panda_motor_xml_path()))
    limits = (model.jnt_range[:, 0].copy(), model.jnt_range[:, 1].copy())
    passed = []
    for name, vel_scale in product(GAIN_SETS, VEL_SCALES):
        kp, kd = GAIN_SETS[name]
        trajectories = generate_excitation_candidates(
            7, np.random.default_rng(11), trials=3, limits=limits,
            velocity_limits=(2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5),
            acceleration_limits=(4.0,) * 7, vel_scale=vel_scale,
        )
        failures = []
        worst_rms = 0.0
        for seed in SEEDS:
            source = RealisticMujocoSource(model, make_config(model, kp, kd, seed))
            for k, trajectory in enumerate(trajectories):
                run = source.run(trajectory, periods=1, seed=seed + 303)
                worst_rms = max(worst_rms, max(run.diagnostics["tracking_rms_rad"]))
                reasons = check(run.diagnostics, source.config)
                if reasons:
                    failures.append(f"seed={seed} traj={k}: " + "，".join(reasons))
        tag = "通过" if not failures else f"{len(failures)}/{len(SEEDS)*len(trajectories)} 条失败"
        print(f"kp={name:6s} vel_scale={vel_scale}: {tag}，最差跟踪RMS={worst_rms:.4f}", flush=True)
        for line in failures[:3]:
            print(f"    {line}", flush=True)
        if not failures:
            passed.append((name, vel_scale, worst_rms))
    print()
    if not passed:
        print("没有配置通过阶段 2 门槛")
        return 1
    print("通过的配置（按最差跟踪 RMS 排序）：")
    for name, vel_scale, worst_rms in sorted(passed, key=lambda row: row[2]):
        print(f"  kp={name} vel_scale={vel_scale} 最差跟踪RMS={worst_rms:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
