"""阶段 5 诊断：用 Franka 规格级的速度/加速度上限，看实际包络与保留率。

Franka Research 3 / Panda 公开的关节上限（rad/s, rad/s^2）用于设计轨迹；
闭环实际值仍需实测，过零点的摩擦抖动会让瞬时加速度高于轨迹设计值。
用法：PYTHONPATH=src python scripts/franka_tuning/envelope.py
"""

import sys
from dataclasses import replace

import mujoco
import numpy as np

from robot_model.excitation import generate_excitation_candidates
from robot_model.experiments.franka import default_config, prepare
from robot_model.robots import PANDA_JOINT_LOWER, PANDA_JOINT_UPPER, panda_motor_xml_path
from robot_model.simulation import RealisticMujocoSource

VELOCITY = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
ACCELERATION = (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0)


def main():
    model = mujoco.MjModel.from_xml_path(str(panda_motor_xml_path()))
    limits = (np.array(PANDA_JOINT_LOWER), np.array(PANDA_JOINT_UPPER))
    config = replace(default_config(), velocity_limit=VELOCITY, acceleration_limit=ACCELERATION)
    source = RealisticMujocoSource(model, config)
    for vel_scale in (0.8, 1.0, 1.2):
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
            d = run.diagnostics
            print(f"vel_scale={vel_scale} cand{k}: retain={kept/len(run.measured):.3f} kept={kept} "
                  f"trackRMS={max(d['tracking_rms_rad']):.4f} sat={max(d['saturation_fraction']):.3f}")
            print(f"   max|dq| ={[round(v, 2) for v in d['max_speed_rad_s']]} 超限={d['speed_limit_exceeded']}")
            print(f"   max|ddq|={[round(v, 1) for v in d['max_acceleration_rad_s2']]} 超限={d['acceleration_limit_exceeded']}",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
