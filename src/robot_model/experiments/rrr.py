"""RRR 实机近似辨识实验，入口：identify_rrr --realistic。

流程编排在 :mod:`robot_model.experiments.realistic`，本模块只提供 RRR 的
实验假设：仿真配置与 :class:`RealisticSetup` 描述。
"""

from robot_model.experiments.realistic import RealisticSetup, prepare, run_experiment as _run
from robot_model.robots.rrr import RRR_BODY_NAMES, RRR_MDH_PARMS, rrr_xml_path
from robot_model.simulation import ActuatorConfig, RealismConfig

__all__ = ["default_config", "prepare", "run_experiment", "setup"]


def default_config(seed=0):
    """小型 RRR 自由空间实验假设；不是任何商业机械臂的实测参数。"""
    return RealismConfig(
        seed=seed, kp=(25., 35., 12.), kd=(1.2, 1.8, 0.4),
        damping=(0.30, 0.25, 0.15), frictionloss=(0.20, 0.15, 0.10),
        armature=(0.05, 0.04, 0.02),
        joint_lower=(-2.5, -1.8, -2.2), joint_upper=(2.5, 1.8, 2.2),
        velocity_limit=(1.5, 1.5, 1.8), acceleration_limit=(5., 5., 6.),
        actuator=ActuatorConfig(torque_limit=(15., 20., 8.), slew_rate=(300., 400., 200.)),
    )


def setup():
    return RealisticSetup(
        name="rrr_realistic",
        mdh_parms=RRR_MDH_PARMS,
        body_names=RRR_BODY_NAMES,
        xml_path=rrr_xml_path(),
        assumptions="RRR 自由空间、刚性传动；示例参数未实机标定。无背隙/柔性/电气/热模型。",
        validation_base_freq=0.19,
        validation_harmonics=4,
        validation_vel_scale=0.85,
        train_base_freq=0.15,
        train_harmonics=4,
        train_vel_scale=1.0,
    )


def run_experiment(config, **kwargs):
    return _run(setup(), config, **kwargs)
