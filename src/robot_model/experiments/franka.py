"""Franka Panda 实机近似辨识实验的配置。

对象是 ``panda_arm_motor.xml``（每关节一个 gear=1 力矩 motor）；前馈只用未扰动
的名义模型，对象真值参数仅参与最后评分。

力矩上限与关节速度上限取 Franka 公开规格的量级，关节限位取自 MJCF；
控制增益、摩擦、低速摩擦峰、外扰与传感器参数都是仿真实验假设，
没有经过实机标定，不应当作工业伺服规格。
"""

from robot_model.experiments.realistic import RealisticSetup, prepare, run_experiment as _run
from robot_model.robots.franka import (
    PANDA_JOINT_LOWER,
    PANDA_JOINT_UPPER,
    PANDA_MDH_BODY_NAMES,
    PANDA_MDH_PARMS,
    panda_motor_xml_path,
)
from robot_model.simulation import ActuatorConfig, RealismConfig

__all__ = ["default_config", "prepare", "run_experiment", "setup"]


def default_config(seed=0):
    """Franka 7 轴自由空间实验假设。

    增益由 ``scripts/franka_tuning/sweep.py`` 扫描确定：3 个种子 × 3 条候选轨迹
    全部满足跟踪 RMS ≤0.02 rad、零饱和、零接触、零限位，最差 0.009 rad。
    不要整体放大增益——``ActuatorConfig`` 的 8 ms 一阶滞后加 2 步通信延迟下，
    kp 加倍即发散（饱和率超过 80%，并出现自碰撞）；提高控制频率不解决滞后。

    关节速度与加速度上限取公开规格值。早先自设的 4 rad/s² 会成为激励设计的
    瓶颈：轨迹幅值被压低后，预处理的低速剔除几乎丢光样本（保留率个位数）。
    """
    return RealismConfig(
        seed=seed,
        physics_dt=0.001,
        control_dt=0.004,
        kp=(120., 180., 120., 180., 80., 80., 40.),
        kd=(12., 18., 12., 18., 8., 8., 4.),
        damping=(1.0,) * 7,
        frictionloss=(0.30,) * 7,
        armature=(0.10,) * 7,
        joint_lower=PANDA_JOINT_LOWER,
        joint_upper=PANDA_JOINT_UPPER,
        # 轨迹设计上限取公开规格的关节速度与加速度上限
        velocity_limit=(2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61),
        acceleration_limit=(15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0),
        actuator=ActuatorConfig(
            torque_limit=(87., 87., 87., 87., 12., 12., 12.),
            slew_rate=(1000.,) * 7,
        ),
    )


def setup():
    return RealisticSetup(
        name="panda_realistic",
        mdh_parms=PANDA_MDH_PARMS,
        body_names=PANDA_MDH_BODY_NAMES,
        xml_path=panda_motor_xml_path(),
        assumptions=(
            "Franka Panda 7 轴自由空间、刚性传动，只到 link7（无 hand/负载）。"
            "力矩与速度上限取公开规格量级，增益与摩擦为仿真假设，未实机标定。"
            "无背隙/柔性/电气/热模型。"
        ),
        # 7 轴走数值回归（dof>=6 自动启用），采样数与 RRR 一致即可满秩
        base_parms_samples=400,
        # 幅值由样本保留率决定，不是由跟踪误差决定：预处理要求所有 7 个关节同时
        # |dq|>=0.08，低幅轨迹的保留率只有个位数百分比。实测 0.4 保留 0~14%，
        # 1.2 保留 23~43%，而跟踪 RMS 在整个区间都是 0.006 左右、零饱和。
        candidate_vel_scale=1.2,
        validation_base_freq=0.19,
        validation_harmonics=4,
        validation_vel_scale=1.0,
        train_base_freq=0.15,
        train_harmonics=4,
        train_vel_scale=1.2,
        # 数值回归已合批提速；仍保留展示实验的随机搜索设置。
        # 高维 SLSQP 预算需独立验证，search_refine 只控制搜索内部精炼。
        search_refine=False,
    )


def run_experiment(config, **kwargs):
    return _run(setup(), config, **kwargs)
