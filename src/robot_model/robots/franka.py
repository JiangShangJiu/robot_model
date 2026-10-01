"""Franka Panda 臂 MDH 标准用例（不含手爪）。

参数在经典 7 轴 MDH 基础上，将末连杆 ``d`` 设为 ``0``，使第 7 帧落在
MuJoCo ``link7``，而不是法兰/hand（``d=0.107``）。

真值：``assets/models/franka_panda/panda_arm.xml``（无夹爪，对齐 MDH→link7）。
完整带手爪模型仍保留为 ``panda.xml``。

两个无夹爪模型分工不同，连杆惯性与关节限位完全相同：

- ``panda_arm.xml``：position servo 执行器，用于运动学/逆动力学对拍与查看器。
- ``panda_arm_motor.xml``：每关节一个 ``gear=1`` 力矩 motor，供
  :class:`~robot_model.simulation.RealisticMujocoSource` 的闭环实机近似使用。

帧对应：
- MDH link i (i=1..7) ↔ MuJoCo body ``link{i}``（位置 + 姿态）
"""

from __future__ import annotations

from pathlib import Path

# (alpha, a, d, theta)；'q' 表示关节变量
# 末项 d=0：只到 link7，不考虑 hand / 夹爪
PANDA_MDH_PARMS = [
    ("0", "0", "0.333", "q"),
    ("-pi/2", "0", "0", "q"),
    ("pi/2", "0", "0.316", "q"),
    ("pi/2", "0.0825", "0", "q"),
    ("-pi/2", "-0.0825", "0.384", "q"),
    ("pi/2", "0", "0", "q"),
    ("pi/2", "0.088", "0", "q"),
]

PANDA_MDH_BODY_NAMES = [f"link{i}" for i in range(1, 8)]

# 关节限位，与两个无夹爪 MJCF 的 jnt_range 一致（由测试断言）。
# 关节 4 的区间完全在负半轴，关节 6 的中点在 1.87；轨迹中心位置必须由限位算出，
# 不能默认 0。
PANDA_JOINT_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
PANDA_JOINT_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)


def repo_root() -> Path:
    """本包根目录（含 ``assets/models/franka_panda``）。"""
    return Path(__file__).resolve().parents[1]


def panda_xml_path() -> Path:
    """无夹爪 7 轴模型（与 MDH 一致）。"""
    arm = repo_root() / "assets" / "models" / "franka_panda" / "panda_arm.xml"
    if arm.is_file():
        return arm
    return repo_root() / "assets" / "models" / "franka_panda" / "panda.xml"


def panda_motor_xml_path() -> Path:
    """力矩 motor 版 7 轴模型，用于闭环实机近似实验。"""
    return repo_root() / "assets" / "models" / "franka_panda" / "panda_arm_motor.xml"
