"""空间 3R 拟人臂：MDH 参数与 MuJoCo 真值模型。

用于完整符号推导、参数辨识和 MuJoCo 对照。三轴模型的计算量较小，
可以快速检查回归矩阵、基参数和导出代码；对应模型位于
``assets/models/rrr_arm/rrr_arm.xml``。

构型（腰 + 肩 + 肘）：关节 1 绕竖直轴，关节 2、3 绕水平轴。
关节 1 轴与重力平行，故重力对 tau_1 无贡献——这是拟人臂的固有性质，不是缺陷。

帧对应：MDH link i (i=1..3) ↔ MuJoCo body ``link{i}``（位置 + 姿态）。
XML 里 body 的 pos/quat 就是按下表算的，改一处必须同步另一处。
"""

from __future__ import annotations

from pathlib import Path

# (alpha_{i-1}, a_{i-1}, d_i, theta_i)；'q' 表示关节变量
# 第 3 行 a、d 同时非零（肘部横向偏置），避免参数表过于平凡
RRR_MDH_PARMS = [
    ("0", "0", "0.30", "q"),
    ("-pi/2", "0", "0", "q"),
    ("0", "0.28", "0.05", "q"),
]

RRR_BODY_NAMES = [f"link{i}" for i in range(1, 4)]

# 末端相对 link3 帧的位置（XML 里的 site "tool"）
RRR_TOOL_OFFSET = (0.22, 0.0, 0.0)


def rrr_xml_path() -> Path:
    """MuJoCo 真值模型路径。"""
    root = Path(__file__).resolve().parents[1]
    return root / "assets" / "models" / "rrr_arm" / "rrr_arm.xml"
