"""可复用的机器人定义；MDH 参数和模型资源路径集中于此。"""

from .franka import (
    PANDA_JOINT_LOWER,
    PANDA_JOINT_UPPER,
    PANDA_MDH_BODY_NAMES,
    PANDA_MDH_PARMS,
    panda_motor_xml_path,
    panda_xml_path,
)
from .rrr import RRR_BODY_NAMES, RRR_MDH_PARMS, RRR_TOOL_OFFSET, rrr_xml_path

__all__ = [
    "PANDA_JOINT_LOWER", "PANDA_JOINT_UPPER",
    "PANDA_MDH_BODY_NAMES", "PANDA_MDH_PARMS", "panda_motor_xml_path", "panda_xml_path",
    "RRR_BODY_NAMES", "RRR_MDH_PARMS", "RRR_TOOL_OFFSET", "rrr_xml_path",
]
