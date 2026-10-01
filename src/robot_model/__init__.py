"""robot_model：符号运动学 / 动力学。

数据流::

    RobotSymbols + Description(DH/PoE)
            │
            ▼
       FrameChain          # T, T_rel, S_body
            │
     ┌──────┴──────┐
     ▼             ▼
 Kinematics     Dynamics   # J            # τ, M, H
"""

from .dynamics import Dynamics
from .geometry import (
    DHParams,
    Description,
    FrameChain,
    ModifiedDH,
    PoEParams,
    StandardDH,
    get_convention,
    normalize_convention,
    normalize_dh_convention,
)
from .kinematics import Kinematics
from .robot import Robot
from .utils import RobotSymbols

__all__ = [
    "Robot",
    "RobotSymbols",
    "Description",
    "DHParams",
    "PoEParams",
    "FrameChain",
    "Kinematics",
    "Dynamics",
    "StandardDH",
    "ModifiedDH",
    "get_convention",
    "normalize_convention",
    "normalize_dh_convention",
]

__version__ = "0.1.0"
