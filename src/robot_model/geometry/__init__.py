"""几何层：运动学描述、约定、连杆坐标系。"""

from .conventions import (
    ModifiedDH,
    StandardDH,
    get_convention,
    normalize_convention,
    normalize_dh_convention,
)
from .dh import DHParams
from .frames import FrameChain
from .poe import PoEParams

Description = DHParams | PoEParams

__all__ = [
    "FrameChain",
    "Description",
    "DHParams",
    "PoEParams",
    "StandardDH",
    "ModifiedDH",
    "get_convention",
    "normalize_convention",
    "normalize_dh_convention",
]
