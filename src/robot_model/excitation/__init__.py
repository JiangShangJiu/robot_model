"""激励轨迹设计：生成、解析搜索、约束优化与闭环筛选。"""

from .trajectory import FourierTrajectory
from .search import search_excitation
from .screening import (
    ExcitationCriteria,
    ExcitationSelection,
    NoFeasibleExcitation,
    generate_excitation_candidates,
    regressor_quality,
    screen_excitation,
)
from .optimization import ExcitationOptimization, optimize_excitation
from .robust import screen_robust_excitation

__all__ = [
    "FourierTrajectory", "search_excitation", "ExcitationCriteria",
    "ExcitationSelection", "NoFeasibleExcitation", "generate_excitation_candidates",
    "regressor_quality", "screen_excitation", "ExcitationOptimization",
    "optimize_excitation", "screen_robust_excitation",
]
