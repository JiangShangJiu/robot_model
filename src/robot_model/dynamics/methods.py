"""动力学方法解析与调度：``park`` / ``khalil`` / ``lagrange``。"""

from __future__ import annotations

from typing import Literal

from ..utils.symbols import RobotSymbols
from ..geometry.frames import FrameChain
from . import lagrange as lag_mod
from .rne import (
    rne_khalil_backward,
    rne_khalil_forward,
    rne_park_backward,
    rne_park_forward,
)

DynamicsMethod = Literal["park", "khalil", "lagrange"]
RneMethod = DynamicsMethod  # 历史别名


def resolve_rne_method(
    frames: FrameChain, method: DynamicsMethod | None = None
) -> DynamicsMethod:
    """``None`` 时：标准 DH→Park，MDH→Khalil，PoE→Park。"""
    if method is not None:
        if method not in ("park", "khalil", "lagrange"):
            raise ValueError(
                f"unknown dynamics method {method!r}; "
                "use 'park', 'khalil', or 'lagrange'"
            )
        if method == "khalil" and frames.convention == "poe":
            raise NotImplementedError(
                "Khalil RNE is not defined for PoE; use method='park' or 'lagrange'"
            )
        return method
    if frames.convention == "standard":
        return "park"
    if frames.convention == "modified":
        return "khalil"
    if frames.convention == "poe":
        return "park"
    raise NotImplementedError(
        f"dynamics for convention={frames.convention!r} not implemented"
    )


def rne_forward(
    symbols: RobotSymbols,
    frames: FrameChain,
    ifunc=None,
    *,
    method: DynamicsMethod | None = None,
):
    m = resolve_rne_method(frames, method)
    if m == "lagrange":
        raise ValueError("rne_forward is not used for method='lagrange'")
    if m == "park":
        return rne_park_forward(symbols, frames, ifunc)
    return rne_khalil_forward(symbols, frames, ifunc)


def rne_backward(
    symbols: RobotSymbols,
    frames: FrameChain,
    fw_results,
    ifunc=None,
    *,
    method: DynamicsMethod | None = None,
):
    m = resolve_rne_method(frames, method)
    if m == "lagrange":
        raise ValueError("rne_backward is not used for method='lagrange'")
    if m == "park":
        return rne_park_backward(symbols, frames, fw_results, ifunc)
    return rne_khalil_backward(symbols, frames, fw_results, ifunc)


def inverse_dynamics(
    symbols: RobotSymbols,
    frames: FrameChain,
    ifunc=None,
    *,
    method: DynamicsMethod | None = None,
    kinematics=None,
):
    m = resolve_rne_method(frames, method)
    if m == "lagrange":
        return lag_mod.inverse_dynamics_lagrange(
            symbols, frames, ifunc, kinematics=kinematics
        )
    return rne_backward(
        symbols,
        frames,
        rne_forward(symbols, frames, ifunc, method=m),
        ifunc,
        method=m,
    )
