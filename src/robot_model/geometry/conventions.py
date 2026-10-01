"""运动学约定：把连杆参数变成相邻齐次变换。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import sympy

ConventionName = Literal["standard", "modified", "poe"]
DhConventionName = Literal["standard", "modified"]


def normalize_convention(name: str) -> ConventionName:
    """统一约定别名 → ``standard`` | ``modified`` | ``poe``。"""
    key = name.lower()
    if key in ("standard", "std", "dh", "sdh"):
        return "standard"
    if key in ("modified", "mod", "mdh"):
        return "modified"
    if key in ("poe", "screw", "twist"):
        return "poe"
    raise ValueError(
        f"unknown convention {name!r}; use 'standard', 'modified', or 'poe'"
    )


def normalize_dh_convention(name: str) -> DhConventionName:
    """仅 DH/MDH：别名 → ``standard`` | ``modified``。"""
    c = normalize_convention(name)
    if c == "poe":
        raise ValueError(
            f"unknown DH-style convention {name!r}; use 'standard' or 'modified'"
        )
    return c


class KinematicConvention(ABC):
    """约定插件接口（仅服务 DH/MDH 相邻变换；PoE 走 ``PoEParams`` / ``FrameChain``）。"""

    name: str

    @abstractmethod
    def adjacent_transform(self, alpha, a, d, theta) -> sympy.Matrix:
        """相邻连杆齐次变换 T_{i-1,i}。"""


class StandardDH(KinematicConvention):
    name = "standard"

    def adjacent_transform(self, alpha, a, d, theta) -> sympy.Matrix:
        cth, sth = sympy.cos(theta), sympy.sin(theta)
        cal, sal = sympy.cos(alpha), sympy.sin(alpha)
        return sympy.Matrix(
            [
                [cth, -sth * cal, sth * sal, a * cth],
                [sth, cth * cal, -cth * sal, a * sth],
                [0, sal, cal, d],
                [0, 0, 0, 1],
            ]
        )


class ModifiedDH(KinematicConvention):
    name = "modified"

    def adjacent_transform(self, alpha, a, d, theta) -> sympy.Matrix:
        cth, sth = sympy.cos(theta), sympy.sin(theta)
        cal, sal = sympy.cos(alpha), sympy.sin(alpha)
        return sympy.Matrix(
            [
                [cth, -sth, 0, a],
                [sth * cal, cth * cal, -sal, -sal * d],
                [sth * sal, cth * sal, cal, cal * d],
                [0, 0, 0, 1],
            ]
        )


class ProductOfExponentials(KinematicConvention):
    """注册表占位：相邻 ``(α,a,d,θ)`` 变换对 PoE 无意义。

    真实 PoE 几何由 ``geometry.poe.PoEParams`` + ``FrameChain._init_poe`` 完成。
    """

    name = "poe"

    def adjacent_transform(self, alpha, a, d, theta) -> sympy.Matrix:
        raise NotImplementedError(
            "PoE does not use DH-style adjacent_transform; "
            "construct Robot.from_poe / PoEParams instead."
        )


def get_convention(name: str) -> KinematicConvention:
    key = normalize_convention(name)
    if key == "standard":
        return StandardDH()
    if key == "modified":
        return ModifiedDH()
    return ProductOfExponentials()
