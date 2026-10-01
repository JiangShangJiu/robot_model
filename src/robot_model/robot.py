"""高层编排：符号 ⊕ 运动学描述 → 坐标系 / 雅可比 / 动力学。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import sympy

from .dynamics import Dynamics
from .geometry import Description, FrameChain, normalize_convention
from .geometry.dh import DHParams
from .geometry.poe import PoEParams
from .kinematics import Kinematics
from .utils.symbols import RobotSymbols

if TYPE_CHECKING:
    from .export import Exporter


class Robot:
    """符号机器人：组合 ``RobotSymbols`` 与 ``Description``（DH / PoE）。
        symbols      有哪些未知量：q/dq/ddq 与质量、一阶矩、惯性、摩擦等符号
        description  机器人长什么样：DH/MDH 参数表或 PoE 旋量（纯几何）
        frames       每根连杆在哪：由上两者推出的位姿表达式 T[i]
        kinematics   关节动一下末端怎么动：各连杆雅可比 J
        dynamics     要多大力矩：τ、M、C、g 与回归矩阵 H

    前两个由构造函数给定，后三个惰性推导。
    另有 ``export``：把上面的符号结果落成 LaTeX / Python / C。
    """

    def __init__(self, symbols: RobotSymbols, description: Description):
        if symbols.dof != description.dof:
            raise ValueError("RobotSymbols.dof and description.dof must match")
        self.symbols = symbols
        self.description = description
        # 由同名 property 惰性构造，首次访问才计算。三者有依赖链：
        # 访问 dynamics 会连带建好 frames 与 kinematics；想「构造即成形」用 precompute()。
        self._frames: FrameChain | None = None
        self._kin: Kinematics | None = None
        self._dyn: Dynamics | None = None

    @classmethod
    def from_dh(
        cls,
        name: str,
        dh_parms: Sequence[Sequence],
        *,
        frictionmodel: Iterable[str] | None = None,
        **kwargs,
    ) -> Robot:
        symbols = RobotSymbols(
            name, len(dh_parms), frictionmodel=frictionmodel, **kwargs
        )
        description = DHParams.from_rows(dh_parms, symbols.q, "standard")
        return cls(symbols, description)

    @classmethod
    def from_mdh(
        cls,
        name: str,
        dh_parms: Sequence[Sequence],
        *,
        frictionmodel: Iterable[str] | None = None,
        **kwargs,
    ) -> Robot:
        symbols = RobotSymbols(
            name, len(dh_parms), frictionmodel=frictionmodel, **kwargs
        )
        description = DHParams.from_rows(dh_parms, symbols.q, "modified")
        return cls(symbols, description)

    @classmethod
    def from_poe(
        cls,
        name: str,
        screws: Sequence,
        M,
        *,
        frame: str = "space",
        sigma: Sequence[int] | None = None,
        home_convention: str | None = None,
        frictionmodel: Iterable[str] | None = None,
        **kwargs,
    ) -> Robot:
        """PoE 指数积（``M`` 也可为每连杆零位列表）。

        - ``frame='space'``：``T=e^{[S_1]q_1}…e^{[S_n]q_n} M``，旋量在基座系
        - ``frame='body'``：``T=M e^{[B_1]q_1}…e^{[B_n]q_n}``，旋量在末端零位系
        """
        if frame == "space":
            description = PoEParams(
                screws, M, sigma=sigma, home_convention=home_convention
            )
        elif frame == "body":
            description = PoEParams.from_body_screws(
                screws, M, sigma=sigma, home_convention=home_convention
            )
        else:
            raise ValueError(f"unknown frame {frame!r}; use 'space' or 'body'")
        symbols = RobotSymbols(
            name, description.dof, frictionmodel=frictionmodel, **kwargs
        )
        return cls(symbols, description)

    def to_poe(self) -> Robot:
        """由当前 DH/MDH 零位提取空间旋量，得到等价 PoE 模型（正运动学）。"""
        if self.convention == "poe":
            return Robot(self.symbols.copy(), self.description)
        poe = PoEParams.from_zero_config(self.frames, sigma=self.description.sigma)
        return Robot(self.symbols.copy(), poe)

    def to_convention(self, convention: str) -> Robot:
        """DH / MDH / PoE 互转（新 ``Robot``，符号复制）。

        - DH ↔ MDH：Craig ``(α,a)`` 行间平移
        - DH/MDH → PoE：零位抽空间旋量
        - PoE → DH/MDH：由零位 ``M_i`` 相邻变换反解参数
        """
        target = normalize_convention(convention)
        if target == "poe":
            return self.to_poe()

        if self.convention == target:
            return Robot(self.symbols.copy(), self.description)

        if self.convention == "poe":
            desc = self.description.to_dh_params(self.symbols.q, target)
            return Robot(self.symbols.copy(), desc)

        if isinstance(self.description, DHParams):
            if target in ("standard", "modified"):
                return Robot(
                    self.symbols.copy(), self.description.to_convention(target)
                )

        raise TypeError(
            f"cannot convert convention {self.convention!r} → {convention!r}"
        )

    def to_dh(self) -> Robot:
        """转为标准 DH。"""
        return self.to_convention("standard")

    def to_mdh(self) -> Robot:
        """转为修正 DH（MDH）。"""
        return self.to_convention("modified")

    @property
    def dof(self) -> int:
        return self.symbols.dof

    @property
    def q(self) -> sympy.Matrix:
        return self.symbols.q

    @property
    def convention(self) -> str:
        return self.description.convention

    @property
    def frames(self) -> FrameChain:
        if self._frames is None:
            self._frames = FrameChain(self.symbols, self.description)
        return self._frames

    @property
    def kinematics(self) -> Kinematics:
        if self._kin is None:
            self._kin = Kinematics(self.symbols, self.frames)
        return self._kin

    @property
    def dynamics(self) -> Dynamics:
        if self._dyn is None:
            self._dyn = Dynamics(self.symbols, self.description, self.frames)
            self._dyn.bind_kinematics(self.kinematics)
        return self._dyn

    @property
    def export(self) -> Exporter:
        """导出门面：``latex`` / ``python`` / ``c`` / ``show``。

        无状态，每次返回新实例；延迟导入以免把打印器拉进主 import 链。
        """
        from .export import Exporter

        return Exporter(self)

    def precompute(self) -> Robot:
        """一次性建好 ``frames`` / ``kinematics`` / ``dynamics``。

        三者默认惰性构造，首次访问才计算。需要「构造即成形」时（计时、
        提前暴露错误）显式调用本方法。
        """
        _ = self.dynamics  # 链式触发 frames → kinematics → dynamics
        return self

    def forward_kinematics(self, link: int = -1) -> sympy.Matrix:
        """基座到指定连杆的齐次变换；默认末端。"""
        idx = self.dof - 1 if link < 0 else link
        return self.frames.T[idx]

    def fk_numpy(self, q, link: int = -1):
        """数值正运动学：``q`` 为长度 ``dof`` 的序列，返回 ``numpy`` 的 4×4。"""
        import numpy as np

        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.dof:
            raise ValueError(f"q length {q.size} != dof {self.dof}")
        T_fun = sympy.lambdify(list(self.q), self.forward_kinematics(link), "numpy")
        return np.asarray(T_fun(*q), dtype=float)

    def compare_fk(self, other: Robot, q, *, link: int = -1) -> dict:
        """与另一模型在同一 ``q`` 下对比正运动学（位置/姿态误差）。"""
        import numpy as np

        if self.dof != other.dof:
            raise ValueError("dof mismatch")
        Ta = self.fk_numpy(q, link)
        Tb = other.fk_numpy(q, link)
        return {
            "pos_a": Ta[:3, 3],
            "pos_b": Tb[:3, 3],
            "pos_err": float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])),
            "rot_err": float(np.linalg.norm(Ta[:3, :3] - Tb[:3, :3])),
            "T_a": Ta,
            "T_b": Tb,
        }

    def dh_table(self) -> str:
        """当前 DH/MDH 参数表文本。"""
        if not isinstance(self.description, DHParams):
            raise TypeError("dh_table requires a DH/MDH description")
        return self.description.format_table()

    def print_dh_table(self) -> None:
        if not isinstance(self.description, DHParams):
            raise TypeError("print_dh_table requires a DH/MDH description")
        self.description.print_table()

    def print_poe_table(self) -> None:
        if not isinstance(self.description, PoEParams):
            raise TypeError("print_poe_table requires a PoE description")
        self.description.print_table()

    def poe_table(self) -> str:
        if not isinstance(self.description, PoEParams):
            raise TypeError("poe_table requires a PoE description")
        return self.description.format_table()
