"""由运动学描述生成各连杆坐标系（齐次变换等）。"""

from __future__ import annotations

from typing import Callable

import sympy

from ..utils.mathutil import adjoint_twist, identity, inverse_T, se3_exp
from ..utils.symbols import RobotSymbols
from .conventions import get_convention
from .dh import DHParams
from .poe import PoEParams


class FrameChain:
    """开链几何：相邻变换 + 累积位姿 + 体/空间关节旋量。

    字段（约定无关）::

        T[i]         基座 → 连杆 i
        T_rel[i]     相邻 ``T_{i-1,i}``
        S_body[i]    体坐标系关节旋量（Park RNE）
        S_space[i]   空间旋量（PoE 雅可比；DH 路径为 None）
        E[i]         ``e^{[S_i]q_i}``（仅 PoE）
    """

    def __init__(
        self,
        symbols: RobotSymbols,
        description: DHParams | PoEParams,
        ifunc: Callable | None = None,
    ):
        if description.dof != symbols.dof:
            raise ValueError("description.dof must match RobotSymbols.dof")

        self.symbols = symbols
        self.description = description
        self.dof = symbols.dof
        self.convention = description.convention
        self.sigma = list(description.sigma)
        ifunc = ifunc or identity

        self.T_rel = [None] * self.dof
        self.T_rel_inv = [None] * self.dof
        self.R_rel = [None] * self.dof
        self.p_rel = [None] * self.dof
        self.S_body = None
        self.S_space = None
        self.E = None

        if description.convention == "poe":
            self._init_poe(ifunc)
        else:
            self._init_dh(ifunc)

        self.R = [None] * self.dof
        self.p = [None] * self.dof
        self.z = [None] * self.dof
        for i in range(self.dof):
            self.R[i] = self.T[i][0:3, 0:3]
            self.p[i] = self.T[i][0:3, 3]
            self.z[i] = self.R[i][0:3, 2]

    def _init_poe(self, ifunc: Callable) -> None:
        """空间 PoE：``T_i = e^{[S_1]q_1} … e^{[S_i]q_i} M_i``。"""
        screws = self.description.screws
        M = self.description.M
        q = self.symbols.q

        self.E = [None] * self.dof
        self.S_space = [sympy.Matrix(s) for s in screws]
        for i in range(self.dof):
            self.E[i] = ifunc(
                se3_exp(screws[i], q[i], prismatic=bool(self.sigma[i]))
            )

        self.T = [None] * self.dof
        acc = sympy.eye(4)
        for i in range(self.dof):
            acc = ifunc(acc * self.E[i])
            self.T[i] = ifunc(acc * M[i])

        T_prev = sympy.eye(4)
        self.S_body = [None] * self.dof
        for i in range(self.dof):
            self.T_rel[i] = ifunc(inverse_T(T_prev) * self.T[i])
            self.T_rel_inv[i] = ifunc(inverse_T(self.T_rel[i]))
            self.R_rel[i] = self.T_rel[i][0:3, 0:3]
            self.p_rel[i] = self.T_rel[i][0:3, 3]
            self.S_body[i] = ifunc(
                adjoint_twist(
                    inverse_T(sympy.Matrix(M[i])), sympy.Matrix(screws[i])
                )
            )
            T_prev = self.T[i]

    def _init_dh(self, ifunc: Callable) -> None:
        xform = get_convention(self.description.convention)
        alpha, a, d, theta = sympy.symbols("alpha,a,d,theta", real=True)
        T_template = xform.adjacent_transform(alpha, a, d, theta)
        T_inv_template = inverse_T(T_template)
        parm_syms = (alpha, a, d, theta)

        for i in range(self.dof):
            subs = dict(zip(parm_syms, self.description.parms[i]))
            self.T_rel[i] = ifunc(T_template.subs(subs))
            self.T_rel_inv[i] = ifunc(T_inv_template.subs(subs))
            self.R_rel[i] = self.T_rel[i][0:3, 0:3]
            self.p_rel[i] = self.T_rel[i][0:3, 3]

        self.T = [None] * self.dof
        T_prev = sympy.eye(4)
        for i in range(self.dof):
            self.T[i] = ifunc(T_prev * self.T_rel[i])
            T_prev = self.T[i]

        if self.description.convention == "standard":
            self.S_body = self._screw_axes_standard(parm_syms, ifunc)
        else:
            self.S_body = self._screw_axes_modified(ifunc)

    def _screw_axes_modified(self, ifunc):
        """MDH 体坐标系关节旋量 ``S_body``（供 Park RNE）。

        Modified DH 下关节轴即连杆系 ``z`` 且过原点，旋量为常值 6 维向量
        ``[ω; v]``：转动 ``[0,0,1,0,0,0]``，移动 ``[0,0,0,0,0,1]``。
        """
        axes = [None] * self.dof
        for i in range(self.dof):
            if self.sigma[i]:
                # 移动关节：沿体 z
                axes[i] = ifunc(sympy.Matrix([0, 0, 0, 0, 0, 1]))
            else:
                # 转动关节：绕体 z
                axes[i] = ifunc(sympy.Matrix([0, 0, 1, 0, 0, 0]))
        return axes

    def _screw_axes_standard(self, parm_syms, ifunc):
        """标准 DH 体坐标系关节旋量 ``S_body``（供 Park RNE）。

        标准 DH 的关节轴一般不落在连杆原点处的 ``z``，不能像 MDH 那样写常数。
        做法：在易写的中间系里给出运动生成元 ``P``（4×4 李代数），再用
        ``T^{-1} P T`` 变到体坐标系，``se3_unskew`` 拆成 ``[ω; v]``，
        最后代入该连杆的 ``(α,a,d,θ)``。
        """
        alpha, a, d, theta = parm_syms
        cos, sin = sympy.cos, sympy.sin

        # 转动：中间系 → 体；Pr 为绕 z 的 se(3) 生成元
        Mr = sympy.Matrix(
            [
                [1, 0, 0, a],
                [0, cos(alpha), -sin(alpha), 0],
                [0, sin(alpha), cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )
        Pr = sympy.Matrix(
            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        # 移动：中间系 → 体；Pp 为沿 z 的 se(3) 生成元
        Mp = sympy.Matrix(
            [
                [
                    cos(theta),
                    -sin(theta) * cos(alpha),
                    sin(theta) * sin(alpha),
                    a * cos(theta),
                ],
                [
                    sin(theta),
                    cos(theta) * cos(alpha),
                    -cos(theta) * sin(alpha),
                    a * sin(theta),
                ],
                [0, sin(alpha), cos(alpha), 0],
                [0, 0, 0, 1],
            ]
        )
        Pp = sympy.Matrix(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
        )

        # 伴随变换到体坐标系：ξ̂_body = T^{-1} ξ̂ T
        Sr = (inverse_T(Mr) * Pr * Mr).applyfunc(lambda x: sympy.trigsimp(x))
        Sp = (inverse_T(Mp) * Pp * Mp).applyfunc(lambda x: sympy.trigsimp(x))

        def se3_unskew(g):
            """4×4 se(3) 元素 → 6×1 旋量 ``[ω; v]``。"""
            w = sympy.Matrix([g[2, 1], g[0, 2], g[1, 0]])
            v = g[0:3, 3]
            return w.col_join(v)

        axes = [None] * self.dof
        for i in range(self.dof):
            S = Sp if self.sigma[i] else Sr
            axes[i] = ifunc(
                se3_unskew(
                    S.subs(dict(zip(parm_syms, self.description.parms[i])))
                )
            )
        return axes
