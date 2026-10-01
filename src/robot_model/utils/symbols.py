"""符号层：关节与动力学参数符号（不含 DH/PoE 几何）。"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import sympy

from .mathutil import elements_to_tensor, skew


def _sym(name: str) -> sympy.Symbol:
    return sympy.symbols(name, real=True)


class RobotSymbols:
    """q / dq / ddq 与惯性、摩擦等符号。"""

    def __init__(
        self,
        name: str,
        dof: int,
        *,
        frictionmodel: Iterable[str] | None = None,
        driveinertiamodel: str | None = None,
        gravityacc: sympy.Matrix | None = None,
        dyn_parms_order: str = "khalil",
    ):
        if dof < 1:
            raise ValueError("dof must be >= 1")
        self.name = str(name)
        self.dof = int(dof)
        self.frictionmodel = None if frictionmodel is None else set(frictionmodel)
        self.driveinertiamodel = driveinertiamodel
        self.gravityacc = (
            sympy.Matrix([[0.0], [0.0], [-9.81]])
            if gravityacc is None
            else sympy.Matrix(gravityacc)
        )
        self.dyn_parms_order = dyn_parms_order
        self._init_motion_symbols()
        self._init_dyn_symbols()

    @property
    def L(self):
        """相对连杆系原点的惯性张量（3×3）。"""
        return [elements_to_tensor(e) for e in self.Le]

    @property
    def I(self):
        """相对质心的惯性张量（3×3）。"""
        return [elements_to_tensor(e) for e in self.Ie]

    def copy(self) -> RobotSymbols:
        return deepcopy(self)

    def _link_index(self, link: int) -> int:
        i = int(link)
        if i < 0 or i >= self.dof:
            raise IndexError(f"link index {i} out of range [0, {self.dof})")
        return i

    def link_dyn(self, link: int) -> dict:
        """查看单连杆动力学符号与 I↔L 平行轴表达式（``link`` 为 0-based）。"""
        i = self._link_index(link)
        L_from_I = self.I[i] + self.m[i] * skew(self.r[i]).T * skew(self.r[i])
        I_from_L = self.L[i] - self.m[i] * skew(self.r[i]).T * skew(self.r[i])
        return {
            "m": self.m[i],
            "l": self.l[i],
            "r": self.r[i],
            "Le": self.Le[i],
            "L": self.L[i],
            "Ie": self.Ie[i],
            "I": self.I[i],
            "Ia": self.Ia[i],
            "fv": self.fv[i],
            "fc": self.fc[i],
            "fo": self.fo[i],
            "L_from_I": L_from_I,
            "I_from_L": I_from_L,
        }

    def inertia_subs(self, link: int | None = None) -> dict:
        """I↔L、l↔r 的 ``subs`` 字典；``link`` 给定时只含该连杆。"""
        if link is None:
            return {
                "I2L": dict(self.dict_I2Lexp),
                "L2I": dict(self.dict_L2Iexp),
                "l2mr": dict(self.dict_l2mr),
                "r2lm": dict(self.dict_r2lm),
            }
        i = self._link_index(link)
        I_syms = set(self.I[i])
        L_syms = set(self.L[i])
        l_syms = set(self.l[i])
        r_syms = set(self.r[i])
        return {
            "I2L": {k: v for k, v in self.dict_I2Lexp.items() if k in I_syms},
            "L2I": {k: v for k, v in self.dict_L2Iexp.items() if k in L_syms},
            "l2mr": {k: v for k, v in self.dict_l2mr.items() if k in l_syms},
            "r2lm": {k: v for k, v in self.dict_r2lm.items() if k in r_syms},
        }

    def dynparms(self, parm_order: str | None = None) -> list:
        """Barycentric 动力学参数向量（默认 Khalil 顺序）。"""
        order = (parm_order or self.dyn_parms_order).lower()
        parms: list = []
        for i in range(self.dof):
            if order in ("khalil", "tensor first"):
                parms += self.Le[i]
                parms += list(sympy.flatten(self.l[i]))
                parms += [self.m[i]]
            elif order in ("siciliano", "mass first"):
                parms += [self.m[i]]
                parms += list(sympy.flatten(self.l[i]))
                parms += self.Le[i]
            else:
                raise ValueError(f"unknown dyn parms order: {order}")

            if self.driveinertiamodel == "simplified":
                parms += [self.Ia[i]]
            if self.frictionmodel:
                if "viscous" in self.frictionmodel:
                    parms += [self.fv[i]]
                if "Coulomb" in self.frictionmodel:
                    parms += [self.fc[i]]
                if "offset" in self.frictionmodel:
                    parms += [self.fo[i]]
        return parms

    def _init_motion_symbols(self) -> None:
        self.q = sympy.Matrix([[_sym(f"q{i + 1}")] for i in range(self.dof)])
        self.dq = sympy.Matrix(
            [[_sym(rf"\dot{{q}}_{i + 1}")] for i in range(self.dof)]
        )
        self.ddq = sympy.Matrix(
            [[_sym(rf"\ddot{{q}}_{i + 1}")] for i in range(self.dof)]
        )

    def _init_dyn_symbols(self) -> None:
        n = self.dof
        # 质量
        self.m = [_sym(f"m_{i + 1}") for i in range(n)]
        # 一阶矩 l = m * r
        self.l = [
            sympy.Matrix([_sym(f"l_{i + 1}{d}") for d in "xyz"]) for i in range(n)
        ]
        # 相对连杆系原点的惯性（6 分量）
        self.Le = [
            [_sym(f"L_{i + 1}{e}") for e in ("xx", "xy", "xz", "yy", "yz", "zz")]
            for i in range(n)
        ]
        # 质心相对连杆系原点的位置
        self.r = [
            sympy.Matrix([_sym(f"r_{i + 1}{d}") for d in "xyz"]) for i in range(n)
        ]
        # 相对质心的惯性（6 分量）
        self.Ie = [
            [_sym(f"I_{i + 1}{e}") for e in ("xx", "xy", "xz", "yy", "yz", "zz")]
            for i in range(n)
        ]
        # 电机/驱动侧等效转动惯量（简化模型）
        self.Ia = [_sym(f"Ia_{i + 1}") for i in range(n)]
        # 粘性 / 库仑 / 偏置摩擦
        self.fv = [_sym(f"fv_{i + 1}") for i in range(n)]
        self.fc = [_sym(f"fc_{i + 1}") for i in range(n)]
        self.fo = [_sym(f"fo_{i + 1}") for i in range(n)]

        self.dict_l2mr = {}
        self.dict_r2lm = {}
        for i in range(n):
            for k in range(3):
                self.dict_l2mr[self.l[i][k]] = self.m[i] * self.r[i][k]
                self.dict_r2lm[self.r[i][k]] = self.l[i][k] / self.m[i]

        self.dict_I2Lexp = {}
        self.dict_L2Iexp = {}
        for i in range(n):
            L_of_I = self.I[i] + self.m[i] * skew(self.r[i]).T * skew(self.r[i])
            I_of_L = self.L[i] - self.m[i] * skew(self.r[i]).T * skew(self.r[i])
            for elem, expr in enumerate(I_of_L):
                self.dict_I2Lexp[self.I[i][elem]] = expr
            for elem, expr in enumerate(L_of_I):
                self.dict_L2Iexp[self.L[i][elem]] = expr
