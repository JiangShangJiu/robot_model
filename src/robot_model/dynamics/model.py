"""动力学模型门面。"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..geometry.dh import DHParams
from ..geometry.frames import FrameChain
from ..geometry.poe import PoEParams
from ..utils.symbols import RobotSymbols
from . import base_parms as base_mod
from . import derived as derived_mod
from . import methods as methods_mod
from . import regressor as regressor_mod
from .methods import RneMethod
from .numeric import NumericDynamics


class Dynamics:
    """按需生成逆动力学 / M / g / c / H / 基参数等符号量。

    ``method``：
    - ``\"park\"`` / ``\"khalil\"``：牛顿–欧拉（DH·MDH；PoE 仅 park）
    - ``\"lagrange\"``：拉格朗日（DH / MDH / PoE）
    - ``None``：DH→Park，MDH→Khalil，PoE→Park
    """

    def __init__(
        self,
        symbols: RobotSymbols,
        description: DHParams | PoEParams,
        frames: FrameChain | None = None,
        *,
        method: RneMethod | None = None,
    ):
        self.symbols = symbols
        self.description = description
        self.frames = frames or FrameChain(symbols, description)
        self.method = method
        self.dynparms = symbols.dynparms()
        self.n_dynparms = len(self.dynparms)
        self._kinematics = None

        # 由各 gen_* / calc_* 按需填充；在此声明便于类型检查与补全。
        self.invdyn = None  # τ
        self.M = None  # 惯性矩阵
        self.c = None  # 科氏 / 离心项
        self.C = None  # 科氏矩阵（Christoffel）
        self.g = None  # 重力项
        self.f = None  # 摩擦项
        self.S_skew = None  # Ṁ - 2C
        self.H = None  # 完整回归矩阵
        self.Hb = None  # 最小参数回归（符号）
        self.Hb_func = None  # 最小参数回归（数值）
        self.Pb = None
        self.Pd = None
        self.Kd = None
        self.base_idxs = None
        self.baseparms = None
        self.n_base = None
        self._regressor_func = None

    def _method(self, method: RneMethod | None) -> RneMethod | None:
        return self.method if method is None else method

    def bind_kinematics(self, kinematics) -> None:
        """可选：绑定已有 ``Kinematics``，拉格朗日复用雅可比。"""
        self._kinematics = kinematics

    def gen_invdyn(
        self, ifunc: Callable | None = None, *, method: RneMethod | None = None
    ):
        self.invdyn = methods_mod.inverse_dynamics(
            self.symbols,
            self.frames,
            ifunc,
            method=self._method(method),
            kinematics=self._kinematics,
        )
        return self.invdyn

    def gen_gravityterm(
        self, ifunc: Callable | None = None, *, method: RneMethod | None = None
    ):
        self.g = derived_mod.gravity_term(
            self.symbols, self.frames, ifunc, method=self._method(method)
        )
        return self.g

    def gen_coriolisterm(
        self, ifunc: Callable | None = None, *, method: RneMethod | None = None
    ):
        self.c = derived_mod.coriolis_term(
            self.symbols, self.frames, ifunc, method=self._method(method)
        )
        return self.c

    def gen_coriolismatrix(
        self, ifunc: Callable | None = None, *, method: RneMethod | None = None
    ):
        """科氏矩阵 ``C``（Christoffel）：``c=C dq``，保证 ``Ṁ-2C`` 斜对称。"""
        if self.M is None:
            self.gen_inertiamatrix(ifunc, method=method)
        self.C = derived_mod.coriolis_matrix(
            self.symbols,
            self.frames,
            ifunc,
            method=self._method(method),
            M=self.M,
        )
        return self.C

    def gen_skew_symmetry(
        self, ifunc: Callable | None = None, *, method: RneMethod | None = None
    ):
        """``S=Ṁ-2C``（应斜对称）。若缺 ``M/C`` 则先生成。"""
        if self.M is None:
            self.gen_inertiamatrix(ifunc, method=method)
        if self.C is None:
            self.gen_coriolismatrix(ifunc, method=method)
        self.S_skew = derived_mod.mdot_minus_2C(
            self.M, self.C, self.symbols.q, self.symbols.dq, ifunc
        )
        return self.S_skew

    def gen_frictionterm(self, ifunc: Callable | None = None):
        self.f = derived_mod.friction_term(self.symbols, ifunc)
        return self.f

    def gen_inertiamatrix(
        self, ifunc: Callable | None = None, *, method: RneMethod | None = None
    ):
        self.M = derived_mod.inertia_matrix(
            self.symbols, self.frames, ifunc, method=self._method(method)
        )
        return self.M

    def gen_regressor(
        self, ifunc: Callable | None = None, *, method: RneMethod | None = None
    ):
        """完整回归矩阵 ``H``，``τ = H π``（``π = dynparms``）。"""
        self.H = regressor_mod.regressor(
            self.symbols,
            self.frames,
            ifunc,
            method=self._method(method),
            kinematics=self._kinematics,
        )
        return self.H

    def numeric_dynamics(self, *, method: RneMethod | None = None) -> NumericDynamics:
        """数值 RNE / 回归（7 轴推荐，避免全符号 ``H``）。"""
        return NumericDynamics(
            self.symbols, self.frames, method=self._method(method)
        )

    def regressor_numpy(
        self, q, dq, ddq, *, method: RneMethod | None = None
    ):
        """数值回归矩阵 ``H(q,dq,ddq)``，``(dof × n_dynparms)``。"""
        return self.numeric_dynamics(method=method).regressor(q, dq, ddq)

    def make_regressor_func(
        self, *, method: RneMethod | None = None, numeric: bool | None = None
    ):
        """返回 ``func(q,dq,ddq)->H``。

        ``numeric=True``（或 ``dof>=6`` 且未指定）走数值列构造；
        否则 lambdify 符号 ``H``。
        """
        use_numeric = numeric if numeric is not None else (self.symbols.dof >= 6)
        if use_numeric:
            return self.numeric_dynamics(method=method).as_regressor_func()
        if self.H is None:
            self.gen_regressor(method=method)
        return base_mod.make_regressor_func(self.symbols, self.H)

    def calc_base_parms(
        self,
        regressor_func: Callable | None = None,
        *,
        method: RneMethod | None = None,
        samples: int = 2000,
        seed: int = 0,
        numeric: bool | None = None,
    ):
        """提取最小参数集（基参数）。

        高自由度默认 ``numeric=True``，用数值 ``H`` 采样做 QR；
        低自由度可符号 ``gen_regressor`` 后 lambdify。
        """
        if regressor_func is None:
            regressor_func = self.make_regressor_func(
                method=method, numeric=numeric
            )

        result = base_mod.calc_base_parms(
            self.dynparms,
            self.symbols.dof,
            regressor_func,
            samples=samples,
            seed=seed,
        )
        self.Pb = result["Pb"]
        self.Pd = result["Pd"]
        self.Kd = result["Kd"]
        self.base_idxs = result["base_idxs"]
        self.baseparms = result["baseparms"]
        self.n_base = result["n_base"]
        self._regressor_func = regressor_func
        return self.baseparms

    def gen_base_regressor(self, *, method: RneMethod | None = None):
        """最小参数回归：符号 ``Hb=H Pb``（若有符号 ``H``），并缓存数值 ``Hb_func``。"""
        if self.Pb is None:
            self.calc_base_parms(method=method)

        if self.H is not None:
            self.Hb = self.H * self.Pb
        else:
            self.Hb = None

        Pb = np.asarray(self.Pb, dtype=float)
        if self._regressor_func is None:
            self._regressor_func = self.make_regressor_func(method=method)

        def Hb_func(q, dq, ddq):
            return np.asarray(self._regressor_func(q, dq, ddq), float) @ Pb

        self.Hb_func = Hb_func
        return self.Hb if self.Hb is not None else self.Hb_func

    def gen_all(
        self, ifunc: Callable | None = None, *, method: RneMethod | None = None
    ):
        m = self._method(method)
        self.gen_invdyn(ifunc, method=m)
        self.gen_gravityterm(ifunc, method=m)
        self.gen_coriolisterm(ifunc, method=m)
        self.gen_inertiamatrix(ifunc, method=m)
        if self.symbols.dof < 6:
            self.gen_coriolismatrix(ifunc, method=m)
            self.gen_regressor(ifunc, method=m)
        return self
