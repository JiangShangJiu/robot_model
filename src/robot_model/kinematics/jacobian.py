"""符号雅可比：几何 ``J`` 与解析 ``Ja``（对 ``[p; rpy]`` 直接求导）。"""

from __future__ import annotations

from typing import Callable

import sympy

from ..geometry.frames import FrameChain
from ..utils.symbols import RobotSymbols
from ..utils.mathutil import (
    adjoint_twist,
    analytical_from_geometric,
    identity,
    rpy_zyx_E,
    rpy_zyx_from_R,
    skew,
)


class Kinematics:
    """各连杆雅可比。

    - 几何：``J=[Jp; Jo]``，``[v; ω]=J \\dot q``（基座系，连杆原点）
    - 解析：``xa=[p;φ]``，``φ`` 为 ZYX ``[roll,pitch,yaw]``；
      ``Ja`` 由 ``∂xa/∂q`` 直接得到（低自由度符号；高自由度用数值差分）。
    """

    def __init__(
        self,
        symbols: RobotSymbols,
        frames: FrameChain,
        ifunc: Callable | None = None,
        *,
        analytical: bool = True,
    ):
        self.symbols = symbols
        self.frames = frames
        self.dof = symbols.dof
        ifunc = ifunc or identity
        convention = frames.convention
        sigma = frames.sigma

        self.Jp = [None] * self.dof
        self.Jo = [None] * self.dof

        if convention == "standard":
            self._jacobian_standard(frames, sigma, ifunc)
        elif convention == "modified":
            self._jacobian_modified(frames, sigma, ifunc)
        elif convention == "poe":
            self._jacobian_poe(frames, ifunc)
        else:
            raise NotImplementedError(
                f"kinematics for convention={convention!r} not implemented"
            )

        # 连杆 l 原点的几何雅可比：[v; ω] = J[l] dq，二者均在基座系表达
        self.J = [self.Jp[l].col_join(self.Jo[l]) for l in range(self.dof)]

        # 质心雅可比：角速度与连杆原点相同，线速度需由原点平移到质心。
        self.Jco = self.Jo
        self.Jcp = [None] * self.dof
        self.Jc = [None] * self.dof
        for l in range(self.dof):
            self.Jcp[l] = ifunc(
                self.Jp[l]
                - skew(frames.R[l] * sympy.Matrix(symbols.r[l])) * self.Jo[l]
            )
            self.Jc[l] = self.Jcp[l].col_join(self.Jco[l])

        # 解析位姿 xa=[p; rpy] 与解析雅可比 Ja=∂xa/∂q；仅末端在构造时填充
        self.rpy = [None] * self.dof  # ZYX 欧拉角 [roll, pitch, yaw]
        self.xa = [None] * self.dof
        self.Ja = [None] * self.dof  # 符号 Ja 按需；默认用 ja_numpy
        if analytical:
            self._init_analytical_pose(frames)

    def _init_analytical_pose(self, frames) -> None:
        """末端解析位姿 ``xa=[p; rpy]``（ZYX）。"""
        l = self.dof - 1
        phi = rpy_zyx_from_R(frames.R[l])
        self.rpy[l] = phi
        self.xa[l] = frames.p[l].col_join(phi)

    def gen_analytical_symbolic(self, link: int = -1, ifunc: Callable | None = None):
        """显式生成符号 ``Ja=∂xa/∂q``（高自由度可能很慢）。"""
        ifunc = ifunc or identity
        idx = self.dof - 1 if link < 0 else link
        if self.xa[idx] is None:
            phi = rpy_zyx_from_R(self.frames.R[idx])
            self.rpy[idx] = phi
            self.xa[idx] = self.frames.p[idx].col_join(phi)
        q = sympy.Matrix(self.symbols.q)
        self.Ja[idx] = ifunc(self.xa[idx].jacobian(q))
        return self.Ja[idx]

    def ja_numpy(self, q, link: int = -1, eps: float = 1e-7):
        """解析雅可比数值：对 ``xa=[p;rpy]`` **直接差分**（不是由几何 J 转换）。"""
        import numpy as np

        idx = self.dof - 1 if link < 0 else link
        if self.xa[idx] is None:
            phi = rpy_zyx_from_R(self.frames.R[idx])
            self.rpy[idx] = phi
            self.xa[idx] = self.frames.p[idx].col_join(phi)

        q = np.asarray(q, dtype=float).reshape(-1)
        xa_fun = sympy.lambdify(list(self.symbols.q), self.xa[idx], "numpy")
        xa0 = np.asarray(xa_fun(*q), dtype=float).reshape(6)
        Ja = np.zeros((6, self.dof))
        for j in range(self.dof):
            dq = np.zeros(self.dof)
            dq[j] = eps
            xa1 = np.asarray(xa_fun(*(q + dq)), dtype=float).reshape(6)
            # 角度差分折到 (-π, π]，避免 ±π 跳变
            d = xa1 - xa0
            d[3:] = (d[3:] + np.pi) % (2 * np.pi) - np.pi
            Ja[:, j] = d / eps
        return Ja

    def ja_from_geometric(self, link: int = -1) -> sympy.Matrix:
        """用 ``Ja=[Jp; E^{-1} Jo]`` 由几何雅可比转换（对照关系用）。"""
        idx = self.dof - 1 if link < 0 else link
        if self.rpy[idx] is None:
            self.rpy[idx] = rpy_zyx_from_R(self.frames.R[idx])
        return analytical_from_geometric(self.J[idx], self.rpy[idx])

    def ja_from_geometric_numpy(self, q, link: int = -1):
        """数值版：``Ja = [Jp; E(φ)^{-1} Jo]``（对照关系）。"""
        import numpy as np

        idx = self.dof - 1 if link < 0 else link
        if self.rpy[idx] is None:
            self.rpy[idx] = rpy_zyx_from_R(self.frames.R[idx])
        q = np.asarray(q, dtype=float).reshape(-1)
        Jg = np.asarray(
            sympy.lambdify(list(self.symbols.q), self.J[idx], "numpy")(*q),
            dtype=float,
        )
        phi = np.asarray(
            sympy.lambdify(list(self.symbols.q), self.rpy[idx], "numpy")(*q),
            dtype=float,
        ).reshape(3)
        r, p, y = sympy.symbols("roll pitch yaw", real=True)
        E = np.asarray(
            sympy.lambdify((r, p, y), rpy_zyx_E([r, p, y]), "numpy")(*phi),
            dtype=float,
        )
        return np.vstack([Jg[:3], np.linalg.inv(E) @ Jg[3:]])

    def _jacobian_standard(self, frames, sigma, ifunc) -> None:
        z_ext = frames.z + [sympy.Matrix([0, 0, 1])]
        p_ext = frames.p + [sympy.zeros(3, 1)]
        for l in range(self.dof):
            self.Jp[l] = sympy.zeros(3, self.dof)
            self.Jo[l] = sympy.zeros(3, self.dof)
            for j in range(l + 1):
                if sigma[j]:
                    self.Jp[l][0:3, j] = ifunc(z_ext[j - 1])
                    self.Jo[l][0:3, j] = sympy.zeros(3, 1)
                else:
                    self.Jp[l][0:3, j] = ifunc(
                        z_ext[j - 1].cross(
                            (p_ext[l] - p_ext[j - 1]).reshape(3, 1)
                        )
                    )
                    self.Jo[l][0:3, j] = ifunc(z_ext[j - 1])

    def _jacobian_modified(self, frames, sigma, ifunc) -> None:
        for l in range(self.dof):
            self.Jp[l] = sympy.zeros(3, self.dof)
            self.Jo[l] = sympy.zeros(3, self.dof)
            for j in range(l + 1):
                if sigma[j]:
                    self.Jp[l][0:3, j] = ifunc(frames.z[j])
                    self.Jo[l][0:3, j] = sympy.zeros(3, 1)
                else:
                    self.Jp[l][0:3, j] = ifunc(
                        frames.z[j].cross(
                            (frames.p[l] - frames.p[j]).reshape(3, 1)
                        )
                    )
                    self.Jo[l][0:3, j] = ifunc(frames.z[j])

    def _jacobian_poe(self, frames, ifunc) -> None:
        if frames.S_space is None or frames.E is None:
            raise ValueError("PoE FrameChain missing S_space / E")

        S_inst = [None] * self.dof
        Tacc = sympy.eye(4)
        for j in range(self.dof):
            S_inst[j] = ifunc(adjoint_twist(Tacc, frames.S_space[j]))
            Tacc = ifunc(Tacc * frames.E[j])

        for l in range(self.dof):
            self.Jp[l] = sympy.zeros(3, self.dof)
            self.Jo[l] = sympy.zeros(3, self.dof)
            p_l = frames.p[l]
            for j in range(l + 1):
                w = S_inst[j][0:3, 0]
                v_s = S_inst[j][3:6, 0]
                self.Jo[l][0:3, j] = ifunc(w)
                self.Jp[l][0:3, j] = ifunc(v_s + w.cross(p_l).reshape(3, 1))
