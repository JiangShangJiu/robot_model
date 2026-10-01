"""数值 RNE / 回归矩阵：适合 7 轴等不宜全符号展开的情况。"""

from __future__ import annotations

from typing import Callable

import numpy as np
import sympy

from ..utils.lie import Adj_np, Adjdual_np, adj_np, adjdual_np, skew_np
from .methods import RneMethod, resolve_rne_method

# 局部别名，保持递推代码可读
_skew = skew_np
_Adj = Adj_np
_Adjdual = Adjdual_np
_adj = adj_np
_adjdual = adjdual_np


def _tensor_L(Le6) -> np.ndarray:
    xx, xy, xz, yy, yz, zz = np.asarray(Le6, dtype=float).reshape(6)
    return np.array(
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float
    )


def _gravity_vec(symbols) -> np.ndarray:
    g = symbols.gravityacc
    try:
        return np.array([float(g[0]), float(g[1]), float(g[2])], dtype=float)
    except TypeError as exc:
        free = ", ".join(sorted(str(s) for s in g.free_symbols))
        raise TypeError(
            f"数值动力学要求 gravityacc 为常数，当前含符号 {{{free}}}。"
            "请在构造时传数值重力（如 gravityacc=[0, 0, -9.81]），"
            "或改走符号路径（gen_invdyn / gen_regressor / gen_gravityterm）。"
        ) from exc


class NumericDynamics:
    """缓存 lambdify 后的几何，做数值 Park/Khalil 与回归矩阵。"""

    def __init__(self, symbols, frames, *, method: RneMethod | None = None):
        self.symbols = symbols
        self.frames = frames
        self.dof = symbols.dof
        self.method = resolve_rne_method(self.frames, method)
        if self.method == "lagrange":
            # 数值回归用 RNE 列构造（与拉格朗日同一线性 H）
            self.method = (
                "park"
                if self.frames.convention in ("standard", "poe")
                else "khalil"
            )
        if self.method == "khalil" and self.frames.convention == "poe":
            raise NotImplementedError("numeric Khalil not available for PoE")

        qsyms = list(self.symbols.q)
        n = self.dof
        if self.method == "park":
            self._Tinv_fun = [
                sympy.lambdify(qsyms, self.frames.T_rel_inv[i], "numpy")
                for i in range(n)
            ]
            self._S_fun = [
                sympy.lambdify(qsyms, self.frames.S_body[i], "numpy")
                for i in range(n)
            ]
        else:
            self._R_fun = [
                sympy.lambdify(qsyms, self.frames.R_rel[i], "numpy")
                for i in range(n)
            ]
            self._p_fun = [
                sympy.lambdify(qsyms, self.frames.p_rel[i], "numpy")
                for i in range(n)
            ]
        self.dynparms = list(self.symbols.dynparms())
        self.n_dynparms = len(self.dynparms)
        self._parm_slots = self._build_parm_slots()
        self._rigid_groups = self._build_rigid_groups()

    def _build_parm_slots(self):
        """每个 dynparm → 写入 (m,l,Le,Ia,fv,fc,fo) 的位置。"""
        sym = self.symbols
        slots = []
        for parm in self.dynparms:
            kind = None
            for i in range(self.dof):
                if parm == sym.m[i]:
                    kind = ("m", i, None)
                    break
                for j in range(3):
                    if parm == sym.l[i][j]:
                        kind = ("l", i, j)
                        break
                if kind:
                    break
                for j in range(6):
                    if parm == sym.Le[i][j]:
                        kind = ("Le", i, j)
                        break
                if kind:
                    break
                if parm == sym.Ia[i]:
                    kind = ("Ia", i, None)
                    break
                if sym.fv is not None and parm == sym.fv[i]:
                    kind = ("fv", i, None)
                    break
                if sym.fc is not None and parm == sym.fc[i]:
                    kind = ("fc", i, None)
                    break
                if sym.fo is not None and parm == sym.fo[i]:
                    kind = ("fo", i, None)
                    break
            if kind is None:
                raise ValueError(f"cannot map dynparm {parm}")
            slots.append(kind)
        return slots

    def _build_rigid_groups(self):
        """每连杆的惯性单位基；列位置由参数映射决定，兼容两种参数顺序。"""
        groups = []
        for link in range(self.dof):
            cols = [
                c for c, (kind, i, _) in enumerate(self._parm_slots)
                if i == link and kind in ("m", "l", "Le")
            ]
            mass = np.zeros(len(cols))
            first_moment = np.zeros((len(cols), 3))
            inertia = np.zeros((len(cols), 3, 3))
            spatial = np.zeros((len(cols), 6, 6))
            for local, col in enumerate(cols):
                kind, _, component = self._parm_slots[col]
                if kind == "m":
                    mass[local] = 1.0
                elif kind == "l":
                    first_moment[local, component] = 1.0
                else:
                    elements = np.zeros(6)
                    elements[component] = 1.0
                    inertia[local] = _tensor_L(elements)
                sk_l = _skew(first_moment[local])
                spatial[local, :3, :3] = inertia[local]
                spatial[local, :3, 3:] = sk_l
                spatial[local, 3:, :3] = -sk_l
                spatial[local, 3:, 3:] = mass[local] * np.eye(3)
            groups.append((np.asarray(cols, dtype=int), mass, first_moment,
                           inertia, spatial))
        return groups

    def _unit_params(self, col: int):
        n = self.dof
        m = np.zeros(n)
        l = np.zeros((n, 3))
        Le = np.zeros((n, 6))
        Ia = np.zeros(n)
        fv = np.zeros(n)
        fc = np.zeros(n)
        fo = np.zeros(n)
        kind, i, j = self._parm_slots[col]
        if kind == "m":
            m[i] = 1.0
        elif kind == "l":
            l[i, j] = 1.0
        elif kind == "Le":
            Le[i, j] = 1.0
        elif kind == "Ia":
            Ia[i] = 1.0
        elif kind == "fv":
            fv[i] = 1.0
        elif kind == "fc":
            fc[i] = 1.0
        elif kind == "fo":
            fo[i] = 1.0
        return m, l, Le, Ia, fv, fc, fo

    def _park_forward(self, q, dq, ddq):
        g = _gravity_vec(self.symbols)
        n = self.dof
        qq = [float(x) for x in np.asarray(q, float).reshape(-1)]
        dq = np.asarray(dq, float).reshape(-1)
        ddq = np.asarray(ddq, float).reshape(-1)
        Tinv = [
            np.asarray(f(*qq), float).reshape(4, 4) for f in self._Tinv_fun
        ]
        S = [np.asarray(f(*qq), float).reshape(6) for f in self._S_fun]
        V = [None] * (n + 1)
        dV = [None] * (n + 1)
        V[-1] = np.zeros(6)
        dV[-1] = np.hstack([np.zeros(3), -g])
        for i in range(n):
            V[i] = _Adj(Tinv[i], V[i - 1]) + S[i] * dq[i]
            dV[i] = (
                S[i] * ddq[i]
                + _Adj(Tinv[i], dV[i - 1])
                + _adj(_Adj(Tinv[i], V[i - 1]), S[i] * dq[i])
            )
        return Tinv, S, V, dV, dq, ddq

    def _park_backward(self, Tinv, S, V, dV, dq, ddq, m, l, Le, Ia, fv, fc, fo):
        n = self.dof
        F = [None] * (n + 1)
        F[n] = np.zeros(6)
        Tinv_ext = Tinv + [np.eye(4)]
        tau = np.zeros(n)
        for i in range(n - 1, -1, -1):
            L = _tensor_L(Le[i])
            sk_l = _skew(l[i])
            Llm = np.block([[L, sk_l], [-sk_l, m[i] * np.eye(3)]])
            F[i] = (
                _Adjdual(Tinv_ext[i + 1], F[i + 1])
                + Llm @ dV[i]
                - _adjdual(V[i], Llm @ V[i])
            )
            fric = fv[i] * dq[i] + fo[i]
            if fc[i] != 0.0:
                fric += fc[i] * np.sign(dq[i])
            tau[i] = float(S[i] @ F[i]) + fric + Ia[i] * ddq[i]
        return tau

    def invdyn_park(self, q, dq, ddq, m, l, Le, Ia=None, fv=None, fc=None, fo=None):
        n = self.dof
        Ia = np.zeros(n) if Ia is None else np.asarray(Ia, float).reshape(-1)
        fv = np.zeros(n) if fv is None else np.asarray(fv, float).reshape(-1)
        fc = np.zeros(n) if fc is None else np.asarray(fc, float).reshape(-1)
        fo = np.zeros(n) if fo is None else np.asarray(fo, float).reshape(-1)
        Tinv, S, V, dV, dq, ddq = self._park_forward(q, dq, ddq)
        return self._park_backward(
            Tinv, S, V, dV, dq, ddq, m, l, Le, Ia, fv, fc, fo
        )

    def _khalil_forward(self, q, dq, ddq):
        g = _gravity_vec(self.symbols)
        n = self.dof
        conv = self.frames.convention
        qq = [float(x) for x in np.asarray(q, float).reshape(-1)]
        dq = np.asarray(dq, float).reshape(-1)
        ddq = np.asarray(ddq, float).reshape(-1)
        Rdh = [np.asarray(f(*qq), float).reshape(3, 3) for f in self._R_fun]
        pdh = [np.asarray(f(*qq), float).reshape(3) for f in self._p_fun]
        sigma = list(self.frames.sigma)
        z = np.array([0.0, 0.0, 1.0])

        w = [None] * (n + 1)
        dw = [None] * (n + 1)
        dV = [None] * (n + 1)
        U = [None] * (n + 1)
        w[-1] = np.zeros(3)
        dw[-1] = np.zeros(3)
        dV[-1] = -g
        U[-1] = np.zeros((3, 3))
        for i in range(n):
            s = sigma[i]
            ns = 1 - s
            if conv == "modified":
                w_pj = Rdh[i].T @ w[i - 1]
                w[i] = w_pj + ns * dq[i] * z
                dw[i] = Rdh[i].T @ dw[i - 1] + ns * (
                    ddq[i] * z + np.cross(w_pj, dq[i] * z)
                )
                dV[i] = Rdh[i].T @ (dV[i - 1] + U[i - 1] @ pdh[i]) + s * (
                    ddq[i] * z + 2 * np.cross(w_pj, dq[i] * z)
                )
            else:
                w_l = w[i - 1] + ns * dq[i] * z
                dw_l = dw[i - 1] + ns * (
                    ddq[i] * z + np.cross(w[i - 1], dq[i] * z)
                )
                U_l = _skew(dw_l) + _skew(w_l) @ _skew(w_l)
                w[i] = Rdh[i].T @ w_l
                dw[i] = Rdh[i].T @ dw_l
                z_b = Rdh[i].T @ z
                dV[i] = Rdh[i].T @ (dV[i - 1] + U_l @ pdh[i]) + s * (
                    ddq[i] * z_b
                    + 2 * np.cross(Rdh[i].T @ w_l, dq[i] * z_b)
                )
            U[i] = _skew(dw[i]) + _skew(w[i]) @ _skew(w[i])
        return Rdh, pdh, w, dw, dV, U, dq, ddq, sigma, conv, z

    def _khalil_backward(
        self, Rdh, pdh, w, dw, dV, U, dq, ddq, sigma, conv, z, m, l, Le, Ia, fv, fc, fo
    ):
        n = self.dof
        Rdh_e = Rdh + [np.eye(3)]
        pdh_e = pdh + [np.zeros(3)]
        f = [None] * (n + 1)
        mm = [None] * (n + 1)
        f[n] = np.zeros(3)
        mm[n] = np.zeros(3)
        tau = np.zeros(n)
        for i in range(n - 1, -1, -1):
            s = sigma[i]
            ns = 1 - s
            L = _tensor_L(Le[i])
            Fi = m[i] * dV[i] + U[i] @ l[i]
            Mi = (
                L @ dw[i]
                + np.cross(w[i], L @ w[i])
                + np.cross(l[i], dV[i])
            )
            f_nj = Rdh_e[i + 1] @ f[i + 1]
            f[i] = Fi + f_nj
            mm[i] = Mi + Rdh_e[i + 1] @ mm[i + 1] + np.cross(
                pdh_e[i + 1], f_nj
            )
            if conv == "modified":
                tau[i] = float((s * f[i] + ns * mm[i]) @ z)
            else:
                fim1 = Rdh[i] @ f[i]
                mim1 = Rdh[i] @ mm[i] + np.cross(pdh[i], fim1)
                tau[i] = float((s * fim1 + ns * mim1) @ z)
            fric = fv[i] * dq[i] + fo[i]
            if fc[i] != 0.0:
                fric += fc[i] * np.sign(dq[i])
            tau[i] += fric + Ia[i] * ddq[i]
        return tau

    def invdyn_khalil(self, q, dq, ddq, m, l, Le, Ia=None, fv=None, fc=None, fo=None):
        n = self.dof
        Ia = np.zeros(n) if Ia is None else np.asarray(Ia, float).reshape(-1)
        fv = np.zeros(n) if fv is None else np.asarray(fv, float).reshape(-1)
        fc = np.zeros(n) if fc is None else np.asarray(fc, float).reshape(-1)
        fo = np.zeros(n) if fo is None else np.asarray(fo, float).reshape(-1)
        state = self._khalil_forward(q, dq, ddq)
        return self._khalil_backward(*state, m, l, Le, Ia, fv, fc, fo)

    def invdyn(self, q, dq, ddq, m, l, Le, Ia=None, fv=None, fc=None, fo=None):
        if self.method == "park":
            return self.invdyn_park(q, dq, ddq, m, l, Le, Ia, fv, fc, fo)
        return self.invdyn_khalil(q, dq, ddq, m, l, Le, Ia, fv, fc, fo)

    @staticmethod
    def _actuator_column(kind: str, joint: int, dq, ddq, dof: int) -> np.ndarray:
        """电机惯量/摩擦列不依赖刚体力递推，直接写对应分量。"""
        tau = np.zeros(dof)
        if kind == "Ia":
            tau[joint] = float(ddq[joint])
        elif kind == "fv":
            tau[joint] = float(dq[joint])
        elif kind == "fc":
            tau[joint] = float(np.sign(dq[joint]))
        elif kind == "fo":
            tau[joint] = 1.0
        else:
            raise ValueError(f"not an actuator column: {kind}")
        return tau

    def regressor(self, q, dq, ddq) -> np.ndarray:
        """数值 ``H(q,dq,ddq)``，形状 ``(dof, n_dynparms)``。

        正向递推只算一次；每连杆的 10 个刚体参数列合批向根传播，
        跳过其下游的零力。``Ia/fv/fc/fo`` 列直接写入对应关节。
        """
        H = np.zeros((self.dof, self.n_dynparms))
        if self.method == "park":
            state = self._park_forward(q, dq, ddq)
            self._park_regressor(H, *state)
            dq_a, ddq_a = state[4], state[5]
        else:
            state = self._khalil_forward(q, dq, ddq)
            self._khalil_regressor(H, *state)
            dq_a, ddq_a = state[6], state[7]
        for c, (kind, joint, _) in enumerate(self._parm_slots):
            if kind in ("Ia", "fv", "fc", "fo"):
                H[:, c] = self._actuator_column(kind, joint, dq_a, ddq_a, self.dof)
        return H

    def _park_regressor(self, H, Tinv, S, V, dV, dq, ddq):
        # Ad_T^* 在同一样本的各参数批次间复用。
        force_transforms = []
        for transform in Tinv:
            R, p = transform[:3, :3], transform[:3, 3]
            dual = np.zeros((6, 6))
            dual[:3, :3] = R.T
            dual[:3, 3:] = (_skew(p) @ R).T
            dual[3:, 3:] = R.T
            force_transforms.append(dual)
        for link, (cols, _, _, _, spatial) in enumerate(self._rigid_groups):
            velocity_dual = np.zeros((6, 6))
            velocity_dual[:3, :3] = _skew(V[link][:3]).T
            velocity_dual[:3, 3:] = _skew(V[link][3:]).T
            velocity_dual[3:, 3:] = _skew(V[link][:3]).T
            force = (spatial @ dV[link]).T - velocity_dual @ (spatial @ V[link]).T
            H[link, cols] = S[link] @ force
            for joint in range(link - 1, -1, -1):
                force = force_transforms[joint + 1] @ force
                H[joint, cols] = S[joint] @ force

    def _khalil_regressor(
        self, H, Rdh, pdh, w, dw, dV, U, dq, ddq, sigma, conv, z
    ):
        for link, (cols, mass, first_moment, inertia, _) in enumerate(self._rigid_groups):
            force = dV[link][:, None] * mass + U[link] @ first_moment.T
            moment = (
                (inertia @ dw[link]).T
                + np.cross(w[link], inertia @ w[link]).T
                + np.cross(first_moment, dV[link]).T
            )
            for joint in range(link, -1, -1):
                if conv == "modified":
                    H[joint, cols] = (
                        sigma[joint] * force[2] + (1 - sigma[joint]) * moment[2]
                    )
                else:
                    parent_force = Rdh[joint] @ force
                    parent_moment = Rdh[joint] @ moment + np.cross(
                        pdh[joint], parent_force.T
                    ).T
                    H[joint, cols] = (
                        sigma[joint] * parent_force[2]
                        + (1 - sigma[joint]) * parent_moment[2]
                    )
                if joint > 0:
                    if conv == "modified":
                        parent_force = Rdh[joint] @ force
                        parent_moment = Rdh[joint] @ moment + np.cross(
                            pdh[joint], parent_force.T
                        ).T
                    force, moment = parent_force, parent_moment

    def as_regressor_func(self) -> Callable:
        def regressor_func(q, dq, ddq):
            return self.regressor(q, dq, ddq)

        return regressor_func
