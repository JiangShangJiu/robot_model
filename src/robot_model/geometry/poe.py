"""PoE / 旋量（空间指数积）运动学描述。"""

from __future__ import annotations

from typing import Sequence

import sympy

from ..utils.mathutil import adjoint_twist, inverse_T
from .conventions import normalize_dh_convention
from .dh import (
    DHParams,
    _with_joint,
    decompose_modified_dh,
    decompose_standard_dh,
)


def _as_screw(S) -> sympy.Matrix:
    S = sympy.Matrix(S).reshape(6, 1)
    return S.applyfunc(sympy.sympify)


def _as_SE3(M) -> sympy.Matrix:
    M = sympy.Matrix(M)
    if M.shape != (4, 4):
        raise ValueError(f"home pose M must be 4x4, got {M.shape}")
    return M.applyfunc(sympy.sympify)


class PoEParams:
    """空间形式 PoE：``T_i(q)=e^{[S_1]q_1}…e^{[S_i]q_i} M_i``。

    - ``screws[i]``：零位下第 ``i`` 关节的空间运动旋量 ``(ω,v)``（6×1）
    - ``M[i]``：零位下第 ``i`` 连杆坐标系相对基座的齐次位姿（4×4）
    - ``sigma[i]``：``0`` 转动，``1`` 移动
    - ``home_convention``：``M_i`` 来自 ``standard`` / ``modified``（用于反解 DH）
    """

    convention = "poe"

    def __init__(
        self,
        screws: Sequence,
        M,
        sigma: Sequence[int] | None = None,
        *,
        home_convention: str | None = None,
    ):
        self.screws = [_as_screw(s) for s in screws]
        self.dof = len(self.screws)
        if self.dof < 1:
            raise ValueError("need at least one screw")

        if isinstance(M, (list, tuple)) and len(M) == self.dof:
            self.M = [_as_SE3(m) for m in M]
        else:
            Mee = _as_SE3(M)
            self.M = [sympy.eye(4) for _ in range(self.dof - 1)] + [Mee]

        self.sigma = list(sigma) if sigma is not None else [0] * self.dof
        if len(self.sigma) != self.dof:
            raise ValueError("sigma length must match dof")

        self.home_convention = (
            None
            if home_convention is None
            else normalize_dh_convention(home_convention)
        )

    @classmethod
    def from_body_screws(
        cls,
        screws: Sequence,
        M,
        sigma: Sequence[int] | None = None,
        *,
        home_convention: str | None = None,
    ) -> PoEParams:
        """物体形式 PoE：``T=M e^{[B_1]q_1}…e^{[B_n]q_n}``。

        ``screws[i]`` 为**末端零位系**下的关节旋量 ``B_i``，按
        ``S_i = Ad_{M_ee} B_i`` 转成空间旋量后复用空间形式实现。
        """
        n = len(screws)
        if isinstance(M, (list, tuple)) and len(M) == n:
            M_ee = _as_SE3(M[-1])
        else:
            M_ee = _as_SE3(M)
        space = [adjoint_twist(M_ee, _as_screw(B)) for B in screws]
        return cls(space, M, sigma=sigma, home_convention=home_convention)

    @classmethod
    def from_zero_config(cls, frames, sigma: Sequence[int] | None = None) -> PoEParams:
        """由 DH/MDH 零位提取空间旋量与 ``M_i``。

        - 修正 DH：关节 ``i`` 绕帧 ``i`` 的 ``z``（原点在轴上）
        - 标准 DH：关节 ``i`` 绕帧 ``i-1`` 的 ``z``（``i=0`` 时为基座 ``z``）
        """
        q = frames.symbols.q
        zeros = {q[i]: 0 for i in range(frames.dof)}
        sig = list(sigma) if sigma is not None else list(frames.sigma)
        conv = frames.convention

        Ms: list[sympy.Matrix] = []
        screws: list[sympy.Matrix] = []
        for i in range(frames.dof):
            Ti0 = sympy.Matrix(frames.T[i].subs(zeros)).applyfunc(sympy.simplify)
            Ms.append(Ti0)

            if conv == "modified":
                z = Ti0[0:3, 2]
                p = Ti0[0:3, 3]
            elif conv == "standard":
                if i == 0:
                    z = sympy.Matrix([0, 0, 1])
                    p = sympy.zeros(3, 1)
                else:
                    Tprev = sympy.Matrix(frames.T[i - 1].subs(zeros)).applyfunc(
                        sympy.simplify
                    )
                    z = Tprev[0:3, 2]
                    p = Tprev[0:3, 3]
            else:
                raise ValueError(
                    f"from_zero_config unsupported convention={conv!r}"
                )

            if sig[i] == 1:
                S = sympy.zeros(3, 1).col_join(z)
            else:
                S = z.col_join((-z.cross(p)).reshape(3, 1))
            screws.append(sympy.simplify(S))
        return cls(screws, Ms, sigma=sig, home_convention=conv)

    def _decompose_rows(self, q: sympy.Matrix, convention: str) -> list[tuple]:
        decompose = (
            decompose_modified_dh
            if convention == "modified"
            else decompose_standard_dh
        )
        prev = sympy.eye(4)
        rows: list[tuple] = []
        for i in range(self.dof):
            A = sympy.simplify(inverse_T(prev) * self.M[i])
            prev = self.M[i]
            alpha, a, d0, th0 = decompose(A)
            if self.sigma[i] == 0:
                theta = _with_joint(th0, q[i], True)
                d = d0
            else:
                theta = th0
                d = _with_joint(d0, q[i], True)
            rows.append((alpha, a, d, theta))
        return rows

    def to_dh_params(self, q: sympy.Matrix, convention: str) -> DHParams:
        """由各连杆零位 ``M_i`` 反解 DH/MDH。

        ``M_i`` 的几何来自 ``home_convention``：先按该约定反解，再 Craig 平移到目标约定。
        手写 PoE 未标注 ``home_convention`` 时，默认按 ``modified`` 反解。
        """
        target = normalize_dh_convention(convention)
        if q.shape[0] != self.dof:
            raise ValueError("q length must match PoE dof")

        source = self.home_convention or "modified"
        rows = self._decompose_rows(q, source)
        desc = DHParams(rows, self.sigma, source)
        if source == target:
            return desc
        return desc.to_convention(target)

    def format_table(self) -> str:
        """空间旋量表（等宽文本）。"""
        from unicodedata import east_asian_width

        headers = ("Joint", "ωx", "ωy", "ωz", "vx", "vy", "vz", "关节类型")
        sigma_label = {0: "转动关节", 1: "移动关节"}

        def fmt_num(x) -> str:
            e = sympy.simplify(x)
            if e == 0:
                return "0"
            if isinstance(e, sympy.Float) or getattr(e, "is_Float", False):
                return f"{float(e):.6g}"
            return str(e)

        def disp_w(s: str) -> int:
            return sum(2 if east_asian_width(ch) in ("F", "W") else 1 for ch in s)

        def pad(s: str, w: int) -> str:
            return s + " " * max(0, w - disp_w(s))

        body: list[list[str]] = []
        for i, S in enumerate(self.screws):
            body.append(
                [
                    f"Joint{i + 1}",
                    *[fmt_num(S[k]) for k in range(6)],
                    sigma_label.get(self.sigma[i], str(self.sigma[i])),
                ]
            )
        widths = [
            max(disp_w(h), *(disp_w(r[j]) for r in body))
            for j, h in enumerate(headers)
        ]

        def line(parts):
            return "  ".join(pad(p, widths[j]) for j, p in enumerate(parts))

        home = self.home_convention or "?"
        return "\n".join(
            [
                f"PoE screws (space), dof={self.dof}, home_convention={home}",
                line(headers),
                "  ".join("-" * w for w in widths),
                *[line(r) for r in body],
            ]
        )

    def print_table(self) -> None:
        text = self.format_table()
        try:
            from IPython.display import HTML, display

            sigma_label = {0: "转动关节", 1: "移动关节"}
            headers = ("Joint", "ωx", "ωy", "ωz", "vx", "vy", "vz", "关节类型")
            th = "".join(f"<th>{h}</th>" for h in headers)
            trs = []
            for i, S in enumerate(self.screws):
                cells = [f"Joint{i + 1}"]
                for k in range(6):
                    e = sympy.simplify(S[k])
                    if e == 0:
                        cells.append("0")
                    elif isinstance(e, sympy.Float) or getattr(e, "is_Float", False):
                        cells.append(f"{float(e):.6g}")
                    else:
                        cells.append(str(e))
                cells.append(sigma_label.get(self.sigma[i], str(self.sigma[i])))
                trs.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            home = self.home_convention or "?"
            html = (
                f"<b>PoE</b> screws (space), dof={self.dof}, "
                f"home_convention=<code>{home}</code>"
                f'<table border="1" style="border-collapse:collapse;text-align:center">'
                f"<thead><tr>{th}</tr></thead>"
                f"<tbody>{''.join(trs)}</tbody></table>"
            )
            display(HTML(html))
        except Exception:
            print(text)
