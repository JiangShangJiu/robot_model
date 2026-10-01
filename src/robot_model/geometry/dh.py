"""DH / MDH 连杆参数表。"""

from __future__ import annotations

from typing import Sequence
from unicodedata import east_asian_width

import sympy

from .conventions import normalize_dh_convention

_JOINT = sympy.symbols("q", real=True)
# 内部存储顺序仍是 (α, a, d, θ)；打印列顺序为 a, d, α, θ
_PRINT_COLS = (
    ("a", 1),
    ("d", 2),
    ("α", 0),
    ("θ", 3),
)
_HEADERS = ("Joint",) + tuple(name for name, _ in _PRINT_COLS) + ("关节类型",)
_SIGMA_LABEL = {0: "转动关节", 1: "移动关节"}


def _fmt_parm(expr) -> str:
    """表格用短字符串：真正的 0 显示为 0；大分母分数按小数显示。"""
    e = sympy.sympify(expr)
    if e == 0:
        return "0"
    if isinstance(e, sympy.Rational) and e.q > 16:
        text = f"{float(e):.12g}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    if isinstance(e, sympy.Float):
        text = f"{float(e):.12g}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    text = str(e)
    if text in ("0", "0.0", "0.e-0"):
        return "0"
    return text


def _disp_width(text: str) -> int:
    return sum(2 if east_asian_width(ch) in ("F", "W") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _disp_width(text))


def _clean_scalar(expr) -> sympy.Expr:
    """清理数值：贴合 0 / ±π/2；长度保留短小数，避免 333/1000。"""
    e = sympy.simplify(sympy.sympify(expr))
    if e.free_symbols:
        return e
    try:
        f = float(e.evalf())
    except (TypeError, ValueError):
        return e
    if abs(f) < 1e-12:
        return sympy.Integer(0)
    for cand in (
        0,
        sympy.pi / 2,
        -sympy.pi / 2,
        sympy.pi,
        -sympy.pi,
        3 * sympy.pi / 2,
        -3 * sympy.pi / 2,
        sympy.pi / 4,
        -sympy.pi / 4,
    ):
        if abs(f - float(cand)) < 1e-8:
            return sympy.sympify(cand)
    rat = sympy.Rational(f).limit_denominator(32)
    if rat.q <= 16 and abs(float(rat) - f) < 1e-12:
        return rat
    return sympy.Float(f"{f:.12g}")


def decompose_standard_dh(T: sympy.Matrix) -> tuple:
    """由相邻齐次变换反解标准 DH ``(α,a,d,θ)``。"""
    T = sympy.Matrix(T)
    cth, sth = T[0, 0], T[1, 0]
    theta = sympy.atan2(sth, cth)
    sal, cal = T[2, 1], T[2, 2]
    alpha = sympy.atan2(sal, cal)
    d = T[2, 3]
    a = T[0, 3] * cth + T[1, 3] * sth
    return tuple(_clean_scalar(x) for x in (alpha, a, d, theta))


def decompose_modified_dh(T: sympy.Matrix) -> tuple:
    """由相邻齐次变换反解修正 DH ``(α,a,d,θ)``。"""
    T = sympy.Matrix(T)
    cth, sth = T[0, 0], -T[0, 1]
    theta = sympy.atan2(sth, cth)
    sal, cal = -T[1, 2], T[2, 2]
    alpha = sympy.atan2(sal, cal)
    a = T[0, 3]
    d = -sal * T[1, 3] + cal * T[2, 3]
    return tuple(_clean_scalar(x) for x in (alpha, a, d, theta))


def _with_joint(const, q_i, movable: bool):
    const = _clean_scalar(const)
    if not movable:
        return const
    if const == 0:
        return q_i
    return sympy.simplify(q_i + const)


def _shift_mdh_to_standard(
    parms: Sequence[Sequence[sympy.Expr]],
) -> list[tuple[sympy.Expr, sympy.Expr, sympy.Expr, sympy.Expr]]:
    """Craig 近端 MDH → 远端标准 DH：同行保留 (d,θ)，(α,a) 上移一行。"""
    n = len(parms)
    out: list[tuple] = []
    for i in range(n):
        alpha, a = (parms[i + 1][0], parms[i + 1][1]) if i + 1 < n else (0, 0)
        out.append((sympy.sympify(alpha), sympy.sympify(a), parms[i][2], parms[i][3]))
    return out


def _shift_standard_to_mdh(
    parms: Sequence[Sequence[sympy.Expr]],
) -> list[tuple[sympy.Expr, sympy.Expr, sympy.Expr, sympy.Expr]]:
    """标准 DH → Craig MDH：同行保留 (d,θ)，(α,a) 下移一行（首行 α,a → 0）。"""
    n = len(parms)
    out: list[tuple] = []
    for i in range(n):
        alpha, a = (parms[i - 1][0], parms[i - 1][1]) if i > 0 else (0, 0)
        out.append((sympy.sympify(alpha), sympy.sympify(a), parms[i][2], parms[i][3]))
    return out


class DHParams:
    """标准 DH 或 MDH 的 ``(α, a, d, θ)`` 表（已绑定关节符号）。"""

    def __init__(
        self,
        parms: Sequence[Sequence[sympy.Expr]],
        sigma: Sequence[int],
        convention: str,
    ):
        self.parms = [tuple(row) for row in parms]
        self.sigma = list(sigma)
        self.convention = normalize_dh_convention(convention)
        self.dof = len(self.parms)
        if len(self.sigma) != self.dof:
            raise ValueError("sigma length must match dof")

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Sequence],
        q: sympy.Matrix,
        convention: str,
    ) -> DHParams:
        """解析行表；字符串 ``'q'`` 替换为对应 ``q[i]``。"""
        dof = len(rows)
        if q.shape[0] != dof:
            raise ValueError(f"q length {q.shape[0]} != number of links {dof}")

        parsed: list[tuple] = []
        sigma = [0] * dof
        q_generic = _JOINT
        for i, row in enumerate(rows):
            if len(row) != 4:
                raise ValueError("each DH/MDH row must be (alpha, a, d, theta)")
            link = []
            for p in row:
                p = sympy.sympify(p)
                for s in list(p.free_symbols):
                    if str(s) == str(q_generic):
                        p = p.subs(s, q[i])
                link.append(p)
            parsed.append(tuple(link))
            if parsed[i][3].has(q[i]):
                sigma[i] = 0
            if parsed[i][2].has(q[i]):
                sigma[i] = 1
        return cls(parsed, sigma, convention)

    def _rows(self) -> list[list[str]]:
        rows: list[list[str]] = []
        for i, row in enumerate(self.parms):
            rows.append(
                [
                    f"Joint{i + 1}",
                    *[_fmt_parm(row[idx]) for _, idx in _PRINT_COLS],
                    _SIGMA_LABEL.get(self.sigma[i], str(self.sigma[i])),
                ]
            )
        return rows

    def format_table(self) -> str:
        """等宽对齐的纯文本表（``print`` 也整齐）。"""
        label = "MDH" if self.convention == "modified" else "DH"
        body = self._rows()
        widths = [
            max(_disp_width(h), *(_disp_width(r[j]) for r in body))
            for j, h in enumerate(_HEADERS)
        ]

        def line(parts: Sequence[str]) -> str:
            return "  ".join(_pad(p, widths[j]) for j, p in enumerate(parts))

        rule = "  ".join("-" * w for w in widths)
        lines = [
            f"{label} ({self.convention}), dof={self.dof}",
            line(_HEADERS),
            rule,
            *[line(r) for r in body],
        ]
        return "\n".join(lines)

    def _html_table(self) -> str:
        label = "MDH" if self.convention == "modified" else "DH"
        th = "".join(f"<th>{h}</th>" for h in _HEADERS)
        trs = []
        for row in self._rows():
            tds = "".join(f"<td>{c}</td>" for c in row)
            trs.append(f"<tr>{tds}</tr>")
        return (
            f"<b>{label}</b> (<code>{self.convention}</code>), dof={self.dof}"
            f'<table border="1" style="border-collapse:collapse;text-align:center">'
            f"<thead><tr>{th}</tr></thead>"
            f"<tbody>{''.join(trs)}</tbody></table>"
        )

    def print_table(self) -> None:
        """Notebook 渲染 HTML 表；否则打印等宽文本。"""
        try:
            from IPython.display import HTML, display

            display(HTML(self._html_table()))
        except Exception:
            print(self.format_table())

    def to_convention(self, convention: str) -> DHParams:
        """DH ↔ MDH 参数表转换（Craig 近端/远端索引平移）。

        - 同行的 ``d, θ``（及 ``sigma``）不变；``α, a`` 按约定上/下移一行。
        - MDH→DH：末行 ``α,a`` 置 0（末端法兰需另建工具系时自行补上）。
        - DH→MDH：首行 ``α,a`` 置 0。
        - 中间连杆坐标系一般与原约定不同；这不是逐元素改写同一套齐次变换。

        若源表 MDH 首行已是 ``α=a=0``，则 ``to_convention('standard').to_convention('modified')``
        可回到原表。
        """
        target = normalize_dh_convention(convention)
        if target == self.convention:
            return DHParams(self.parms, self.sigma, self.convention)

        if self.convention == "modified" and target == "standard":
            parms = _shift_mdh_to_standard(self.parms)
        elif self.convention == "standard" and target == "modified":
            parms = _shift_standard_to_mdh(self.parms)
        else:
            raise ValueError(f"cannot convert {self.convention!r} → {target!r}")
        return DHParams(parms, self.sigma, target)
