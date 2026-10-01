"""Python / C 代码生成。

两种语言共用同一条流水线：符号先换成带下标的形参（``q1 -> q[0]``），
再做公共子表达式消除，最后交给对应的 sympy 打印器。
tau 这类量 cse 前后能差十几倍，所以 cse 不是可选项。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import sympy
from sympy.printing.c import C99CodePrinter
from sympy.printing.numpy import NumPyPrinter

from .collect import ARG_GROUPS, ExportItem, arg_symbols


@dataclass(frozen=True)
class Lowered:
    """已索引化 + cse 的中间结果，与目标语言无关。"""

    item: ExportItem
    args: tuple[str, ...]  # 形参名，如 ("q", "parms")
    temps: list[tuple[sympy.Symbol, sympy.Expr]]  # cse 中间变量
    body: sympy.Matrix  # 用中间变量表示的主体


def _index_subs(robot, uses: Sequence[str]) -> dict:
    subs = {}
    for group in uses:
        for i, sym in enumerate(arg_symbols(robot, group)):
            subs[sym] = sympy.Symbol(f"{group}[{i}]")
    return subs


def lower(robot, item: ExportItem, *, cse: bool = True) -> Lowered:
    """把一个 ``ExportItem`` 降低成可直接打印的形式。"""
    # 只是符号换符号，用 xreplace 而非 subs：subs 会逐节点重建并尝试化简，
    # 7 轴的 tau 有几百万个节点，两者差两个数量级（80s vs 0.3s）。
    expr = item.expr.xreplace(_index_subs(robot, item.uses))
    # 形参顺序固定为 ARG_GROUPS 的顺序，跨语言一致
    args = tuple(g for g in ARG_GROUPS if g in item.uses)
    if not cse:
        return Lowered(item=item, args=args, temps=[], body=sympy.Matrix(expr))
    temps, reduced = sympy.cse(expr, symbols=sympy.numbered_symbols("x"))
    return Lowered(
        item=item, args=args, temps=temps, body=sympy.Matrix(reduced[0])
    )


def parms_legend(robot, prefix: str = "#") -> list[str]:
    """``parms`` 各分量对应哪个符号——不写清楚，生成的代码没法用。"""
    syms = arg_symbols(robot, "parms")
    lines = [f"{prefix} parms 顺序（共 {len(syms)} 项，Khalil 排列）:"]
    per_link = 10 if len(syms) % 10 == 0 else None
    for i, s in enumerate(syms):
        mark = "  " if per_link is None or i % per_link else "* "
        lines.append(f"{prefix}   {mark}[{i:2d}] {s}")
    return lines


class _Backend:
    """语言后端：负责把 ``Lowered`` 变成一段函数文本。"""

    printer: sympy.printing.printer.Printer
    comment: str

    def function(self, low: Lowered) -> str:
        raise NotImplementedError

    def module(self, robot, lowered: Iterable[Lowered], name: str) -> str:
        raise NotImplementedError

    def _p(self, expr) -> str:
        return self.printer.doprint(expr)


class PythonBackend(_Backend):
    comment = "#"

    def __init__(self):
        self.printer = NumPyPrinter()

    def function(self, low: Lowered) -> str:
        item = low.item
        sig = ", ".join(low.args)
        rows, cols = item.shape
        out = [f"def {item.key}({sig}):", f'    """{item.doc}。']
        out.append("")
        out.append(f"    返回形状 {rows}x{cols}。")
        out.append('    """')
        for lhs, rhs in low.temps:
            out.append(f"    {lhs} = {self._p(rhs)}")
        elems = [
            [self._p(low.body[r, c]) for c in range(cols)] for r in range(rows)
        ]
        if cols == 1:
            # 列向量逐分量命名（tau_1 / tau_2 …）：tau 的单个分量就有上千字符，
            # 全挤进一个 return 既读不了也没法单独调试。下标随关节从 1 起。
            names = [f"{item.key}_{r + 1}" for r in range(rows)]
            for name, row in zip(names, elems):
                out.append(f"    {name} = {row[0]}")
            out.append(f"    return numpy.array([{', '.join(names)}])")
        else:
            out.append("    return numpy.array([")
            for row in elems:
                out.append("        [" + ", ".join(row) + "],")
            out.append("    ])")
        return "\n".join(out)

    def module(self, robot, lowered, name: str) -> str:
        head = [
            '"""由 robot_model 自动生成，请勿手改。',
            "",
            f"机器人: {name}    自由度: {robot.symbols.dof}",
            f"约定: {robot.frames.convention}",
            "",
            "所有函数的入参都是一维数组：q / dq / ddq 长度为 dof，",
            "parms 为 barycentric 动力学参数，顺序见下表。",
            '"""',
            "",
            "import numpy",
            "",
        ]
        head += parms_legend(robot)
        head.append("")
        body = [self.function(low) for low in lowered]
        return "\n".join(head) + "\n\n" + "\n\n\n".join(body) + "\n"


class CBackend(_Backend):
    comment = "//"

    def __init__(self):
        self.printer = C99CodePrinter()

    def function(self, low: Lowered) -> str:
        item = low.item
        rows, cols = item.shape
        params = ", ".join(f"const double *{a}" for a in low.args)
        out = [
            f"/* {item.doc}。out: {rows}x{cols}，行优先。 */",
            f"void {item.key}({params}, double *out)",
            "{",
        ]
        for lhs, rhs in low.temps:
            out.append(f"    const double {lhs} = {self._p(rhs)};")
        for r in range(rows):
            for c in range(cols):
                out.append(f"    out[{r * cols + c}] = {self._p(low.body[r, c])};")
        out.append("}")
        return "\n".join(out)

    def declaration(self, low: Lowered) -> str:
        params = ", ".join(f"const double *{a}" for a in low.args)
        return f"void {low.item.key}({params}, double *out);"

    def module(self, robot, lowered, name: str) -> str:
        lowered = list(lowered)
        head = [
            "/* 由 robot_model 自动生成，请勿手改。",
            " *",
            f" * 机器人: {name}    自由度: {robot.symbols.dof}",
            f" * 约定: {robot.frames.convention}",
            " */",
            "",
            "#include <math.h>",
            "",
        ]
        head += parms_legend(robot, prefix="//")
        head.append("")
        head += [self.declaration(low) for low in lowered]
        head.append("")
        body = [self.function(low) for low in lowered]
        return "\n".join(head) + "\n\n" + "\n\n".join(body) + "\n"


BACKENDS = {"python": PythonBackend, "c": CBackend}


def render_code(
    robot,
    items: Sequence[ExportItem],
    *,
    language: str = "python",
    cse: bool = True,
    name: str | None = None,
) -> str:
    """生成整个源文件文本。"""
    try:
        backend = BACKENDS[language]()
    except KeyError:
        raise ValueError(
            f"unknown language {language!r}; use {sorted(BACKENDS)}"
        ) from None
    lowered = [lower(robot, it, cse=cse) for it in items]
    return backend.module(robot, lowered, name or robot.symbols.name)
