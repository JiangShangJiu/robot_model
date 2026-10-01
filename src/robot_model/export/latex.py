"""LaTeX 导出：按表达式规模自动选排版。

一个 3R 的 tau 展开有几千个节点，塞进 bmatrix 会溢出页面，
所以这里按元素长度分三档：矩阵 / 逐元素 / 先列中间量再逐元素。
即便如此，单个元素仍可能过长，故再按顶层加法项折行——不依赖 breqn，
这样 pdflatex 和 xelatex 都能编。
"""

from __future__ import annotations

from typing import Sequence

import sympy

from .codegen import lower
from .collect import ExportItem, arg_symbols

# 元素的 LaTeX 长度阈值：小于第一档排成矩阵，大于第二档改用 cse
_MATRIX_LIMIT = 70
_CSE_LIMIT = 260
# 单行超过这个长度就按加法项折行
_WRAP_LIMIT = 90

_TEXT_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_text(s: str) -> str:
    """转义正文里的 LaTeX 特殊字符（机器人名常带下划线）。"""
    return "".join(_TEXT_ESCAPES.get(ch, ch) for ch in s)


def _preamble(cjk: bool) -> str:
    engine = "xelatex" if cjk else "pdflatex"
    lines = [
        f"% 用 {engine} 编译",
        "\\documentclass[11pt]{article}",
        "\\usepackage[margin=2cm]{geometry}",
        "\\usepackage{amsmath,amssymb}",
    ]
    if cjk:
        lines.append("\\usepackage{ctex}")
    lines += ["\\allowdisplaybreaks", "\\setlength{\\parindent}{0pt}", ""]
    return "\n".join(lines)


def _tex(expr) -> str:
    return sympy.latex(expr)


def _elements(expr: sympy.Matrix) -> list[tuple[str, sympy.Expr]]:
    """``[(下标标签, 元素)]``；列向量只标一个下标。"""
    rows, cols = expr.shape
    out = []
    for r in range(rows):
        for c in range(cols):
            label = f"{r + 1}" if cols == 1 else f"{r + 1}{c + 1}"
            out.append((label, expr[r, c]))
    return out


def _pick_style(expr: sympy.Matrix, style: str) -> str:
    if style != "auto":
        return style
    longest = max((len(_tex(e)) for _, e in _elements(expr)), default=0)
    if longest <= _MATRIX_LIMIT:
        return "matrix"
    if longest <= _CSE_LIMIT:
        return "elements"
    return "cse"


def _wrapped_equation(lhs: str, expr) -> str:
    """``lhs &= ...``，过长时按顶层加法项断成多行。"""
    whole = _tex(expr)
    terms = expr.as_ordered_terms() if isinstance(expr, sympy.Add) else []
    if len(whole) <= _WRAP_LIMIT or len(terms) < 2:
        return f"{lhs} &= {whole}"

    chunks: list[str] = []
    cur = ""
    for i, term in enumerate(terms):
        text = _tex(term)
        # sympy 已经把负号写进了项里，别再补一个 +
        piece = text if i == 0 or text.startswith("-") else f"+ {text}"
        if cur and len(cur) + len(piece) + 1 > _WRAP_LIMIT:
            chunks.append(cur)
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    chunks.append(cur)
    head = f"{lhs} &= {chunks[0]}"
    return " \\\\\n".join([head] + [f"&\\quad {c}" for c in chunks[1:]])


def _shorten(expr, extra: list, gen, limit: int):
    """递归缩短表达式：顶层和式留给折行，乘积里的长括号提成中间量。

    只影响排版，不改变数值。没有这一步，``x_1 (a+b+c+...)`` 这种
    单个巨型乘积项折不开，会直接冲出页面。
    """
    if len(_tex(expr)) <= limit:
        return expr
    if isinstance(expr, sympy.Add):
        return sympy.Add(
            *[_shorten(t, extra, gen, limit) for t in expr.as_ordered_terms()]
        )
    if expr.args:
        return expr.func(
            *[_shorten_factor(a, extra, gen, limit) for a in expr.args]
        )
    return expr


def _shorten_factor(expr, extra: list, gen, limit: int):
    if len(_tex(expr)) <= limit:
        return expr
    inner = _shorten(expr, extra, gen, limit)
    if isinstance(inner, sympy.Add):
        sym = next(gen)
        extra.append((sym, inner))
        return sym
    return inner


def _render_matrix(item: ExportItem) -> str:
    return (
        "\\begin{equation*}\n"
        f"{item.key} = {_tex(item.expr)}\n"
        "\\end{equation*}"
    )


def _render_elements(item: ExportItem, expr: sympy.Matrix | None = None) -> str:
    expr = item.expr if expr is None else expr
    body = [
        _wrapped_equation(f"{item.key}_{{{label}}}", e)
        for label, e in _elements(expr)
    ]
    return "\\begin{align*}\n" + " \\\\\n".join(body) + "\n\\end{align*}"


def _render_cse(robot, item: ExportItem) -> str:
    """先列公共子表达式，再用它们写出各元素——唯一能读的大公式排版。"""
    low = lower(robot, item, cse=True)
    # lower 把 q1 换成了 q[0]，对着看公式反而别扭，这里退回原符号
    back = {
        sympy.Symbol(f"{g}[{i}]"): s
        for g in item.uses
        for i, s in enumerate(arg_symbols(robot, g))
    }
    extra: list = []
    gen = sympy.numbered_symbols("y")
    body = low.body.subs(back).applyfunc(
        lambda e: _shorten(e, extra, gen, _WRAP_LIMIT)
    )
    defs = [(lhs, rhs.subs(back)) for lhs, rhs in low.temps] + extra
    temps = [_wrapped_equation(_tex(lhs), rhs) for lhs, rhs in defs]
    return "\n".join(
        [
            "\\textit{中间量：}",
            "\\begin{align*}",
            " \\\\\n".join(temps),
            "\\end{align*}",
            "\\textit{结果：}",
            _render_elements(item, body),
        ]
    )


def render_latex(
    robot,
    items: Sequence[ExportItem],
    *,
    standalone: bool = True,
    style: str = "auto",
    title: str | None = None,
    cjk: bool = False,
) -> str:
    """生成 LaTeX 文本。

    ``style`` 取 ``"auto"`` / ``"matrix"`` / ``"elements"`` / ``"cse"``，
    ``auto`` 按元素长度逐个量自行决定。

    ``cjk=False``（默认）生成 pdflatex 可编译的文档，中文说明降级为
    源码注释；``cjk=True`` 会加载 ctex，正文保留中文，但需要 xelatex。
    """
    chunks = []
    for item in items:
        chosen = _pick_style(item.expr, style)
        heading = f"${item.key}$"
        if cjk:
            chunks.append(f"\\section*{{{heading} —— {item.doc}}}")
        else:
            chunks.append(f"% {item.key} —— {item.doc}")
            chunks.append(f"\\section*{{{heading}}}")
        if chosen == "matrix":
            chunks.append(_render_matrix(item))
        elif chosen == "elements":
            chunks.append(_render_elements(item))
        elif chosen == "cse":
            chunks.append(_render_cse(robot, item))
        else:
            raise ValueError(f"unknown style {chosen!r}")
    body = "\n\n".join(chunks)

    if not standalone:
        return body + "\n"

    name = escape_text(title or robot.symbols.name)
    head = (
        _preamble(cjk)
        + f"\\title{{{name}}}\n\\date{{}}\n"
        + "\\begin{document}\n\\maketitle\n\n"
    )
    return head + body + "\n\n\\end{document}\n"


def render_inline(item: ExportItem) -> str:
    """单个量的行间公式，供 Jupyter ``Math`` 直接渲染。"""
    return f"{item.key} = {_tex(item.expr)}"
