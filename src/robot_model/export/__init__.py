"""符号结果导出：LaTeX（看公式）、Python / C（拿去算）。

入口是 ``robot.export``::

    robot.export.show("M")                 # notebook 里渲染
    robot.export.latex("rrr.tex")          # 可编译的 .tex
    robot.export.python("rrr_model.py")    # 带 cse 的 numpy 代码
    robot.export.c("rrr_model.c")
"""

from __future__ import annotations

from .codegen import render_code
from .collect import AVAILABLE_KEYS, DEFAULT_KEYS, ExportItem, collect
from .exporter import Exporter
from .latex import render_inline, render_latex

__all__ = [
    "AVAILABLE_KEYS",
    "DEFAULT_KEYS",
    "ExportItem",
    "Exporter",
    "collect",
    "render_code",
    "render_inline",
    "render_latex",
]
