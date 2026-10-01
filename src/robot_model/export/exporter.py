"""导出门面：``robot.export`` 就是这个类的实例。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .codegen import render_code
from .collect import AVAILABLE_KEYS, DEFAULT_KEYS, ExportItem, collect
from .latex import render_inline, render_latex


class Exporter:
    """把 ``Robot`` 的符号结果导出成 LaTeX / Python / C。

    每个方法的 ``path`` 都可省略：给了就写文件并返回 ``Path``，
    不给就把生成的文本直接返回，方便在 notebook 里先看再存。
    """

    #: 不指定 ``keys`` 时导出的量
    default_keys = DEFAULT_KEYS
    #: 支持的量（另可用 ``T_i`` / ``T_rel_i`` / ``J_i``）
    available_keys = AVAILABLE_KEYS

    def __init__(self, robot):
        self.robot = robot

    def items(
        self,
        keys: Iterable[str] | None = None,
        *,
        simplify: bool | str = "auto",
    ) -> list[ExportItem]:
        """取出符号量本身，不做渲染。"""
        return collect(self.robot, keys, simplify=simplify)

    def latex(
        self,
        path: str | Path | None = None,
        *,
        keys: Iterable[str] | None = None,
        simplify: bool | str = "auto",
        style: str = "auto",
        standalone: bool = True,
        title: str | None = None,
        cjk: bool = False,
    ):
        """``cjk=True`` 保留中文标题但需 xelatex + ctex；默认走 pdflatex。"""
        text = render_latex(
            self.robot,
            self.items(keys, simplify=simplify),
            standalone=standalone,
            style=style,
            title=title,
            cjk=cjk,
        )
        return self._emit(text, path)

    def python(
        self,
        path: str | Path | None = None,
        *,
        keys: Iterable[str] | None = None,
        simplify: bool | str = False,
        cse: bool = True,
    ):
        """生成可直接 import 的 ``.py``（默认不化简，只做 cse）。"""
        text = render_code(
            self.robot,
            self.items(keys, simplify=simplify),
            language="python",
            cse=cse,
        )
        return self._emit(text, path)

    def c(
        self,
        path: str | Path | None = None,
        *,
        keys: Iterable[str] | None = None,
        simplify: bool | str = False,
        cse: bool = True,
    ):
        text = render_code(
            self.robot,
            self.items(keys, simplify=simplify),
            language="c",
            cse=cse,
        )
        return self._emit(text, path)

    def show(
        self,
        keys: Iterable[str] | None = None,
        *,
        simplify: bool | str = "auto",
    ) -> None:
        """在 Jupyter 里渲染公式；非 notebook 环境退化为打印。"""
        items = self.items(keys, simplify=simplify)
        try:
            from IPython.display import Math, display
        except ImportError:
            for it in items:
                print(f"{it.key} —— {it.doc}\n{it.expr}\n")
            return
        for it in items:
            print(f"{it.key} —— {it.doc}")
            display(Math(render_inline(it)))

    @staticmethod
    def _emit(text: str, path: str | Path | None):
        if path is None:
            return text
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p
