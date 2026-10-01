"""收集待导出的符号量，统一成后端无关的 ``ExportItem``。

LaTeX / Python / C 三个后端都消费这里的结果，所以"导出哪些量、
每个量依赖哪些符号、要不要化简"只在这里决定一次。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import sympy

# 符号组：生成的函数按这个顺序取形参
ARG_GROUPS = ("q", "dq", "ddq", "parms")

# 默认导出集：常用且展开后规模可控（H 有几千项，需显式要求）
DEFAULT_KEYS: tuple[str, ...] = ("T", "J", "M", "c", "g", "tau")

# 超过这个节点数就不自动化简，否则 trigsimp 会卡很久
_AUTO_SIMPLIFY_LIMIT = 400

_PER_LINK = re.compile(r"^(T|T_rel|J)_(\d+)$")


@dataclass(frozen=True)
class ExportItem:
    """一个待导出的矩阵量。"""

    key: str  # 生成的函数 / 公式名
    expr: sympy.Matrix
    doc: str  # 中文说明，写进注释和 LaTeX 标题
    uses: tuple[str, ...]  # 依赖的符号组，见 ARG_GROUPS

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.expr.shape)


def count_ops(expr) -> int:
    """表达式树节点数，用来判断是否值得化简。"""
    return sum(1 for _ in sympy.preorder_traversal(sympy.Matrix(expr)))


def _simplified(expr: sympy.Matrix, mode: bool | str) -> sympy.Matrix:
    if mode is False:
        return expr
    if mode == "auto" and count_ops(expr) > _AUTO_SIMPLIFY_LIMIT:
        return expr
    return sympy.Matrix(expr).applyfunc(sympy.trigsimp)


def _dyn(robot):
    return robot.dynamics


def _base_regressor(robot) -> sympy.Matrix:
    dyn = _dyn(robot)
    if dyn.H is None:
        dyn.gen_regressor()
    if dyn.Pb is None:
        dyn.calc_base_parms()
    Hb = dyn.gen_base_regressor()
    if Hb is None:
        raise RuntimeError("符号 Hb 不可用：需要先有符号 H（gen_regressor）")
    return sympy.Matrix(Hb)


def _baseparms(robot) -> sympy.Matrix:
    dyn = _dyn(robot)
    if dyn.Pb is None:
        dyn.calc_base_parms()
    return sympy.Matrix(dyn.baseparms)


# key -> (取表达式, 说明, 依赖的符号组)
_BUILDERS: dict[str, tuple[Callable, str, tuple[str, ...]]] = {
    "T": (lambda r: r.frames.T[-1], "末端位姿（齐次变换）", ("q",)),
    "J": (
        lambda r: r.kinematics.J[-1],
        "末端几何雅可比 [v; omega] = J dq",
        ("q",),
    ),
    "M": (lambda r: _dyn(r).gen_inertiamatrix(), "惯性矩阵", ("q", "parms")),
    "c": (
        lambda r: _dyn(r).gen_coriolisterm(),
        "科氏 / 离心项",
        ("q", "dq", "parms"),
    ),
    "C": (
        lambda r: _dyn(r).gen_coriolismatrix(),
        "科氏矩阵（Christoffel，c = C dq）",
        ("q", "dq", "parms"),
    ),
    "g": (lambda r: _dyn(r).gen_gravityterm(), "重力项", ("q", "parms")),
    "tau": (
        lambda r: _dyn(r).gen_invdyn(),
        "逆动力学力矩 tau = M ddq + c + g",
        ("q", "dq", "ddq", "parms"),
    ),
    "H": (
        lambda r: _dyn(r).gen_regressor(),
        "回归矩阵 tau = H pi",
        ("q", "dq", "ddq"),
    ),
    "Hb": (_base_regressor, "最小参数回归矩阵 tau = Hb pi_b", ("q", "dq", "ddq")),
    "baseparms": (_baseparms, "最小参数集 pi_b（用原参数表示）", ("parms",)),
}

AVAILABLE_KEYS: tuple[str, ...] = tuple(_BUILDERS)


def _build_per_link(robot, key: str):
    """``T_2`` / ``J_3`` 这类逐连杆的量（下标 1-based）。"""
    m = _PER_LINK.match(key)
    if m is None:
        return None
    kind, idx = m.group(1), int(m.group(2))
    if not 1 <= idx <= robot.symbols.dof:
        raise ValueError(f"{key}: 连杆下标越界（dof={robot.symbols.dof}）")
    i = idx - 1
    if kind == "T":
        return robot.frames.T[i], f"连杆 {idx} 位姿（相对基座）", ("q",)
    if kind == "T_rel":
        return robot.frames.T_rel[i], f"连杆 {idx} 相对上一连杆的位姿", ("q",)
    return robot.kinematics.J[i], f"连杆 {idx} 几何雅可比", ("q",)


def collect(
    robot,
    keys: Iterable[str] | None = None,
    *,
    simplify: bool | str = "auto",
) -> list[ExportItem]:
    """按 ``keys`` 取出符号量。

    除 :data:`AVAILABLE_KEYS` 外，还支持逐连杆的 ``T_i`` / ``T_rel_i`` / ``J_i``
    （下标 1-based，如 ``"T_2"``）。

    ``simplify`` 为 ``"auto"`` 时只化简小表达式（节点数 <=
    ``_AUTO_SIMPLIFY_LIMIT``），避免在 tau / H 上空耗。
    """
    keys = tuple(DEFAULT_KEYS if keys is None else keys)
    items: list[ExportItem] = []
    for key in keys:
        builder = _BUILDERS.get(key)
        if builder is not None:
            expr, doc, uses = builder[0](robot), builder[1], builder[2]
        else:
            per_link = _build_per_link(robot, key)
            if per_link is None:
                raise KeyError(
                    f"未知导出项 {key!r}；可用：{', '.join(AVAILABLE_KEYS)}"
                    "，或 T_i / T_rel_i / J_i"
                )
            expr, doc, uses = per_link
        items.append(
            ExportItem(
                key=key,
                expr=_simplified(sympy.Matrix(expr), simplify),
                doc=doc,
                uses=uses,
            )
        )
    return items


def arg_symbols(robot, group: str) -> Sequence[sympy.Symbol]:
    """某个符号组对应的符号序列，顺序即生成代码里的下标顺序。"""
    s = robot.symbols
    if group == "q":
        return list(s.q)
    if group == "dq":
        return list(s.dq)
    if group == "ddq":
        return list(s.ddq)
    if group == "parms":
        return list(s.dynparms())
    raise ValueError(f"unknown arg group {group!r}")
