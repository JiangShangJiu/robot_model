"""动力学最小参数集（基参数）提取：由回归矩阵列依赖做 QR 分解。"""

from __future__ import annotations

from typing import Callable

import numpy as np
import sympy

from ..utils.validation import integer


def _spans_columns(matrix, indices, tolerance):
    """在小规模 QR 因子上同时检查列独立性和重构误差。"""
    if not indices:
        return bool(np.max(np.linalg.norm(matrix, axis=0)) <= tolerance)
    basis = matrix[:, indices]
    singular_values = np.linalg.svd(basis, compute_uv=False)
    if singular_values[-1] <= tolerance:
        return False
    coefficients = np.linalg.lstsq(basis, matrix, rcond=None)[0]
    residual = matrix - basis @ coefficients
    return bool(np.max(np.linalg.norm(residual, axis=0)) <= tolerance)


def _pivoted_columns(matrix, tolerance):
    """纯 NumPy 的列主元选择，对依赖列做重正交化。

    只有历史基参数选择通不过重构检查时才会走这里；
    常规机器人的基参数因此能保持原有参数顺序。
    """
    residual = matrix.copy()
    orthogonal = np.empty((matrix.shape[0], 0))
    selected = []
    for _ in range(min(matrix.shape)):
        pivot = int(np.argmax(np.linalg.norm(residual, axis=0)))
        direction = residual[:, pivot].copy()
        for _ in range(2):
            direction -= orthogonal @ (orthogonal.T @ direction)
        norm = np.linalg.norm(direction)
        if norm <= tolerance:
            break
        direction /= norm
        orthogonal = np.column_stack((orthogonal, direction))
        for _ in range(2):
            residual -= np.outer(direction, direction @ residual)
        residual[:, pivot] = 0.0
        selected.append(pivot)
    return sorted(selected)


def find_dyn_parm_deps(
    dof: int,
    parm_num: int,
    regressor_func: Callable,
    *,
    samples: int = 2000,
    round_digits: int = 10,
    seed: int = 0,
):
    """由数值回归矩阵采样找列依赖。

    返回 ``(Pb, Pd, Kd)``，使基参数
    ``π_b = (Pbᵀ + Kd Pdᵀ) π``，且 ``τ = H π = (H Pb) π_b``。
    """
    dof = integer(dof, "dof", 1)
    parm_num = integer(parm_num, "parm_num", 1)
    samples = integer(samples, "samples", 1)
    round_digits = integer(round_digits, "round_digits")
    seed = integer(seed, "seed")
    rng = np.random.default_rng(seed)
    Z = np.zeros((dof * samples, parm_num))
    for i in range(samples):
        q = rng.uniform(-np.pi, np.pi, size=dof)
        dq = rng.uniform(-np.pi, np.pi, size=dof)
        ddq = rng.uniform(-np.pi, np.pi, size=dof)
        Hi = np.asarray(regressor_func(q, dq, ddq), dtype=float)
        if Hi.shape != (dof, parm_num) or not np.isfinite(Hi).all():
            raise ValueError(
                f"regressor_func 必须返回有限的 ({dof}, {parm_num}) 矩阵"
            )
        Z[i * dof : (i + 1) * dof, :] = Hi

    # 保留历史基参数列顺序，但不能仅凭无列主元 QR 的对角线判秩：
    # 早先的零列/依赖列可能占据后面独立列的正交方向。
    R = np.linalg.qr(Z, mode="r")
    diag = np.round(np.abs(np.diag(R)), round_digits)
    dbi = [i for i, e in enumerate(diag) if e != 0]
    scale = float(np.max(np.linalg.norm(R, axis=0)))
    tolerance = max(0.5 * 10.0 ** (-round_digits),
                    np.finfo(float).eps * max(Z.shape) * scale)
    if not _spans_columns(R, dbi, tolerance):
        dbi = _pivoted_columns(R, tolerance)
    selected = set(dbi)
    ddi = [i for i in range(parm_num) if i not in selected]
    dbn = len(dbi)

    P = np.eye(parm_num)[:, dbi + ddi]
    Pb = P[:, :dbn]
    Pd = P[:, dbn:]

    Rbd = np.linalg.qr(Z @ P, mode="r")
    Rb = Rbd[:dbn, :dbn]
    Rd = Rbd[:dbn, dbn:]
    Kd = np.round(np.linalg.solve(Rb, Rd), round_digits)
    return Pb, Pd, Kd


def calc_base_parms(
    dynparms,
    dof: int,
    regressor_func: Callable,
    *,
    samples: int = 2000,
    round_digits: int = 10,
    seed: int = 0,
) -> dict:
    """计算基参数符号与投影。

    返回
    -------
    dict
        ``Pb, Pd, Kd, base_idxs, baseparms, n_base``
    """
    dynparms = sympy.Matrix(dynparms)
    n = len(dynparms)
    Pb, Pd, Kd = find_dyn_parm_deps(
        dof,
        n,
        regressor_func,
        samples=samples,
        round_digits=round_digits,
        seed=seed,
    )
    Pb_s = sympy.Matrix(Pb).applyfunc(lambda x: sympy.nsimplify(x))
    Pd_s = sympy.Matrix(Pd).applyfunc(lambda x: sympy.nsimplify(x))
    Kd_s = sympy.Matrix(Kd).applyfunc(lambda x: sympy.nsimplify(x))
    base_idxs = [
        int(i)
        for i in (np.arange(n) @ np.asarray(Pb, dtype=float)).tolist()
    ]
    baseparms = (Pb_s.T + Kd_s * Pd_s.T) * dynparms
    return {
        "Pb": Pb_s,
        "Pd": Pd_s,
        "Kd": Kd_s,
        "base_idxs": base_idxs,
        "baseparms": baseparms,
        "n_base": len(baseparms),
    }


def make_regressor_func(symbols, H: sympy.Matrix) -> Callable:
    """把符号 ``H(q,dq,ddq)`` 变成 ``func(q,dq,ddq)->ndarray``。"""
    args = list(symbols.q) + list(symbols.dq) + list(symbols.ddq)
    fun = sympy.lambdify(args, H, "numpy")

    def regressor_func(q, dq, ddq):
        vals = list(np.asarray(q, float).reshape(-1))
        vals += list(np.asarray(dq, float).reshape(-1))
        vals += list(np.asarray(ddq, float).reshape(-1))
        return np.asarray(fun(*vals), dtype=float)

    return regressor_func
