"""SE(3) 伴随 / 余伴随算子（符号与数值各一套，公式同源）。"""

from __future__ import annotations

import numpy as np
import sympy

from .mathutil import skew


# ----- 符号计算（SymPy） -----


def Adj(G: sympy.Matrix, g: sympy.Matrix) -> sympy.Matrix:
    """``Ad_G g``：刚体变换对旋量的伴随作用。"""
    R = G[0:3, 0:3]
    p = G[0:3, 3]
    return (R.row_join(sympy.zeros(3))).col_join((skew(p) * R).row_join(R)) * g


def Adjdual(G: sympy.Matrix, g: sympy.Matrix) -> sympy.Matrix:
    """``Ad_G^* g``：余伴随（力旋量）。"""
    R = G[0:3, 0:3]
    p = G[0:3, 3]
    return ((R.row_join(sympy.zeros(3))).col_join((skew(p) * R).row_join(R))).T * g


def adj(g: sympy.Matrix, h: sympy.Matrix) -> sympy.Matrix:
    """``ad_g h``：se(3) 李括号伴随。"""
    wg, vg = g[0:3, 0], g[3:6, 0]
    return (skew(wg).row_join(sympy.zeros(3))).col_join(
        (skew(vg)).row_join(skew(wg))
    ) * h


def adjdual(g: sympy.Matrix, h: sympy.Matrix) -> sympy.Matrix:
    """``ad_g^* h``：余伴随。"""
    wg, vg = g[0:3, 0], g[3:6, 0]
    return (
        (skew(wg).row_join(sympy.zeros(3))).col_join((skew(vg)).row_join(skew(wg)))
    ).T * h


# ----- 数值计算（NumPy） -----


def skew_np(v) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def Adj_np(T, g) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = skew_np(p) @ R
    Ad[3:, 3:] = R
    return Ad @ np.asarray(g, dtype=float).reshape(6)


def Adjdual_np(T, g) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = skew_np(p) @ R
    Ad[3:, 3:] = R
    return Ad.T @ np.asarray(g, dtype=float).reshape(6)


def adj_np(g, h) -> np.ndarray:
    g = np.asarray(g, dtype=float).reshape(6)
    wg, vg = g[:3], g[3:]
    mat = np.zeros((6, 6))
    mat[:3, :3] = skew_np(wg)
    mat[3:, :3] = skew_np(vg)
    mat[3:, 3:] = skew_np(wg)
    return mat @ np.asarray(h, dtype=float).reshape(6)


def adjdual_np(g, h) -> np.ndarray:
    g = np.asarray(g, dtype=float).reshape(6)
    wg, vg = g[:3], g[3:]
    mat = np.zeros((6, 6))
    mat[:3, :3] = skew_np(wg)
    mat[3:, :3] = skew_np(vg)
    mat[3:, 3:] = skew_np(wg)
    return mat.T @ np.asarray(h, dtype=float).reshape(6)
