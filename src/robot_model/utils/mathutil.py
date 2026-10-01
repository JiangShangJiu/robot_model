"""通用符号矩阵工具。"""

from __future__ import annotations

import sympy


def identity(x):
    return x


def skew(v: sympy.Matrix) -> sympy.Matrix:
    """三维向量的反对称矩阵，使 a×b = skew(a)·b。"""
    return sympy.Matrix(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def inverse_T(T: sympy.Matrix) -> sympy.Matrix:
    """刚体齐次变换逆：T=[R p; 0 1] → [Rᵀ -Rᵀp; 0 1]。"""
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    return R.T.row_join(-R.T * p).col_join(sympy.zeros(1, 3).row_join(sympy.eye(1)))


def elements_to_tensor(elems) -> sympy.Matrix:
    """[xx, xy, xz, yy, yz, zz] → 对称 3×3。"""
    return sympy.Matrix(
        [
            [elems[0], elems[1], elems[2]],
            [elems[1], elems[3], elems[4]],
            [elems[2], elems[4], elems[5]],
        ]
    )


def se3_hat(S: sympy.Matrix) -> sympy.Matrix:
    """运动旋量 ``S=(ω,v)`` → ``se(3)`` 4×4 帽子矩阵。"""
    S = sympy.Matrix(S).reshape(6, 1)
    w, v = S[0:3, 0], S[3:6, 0]
    return skew(w).row_join(v).col_join(sympy.zeros(1, 4))


def se3_exp(S: sympy.Matrix, theta, *, prismatic: bool = False) -> sympy.Matrix:
    """``exp([S] θ)``（Modern Robotics 空间指数；转动假定 ``‖ω‖=1``）。"""
    S = sympy.Matrix(S).reshape(6, 1)
    th = sympy.sympify(theta)
    if prismatic:
        v = S[3:6, 0]
        return (
            sympy.eye(3)
            .row_join(th * v)
            .col_join(sympy.Matrix([[0, 0, 0, 1]]))
        )

    w = S[0:3, 0]
    v = S[3:6, 0]
    wh = skew(w)
    R = (
        sympy.eye(3)
        + sympy.sin(th) * wh
        + (1 - sympy.cos(th)) * (wh * wh)
    )
    G = (
        th * sympy.eye(3)
        + (1 - sympy.cos(th)) * wh
        + (th - sympy.sin(th)) * (wh * wh)
    )
    p = G * v
    return R.row_join(p).col_join(sympy.Matrix([[0, 0, 0, 1]]))


def adjoint_twist(T: sympy.Matrix, S: sympy.Matrix) -> sympy.Matrix:
    """``Ad_T S``：把运动旋量变到由 ``T`` 变换后的坐标系（空间伴随）。"""
    T = sympy.Matrix(T)
    S = sympy.Matrix(S).reshape(6, 1)
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    w = S[0:3, 0]
    v = S[3:6, 0]
    w2 = R * w
    v2 = skew(p) * w2 + R * v
    return w2.col_join(v2)


def rpy_zyx_from_R(R: sympy.Matrix) -> sympy.Matrix:
    """从旋转矩阵提取 ZYX 欧拉角 ``φ=[roll, pitch, yaw]``（``R=Rz(yaw)Ry(pitch)Rx(roll)``）。"""
    R = sympy.Matrix(R)
    pitch = sympy.atan2(
        -R[2, 0], sympy.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    )
    roll = sympy.atan2(R[2, 1], R[2, 2])
    yaw = sympy.atan2(R[1, 0], R[0, 0])
    return sympy.Matrix([roll, pitch, yaw])


def rpy_zyx_E(phi) -> sympy.Matrix:
    """``ω = E(φ) φ̇``，``φ=[roll,pitch,yaw]``，对应 ``R=Rz(yaw)Ry(pitch)Rx(roll)``。"""
    p, y = phi[1], phi[2]
    cp, sp = sympy.cos(p), sympy.sin(p)
    cy, sy = sympy.cos(y), sympy.sin(y)
    return sympy.Matrix(
        [
            [cp * cy, -sy, 0],
            [cp * sy, cy, 0],
            [-sp, 0, 1],
        ]
    )


def analytical_from_geometric(Jg: sympy.Matrix, phi) -> sympy.Matrix:
    """由几何雅可比经 ``ω=Eφ̇`` 得到解析雅可比（对照用，非直接求导）。"""
    E = rpy_zyx_E(phi)
    Einv = E.inv()  # 勿对工业臂符号姿态 simplify，极慢
    Jp = Jg[0:3, :]
    Jo = Jg[3:6, :]
    return Jp.col_join(Einv * Jo)
