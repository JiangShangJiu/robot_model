"""由 robot_model 自动生成，请勿手改。

机器人: rrr_arm    自由度: 3
约定: modified

所有函数的入参都是一维数组：q / dq / ddq 长度为 dof，
parms 为 barycentric 动力学参数，顺序见下表。
"""

import numpy

# parms 顺序（共 30 项，Khalil 排列）:
#   * [ 0] L_1xx
#     [ 1] L_1xy
#     [ 2] L_1xz
#     [ 3] L_1yy
#     [ 4] L_1yz
#     [ 5] L_1zz
#     [ 6] l_1x
#     [ 7] l_1y
#     [ 8] l_1z
#     [ 9] m_1
#   * [10] L_2xx
#     [11] L_2xy
#     [12] L_2xz
#     [13] L_2yy
#     [14] L_2yz
#     [15] L_2zz
#     [16] l_2x
#     [17] l_2y
#     [18] l_2z
#     [19] m_2
#   * [20] L_3xx
#     [21] L_3xy
#     [22] L_3xz
#     [23] L_3yy
#     [24] L_3yz
#     [25] L_3zz
#     [26] l_3x
#     [27] l_3y
#     [28] l_3z
#     [29] m_3


def T(q):
    """末端位姿（齐次变换）。

    返回形状 4x4。
    """
    x0 = numpy.cos(q[0])
    x1 = numpy.sin(q[1])
    x2 = numpy.sin(q[2])
    x3 = x1*x2
    x4 = numpy.cos(q[1])
    x5 = numpy.cos(q[2])
    x6 = x1*x5
    x7 = x2*x4
    x8 = numpy.sin(q[0])
    return numpy.array([
        [-x0*x3 + x0*x4*x5, -x0*x6 - x0*x7, -x8, 0.28*x0*x4 - 0.05*x8],
        [-x3*x8 + x4*x5*x8, -x6*x8 - x7*x8, x0, 0.05*x0 + 0.28*x4*x8],
        [-x6 - x7, x3 - x4*x5, 0, 0.3 - 0.28*x1],
        [0, 0, 0, 1],
    ])


def J(q):
    """末端几何雅可比 [v; omega] = J dq。

    返回形状 6x3。
    """
    x0 = numpy.cos(q[0])
    x1 = numpy.sin(q[0])
    x2 = numpy.cos(q[1])
    x3 = 0.05*x0 + 0.28*x1*x2
    x4 = 0.28*numpy.sin(q[1])
    x5 = 0.28*x0*x2 - 0.05*x1
    x6 = -x1
    return numpy.array([
        [-x3, -x0*x4, 0],
        [x5, -x1*x4, 0],
        [0, -x0*x5 - x1*x3, 0],
        [0, x6, x6],
        [0, x0, x0],
        [1, 0, 0],
    ])


def M(q, parms):
    """惯性矩阵。

    返回形状 3x3。
    """
    x0 = numpy.sin(q[1])
    x1 = numpy.cos(q[1])
    x2 = numpy.cos(q[2])
    x3 = x0*x2
    x4 = numpy.sin(q[2])
    x5 = x1*x4
    x6 = x3 + x5
    x7 = 0.05*x3 + 0.05*x5
    x8 = parms[28]*x6 + parms[29]*x7
    x9 = x0*x4
    x10 = x1*x2
    x11 = -x10 + x9
    x12 = -0.05*x10 + 0.05*x9
    x13 = parms[28]*x11 + parms[29]*x12
    x14 = 0.05*x4
    x15 = 0.28*x1
    x16 = -x6
    x17 = parms[21]*x16 + parms[23]*x11 - parms[26]*x15 + parms[28]*x12
    x18 = parms[20]*x16 + parms[21]*x11 + parms[27]*x15 - parms[28]*x7
    x19 = 0.28*parms[26]
    x20 = 0.28*parms[27]
    x21 = 0.28*x2
    x22 = 0.28*x4
    x23 = parms[22]*x16 + parms[24]*x11 + parms[26]*x7 - parms[27]*x12
    x24 = -parms[12]*x0 - parms[14]*x1 + x13*x22 + x21*x8 + x23
    x25 = parms[25] + x19*x2 - x20*x4
    return numpy.array([
        [parms[5] - x0*(-parms[10]*x0 - parms[11]*x1 - x13*x14 - x17*x4 + x18*x2 - 0.05*x2*x8) - x1*(-parms[11]*x0 - parms[13]*x1 - 0.0784*parms[29]*x1 + x11*x19 + 0.05*x13*x2 - x14*x8 - x16*x20 + x17*x2 + x18*x4), x24, x23],
        [x24, parms[15] + x21*(parms[26] + parms[29]*x21) + x22*(-parms[27] + 0.28*parms[29]*x4) + x25, x25],
        [x23, x25, parms[25]],
    ])


def c(q, dq, parms):
    """科氏 / 离心项。

    返回形状 3x1。
    """
    x0 = numpy.sin(q[1])
    x1 = numpy.cos(q[1])
    x2 = dq[0]*x1
    x3 = dq[0]*x0
    x4 = dq[1]*parms[14] - parms[11]*x3 - parms[13]*x2
    x5 = dq[1]*parms[15] - parms[12]*x3 - parms[14]*x2
    x6 = numpy.cos(q[2])
    x7 = 0.28*x6
    x8 = dq[0]**2
    x9 = x0*x1*x8
    x10 = numpy.sin(q[2])
    x11 = dq[1]**2
    x12 = x1**2
    x13 = x12*x8
    x14 = -0.28*x11 - 0.28*x13
    x15 = -x10*x14 + x7*x9
    x16 = dq[0]*x6
    x17 = x1*x16
    x18 = dq[2]*x17
    x19 = dq[0]*x10
    x20 = x0*x19
    x21 = -x17 + x20
    x22 = dq[2]*x21
    x23 = x10**2
    x24 = x0**2
    x25 = x24*x8
    x26 = x10*x6
    x27 = x6**2
    x28 = x0*x1*x27*x8 + x10*x12*x6*x8 - x23*x9 - x25*x26
    x29 = 2*x26*x9
    x30 = x13*x23 + x25*x27
    x31 = 2*dq[1]
    x32 = dq[2]**2 + dq[2]*x31 + x11
    x33 = parms[26]*x28 + parms[27]*(-x29 - x30 - x32) + parms[28]*(dq[0]*dq[2]*x0*x10 - x18 - x22) + parms[29]*x15
    x34 = 0.28*x10
    x35 = x14*x6 + x34*x9
    x36 = x0*x16
    x37 = x1*x19
    x38 = -x36 - x37
    x39 = dq[2]*x38
    x40 = dq[2]*x36 + dq[2]*x37
    x41 = x13*x27 + x23*x25
    x42 = parms[26]*(x29 - x32 - x41) + parms[27]*x28 + parms[28]*(-x39 - x40) + parms[29]*x35
    x43 = 0.05*x10
    x44 = 0.05*x8
    x45 = -0.56*dq[1]*x3 - x12*x44 - x24*x44
    x46 = -dq[1]*x17 + dq[1]*x20 + x22
    x47 = -x39
    x48 = dq[1]*x36 + dq[1]*x37 + x47
    x49 = dq[1] + dq[2]
    x50 = parms[20]*x38 + parms[21]*x21 + parms[22]*x49
    x51 = parms[22]*x38 + parms[24]*x21 + parms[25]*x49
    x52 = parms[21]*x46 + parms[23]*x48 - parms[26]*x45 + parms[28]*x35 - x38*x51 + x49*x50
    x53 = parms[21]*x38 + parms[23]*x21 + parms[24]*x49
    x54 = parms[20]*x46 + parms[21]*x48 + parms[27]*x45 - parms[28]*x15 + x21*x51 - x49*x53
    x55 = parms[11]*x2
    x56 = dq[1]*parms[12] - parms[10]*x3 - x55
    x57 = parms[22]*x46 + parms[24]*x48 + parms[26]*x15 - parms[27]*x35 - x21*x50 + x38*x53
    c_1 = -x0*(dq[0]*dq[1]*parms[11]*x0 - dq[1]*parms[10]*x2 - dq[1]*x4 - x10*x52 - x2*x5 - 0.05*x33*x6 - x42*x43 + x54*x6) - x1*(dq[0]*dq[1]*parms[13]*x0 + dq[0]*x0*x5 - dq[1]*x55 + dq[1]*x56 - 0.28*parms[26]*(-x31*x36 - x31*x37 - x40 - x47) - 0.28*parms[27]*(dq[2]*x20 - x17*x31 - x18 + x20*x31 + x22) - 0.28*parms[28]*(-x30 - x41) - 0.28*parms[29]*x45 + x10*x54 - x33*x43 + 0.05*x42*x6 + x52*x6)
    c_2 = -dq[1]*parms[12]*x2 + dq[1]*parms[14]*x3 + x2*x56 - x3*x4 + x33*x7 + x34*x42 + x57
    c_3 = x57
    return numpy.array([c_1, c_2, c_3])


def C(q, dq, parms):
    """科氏矩阵（Christoffel，c = C dq）。

    返回形状 3x3。
    """
    x0 = numpy.sin(q[1])
    x1 = numpy.sin(q[2])
    x2 = x0*x1
    x3 = numpy.cos(q[1])
    x4 = numpy.cos(q[2])
    x5 = x3*x4
    x6 = x2 - x5
    x7 = -x6
    x8 = 0.05*x2 - 0.05*x5
    x9 = -x8
    x10 = parms[28]*x7 + parms[29]*x9
    x11 = 0.05*x4
    x12 = x10*x11
    x13 = parms[28]*x6 + parms[29]*x8
    x14 = x11*x13
    x15 = x0*x4
    x16 = x1*x3
    x17 = x15 + x16
    x18 = parms[21]*x6
    x19 = 0.05*x15 + 0.05*x16
    x20 = parms[28]*x19
    x21 = parms[23]*x17 + x18 + x20
    x22 = parms[20]*x6 + parms[21]*x17 - parms[28]*x9
    x23 = 0.28*x3
    x24 = -x17
    x25 = parms[21]*x24 + parms[23]*x6 - parms[26]*x23 + parms[28]*x8
    x26 = x25*x4
    x27 = parms[20]*x24 + parms[27]*x23 + x18 - x20
    x28 = x1*x27
    x29 = (1/2)*x0
    x30 = 0.28*parms[26]
    x31 = x24*x30
    x32 = 0.28*parms[27]
    x33 = x32*x6
    x34 = 0.05*x1
    x35 = x10*x34
    x36 = x1*x25 + x13*x34 - x27*x4
    x37 = (1/2)*x3
    x38 = x29*(-x1*x21 - x12 - x14 + x22*x4 - x26 - x28) + x37*(x1*x22 + x21*x4 - x31 - x33 - x35 - x36)
    x39 = -x38
    x40 = parms[11]*x0
    x41 = parms[28]*x17 + parms[29]*x19
    x42 = x34*x41
    x43 = x0*x30 + x21
    x44 = -x0*x32 + x22
    x45 = parms[11]*x3
    x46 = x11*x41
    x47 = 0.0784*parms[29]
    x48 = -1/2*x0*(-parms[13]*x3 + x14 - x24*x32 + x26 + x28 - x3*x47 - x30*x7 - x40 - x42) + x29*(-parms[10]*x3 - x1*x43 - x12 + x4*x44 + x40 - x42) + x37*(-parms[10]*x0 - x36 - x45 - x46) + x37*(parms[13]*x0 + x0*x47 + x1*x44 - x31 - x33 - x35 + x4*x43 - x45 + x46)
    x49 = -x48
    x50 = x10*x4
    x51 = 0.14*x4
    x52 = x13*x51 + 0.14*x50
    x53 = parms[22]*x6 + parms[24]*x17 + parms[26]*x9 - parms[27]*x19
    x54 = x52 + x53
    x55 = 0.14*x1
    x56 = parms[26]*x55 + parms[27]*x51 - 0.14*x4*(-parms[27] + 0.28*parms[29]*x1) + x55*(parms[26] + 0.28*parms[29]*x4)
    x57 = -x56
    x58 = -x52
    return numpy.array([
        [dq[1]*x49 + dq[2]*x39, dq[0]*x49 + dq[1]*(-parms[12]*x3 + parms[14]*x0 + 0.28*x1*x41 + 0.28*x50 + x53) + dq[2]*x54, dq[0]*x39 + dq[1]*x54 + dq[2]*x53],
        [dq[0]*x48 + dq[2]*x52, dq[2]*x57, dq[0]*x52 + dq[1]*x57 + dq[2]*(-x1*x30 - x32*x4)],
        [dq[0]*x38 + dq[1]*x58, dq[0]*x58 + dq[1]*x56, 0],
    ])


def g(q, parms):
    """重力项。

    返回形状 3x1。
    """
    x0 = numpy.cos(q[1])
    x1 = numpy.sin(q[1])
    x2 = 9.81*x1
    x3 = numpy.sin(q[2])
    x4 = numpy.cos(q[2])
    x5 = 9.81*x0
    x6 = x2*x3 - x4*x5
    x7 = parms[28]*x3
    x8 = 0.05*parms[29]
    x9 = x3*x8
    x10 = -x2*x4 - x3*x5
    x11 = 0.28*parms[29]
    x12 = parms[26]*x6 - parms[27]*x10
    g_1 = -x0*(-parms[18]*x2 + parms[28]*x10*x4 + 0.05*parms[29]*x10*x4 - x6*x7 - x6*x9) - x1*(9.81*parms[18]*x0 - parms[28]*x4*x6 - x10*x7 - x10*x9 - x4*x6*x8)
    g_2 = -parms[16]*x5 + parms[17]*x2 + x10*x11*x3 + x11*x4*x6 + x12
    g_3 = x12
    return numpy.array([g_1, g_2, g_3])


def tau(q, dq, ddq, parms):
    """逆动力学力矩 tau = M ddq + c + g。

    返回形状 3x1。
    """
    x0 = numpy.sin(q[1])
    x1 = numpy.cos(q[1])
    x2 = 9.81*x1
    x3 = ddq[0]*x1
    x4 = dq[0]*dq[1]*x0 - x3
    x5 = ddq[0]*x0
    x6 = dq[0]*dq[1]
    x7 = x1*x6
    x8 = -x5 - x7
    x9 = dq[0]*x0
    x10 = dq[0]*x1
    x11 = dq[1]*parms[14] - parms[11]*x9 - parms[13]*x10
    x12 = dq[1]*parms[15] - parms[12]*x9 - parms[14]*x10
    x13 = numpy.cos(q[2])
    x14 = dq[0]**2
    x15 = x0*x1*x14
    x16 = 0.28*ddq[1] + 0.28*x15 - x2 + 0.05*x5
    x17 = numpy.sin(q[2])
    x18 = 9.81*x0
    x19 = dq[1]**2
    x20 = x1**2
    x21 = x14*x20
    x22 = -x18 - 0.28*x19 - 0.28*x21 - 0.05*x3
    x23 = x13*x16 - x17*x22
    x24 = x13*x17
    x25 = x13**2
    x26 = x0**2
    x27 = x14*x26
    x28 = x24*x27
    x29 = x17**2
    x30 = x15*x29
    x31 = ddq[1] + ddq[2]
    x32 = 2*x15*x24
    x33 = x21*x29 + x25*x27
    x34 = 2*dq[1]*dq[2] + dq[2]**2 + x19
    x35 = x13*x7
    x36 = dq[0]*x13
    x37 = x1*x36
    x38 = dq[2]*x37
    x39 = dq[0]*x17
    x40 = x0*x39
    x41 = -x37 + x40
    x42 = dq[2]*x41 + x13*x8 + x17*x4
    x43 = parms[26]*(x15*x25 + x21*x24 - x28 - x30 + x31) + parms[27]*(-x32 - x33 - x34) + parms[28]*(dq[0]*dq[1]*x0*x17 + dq[0]*dq[2]*x0*x17 - x35 - x38 - x42) + parms[29]*x23
    x44 = 0.05*x13
    x45 = x13*x22 + x16*x17
    x46 = x21*x25 + x27*x29
    x47 = x17*x8
    x48 = x13*x4
    x49 = x0*x36
    x50 = x1*x39
    x51 = -x49 - x50
    x52 = dq[2]*x51
    x53 = x47 - x48 + x52
    x54 = x0*x6
    x55 = dq[2]*x49 + dq[2]*x50 + x13*x54 + x17*x7
    x56 = parms[26]*(x32 - x34 - x46) + parms[27]*(x0*x1*x14*x25 + x13*x14*x17*x20 - x28 - x30 - x31) + parms[28]*(-x53 - x55) + parms[29]*x45
    x57 = 0.05*x17
    x58 = 0.05*x14
    x59 = 0.28*ddq[0]*x1 - x20*x58 - x26*x58 - 0.56*x54
    x60 = dq[1] + dq[2]
    x61 = parms[21]*x51 + parms[23]*x41 + parms[24]*x60
    x62 = -x53
    x63 = parms[22]*x51 + parms[24]*x41 + parms[25]*x60
    x64 = parms[20]*x42 + parms[21]*x62 + parms[22]*x31 + parms[27]*x59 - parms[28]*x23 + x41*x63 - x60*x61
    x65 = parms[20]*x51 + parms[21]*x41 + parms[22]*x60
    x66 = parms[21]*x42 + parms[23]*x62 + parms[24]*x31 - parms[26]*x59 + parms[28]*x45 - x51*x63 + x60*x65
    x67 = dq[1]*parms[12] - parms[10]*x9 - parms[11]*x10
    x68 = parms[22]*x42 + parms[24]*x62 + parms[25]*x31 + parms[26]*x23 - parms[27]*x45 - x41*x65 + x51*x61
    tau_1 = ddq[0]*parms[5] - x0*(ddq[1]*parms[12] - dq[1]*x11 + parms[10]*x8 + parms[11]*x4 + parms[18]*x2 - x10*x12 + x13*x64 - x17*x66 - x43*x44 - x56*x57) - x1*(ddq[1]*parms[14] + dq[1]*x67 + parms[11]*x8 + parms[13]*x4 - parms[18]*x18 - 0.28*parms[26]*(x47 - x48 + x52 - x55) - 0.28*parms[27]*(dq[2]*x40 + x17*x54 - x35 - x38 + x42) - 0.28*parms[28]*(-x33 - x46) - 0.28*parms[29]*x59 + x12*x9 + x13*x66 + x17*x64 - x43*x57 + x44*x56)
    tau_2 = ddq[1]*parms[15] + parms[12]*x8 + parms[14]*x4 - parms[16]*x2 + parms[17]*x18 + x10*x67 - x11*x9 + 0.28*x13*x43 + 0.28*x17*x56 + x68
    tau_3 = x68
    return numpy.array([tau_1, tau_2, tau_3])


def H(q, dq, ddq):
    """回归矩阵 tau = H pi。

    返回形状 3x30。
    """
    x0 = numpy.sin(q[1])
    x1 = numpy.cos(q[1])
    x2 = dq[0]*dq[1]
    x3 = x1*x2
    x4 = x0*x3
    x5 = ddq[0]*x0
    x6 = -x3 - x5
    x7 = ddq[0]*x1
    x8 = dq[0]**2
    x9 = x0*x1*x8
    x10 = dq[1]**2
    x11 = -x10
    x12 = x0**2*x8
    x13 = dq[0]*dq[1]*x0 - x7
    x14 = -x9
    x15 = x1**2
    x16 = x15*x8
    x17 = numpy.cos(q[2])
    x18 = dq[1] + dq[2]
    x19 = dq[0]*x17
    x20 = x0*x19
    x21 = numpy.sin(q[2])
    x22 = dq[0]*x21
    x23 = x1*x22
    x24 = -x20 - x23
    x25 = x18*x24
    x26 = x17*x25
    x27 = x13*x21
    x28 = x17*x6
    x29 = x0*x22
    x30 = x1*x19
    x31 = x29 - x30
    x32 = dq[2]*x31
    x33 = x27 + x28 + x32
    x34 = x21*x25
    x35 = x18*x31
    x36 = x33 + x35
    x37 = x21*x6
    x38 = dq[2]*x24
    x39 = -x13*x17 + x37 + x38
    x40 = -x25 - x39
    x41 = x18**2
    x42 = -x24**2
    x43 = x41 + x42
    x44 = x24*x31
    x45 = ddq[1] + ddq[2]
    x46 = x44 + x45
    x47 = x21*x35
    x48 = -x39
    x49 = x17*x35
    x50 = x31**2
    x51 = -x41 + x50
    x52 = -x44
    x53 = x45 + x52
    x54 = x0*x2
    x55 = 0.05*x12 + 0.05*x16 + 0.56*x54 - 0.28*x7
    x56 = x17*x21
    x57 = x17**2
    x58 = x12*x56
    x59 = x21**2
    x60 = x59*x9
    x61 = x16*x56 + x45 + x57*x9 - x58 - x60
    x62 = 0.05*x17
    x63 = x16*x57
    x64 = x12*x59
    x65 = 2*x56*x9
    x66 = 2*dq[1]*dq[2] + dq[2]**2 + x10
    x67 = -x63 - x64 + x65 - x66
    x68 = 0.05*x21
    x69 = 0.28*x17
    x70 = 0.28*x21
    x71 = dq[2]*x20
    x72 = dq[2]*x23
    x73 = -x55
    x74 = x16*x59
    x75 = x12*x57
    x76 = -x65 - x66 - x74 - x75
    x77 = x0*x1*x57*x8 + x15*x17*x21*x8 - x45 - x58 - x60
    x78 = -9.81*x1
    x79 = 0.28*ddq[1] + 0.05*x5 + x78 + 0.28*x9
    x80 = 9.81*x0
    x81 = -0.28*x10 - 0.28*x16 - 0.05*x7 - x80
    x82 = x17*x81 + x21*x79
    x83 = x21*x82
    x84 = x17*x79 - x21*x81
    x85 = -x84
    x86 = dq[0]*dq[1]*x0*x21 + dq[0]*dq[2]*x0*x21 - dq[2]*x30 - x17*x3 - x33
    x87 = -x17*x54 - x21*x3 - x39 - x71 - x72
    x88 = x17*x82
    x89 = -x42 - x50
    x90 = x33 - x35
    x91 = x25 - x39
    return numpy.array([
        [0, 0, 0, 0, 0, ddq[0], 0, 0, 0, 0, -x0*x6 + x4, -x0*(2*dq[0]*dq[1]*x0 - x7) - x1*(-2*x3 - x5), -x0*(ddq[1] + x9) - x1*(-x11 - x12), -x1*x13 - x4, -x0*(x11 + x16) - x1*(ddq[1] + x14), 0, 0, 0, 0, 0, -x0*(x17*x33 - x34) - x1*(x21*x33 + x26), -x0*(x17*x40 - x21*x36) - x1*(x17*x36 + x21*x40), -x0*(x17*x46 - x21*x43) - x1*(x17*x43 + x21*x46), -x0*(-x21*x48 - x49) - x1*(x17*x48 - x47), -x0*(x17*x51 - x21*x53) - x1*(x17*x53 + x21*x51), -x0*(x34 + x49) - x1*(-x26 + x47), -x0*(-x21*x55 - x61*x62 - x67*x68) - x1*(0.28*x13*x17 + x17*x55 + x3*x70 - 0.28*x37 - 0.28*x38 + x54*x69 - x61*x68 + x62*x67 + 0.28*x71 + 0.28*x72), -x0*(x17*x73 - x62*x76 - x68*x77) - x1*(0.28*dq[0]*dq[1]*x1*x17 + 0.28*dq[0]*dq[2]*x1*x17 - 0.28*dq[2]*x29 + 0.05*x17*x77 + x21*x73 - 0.28*x27 - 0.28*x28 - 0.28*x32 - x54*x70 - x68*x76), -x0*(x17*x85 - x62*x86 - x68*x87 - x83) - x1*(x21*x85 + x62*x87 + 0.28*x63 + 0.28*x64 - x68*x86 + 0.28*x74 + 0.28*x75 + x88), -x0*(-x62*x84 - 0.05*x83) - x1*(0.014*x12 + 0.014*x16 + 0.1568*x54 - x68*x84 - 0.0784*x7 + 0.05*x88)],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x14, x12 - x16, -x5, x9, -x7, ddq[1], x78, x80, 0, 0, x52, x89, x90, x44, x91, x45, x61*x69 + x67*x70 + x84, 0.28*x17*x76 + 0.28*x21*x77 - x82, x69*x86 + x70*x87, x69*x84 + 0.28*x83],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x52, x89, x90, x44, x91, x45, x84, -x82, 0, 0],
    ])


def Hb(q, dq, ddq):
    """最小参数回归矩阵 tau = Hb pi_b。

    返回形状 3x15。
    """
    x0 = numpy.sin(q[1])
    x1 = numpy.cos(q[1])
    x2 = dq[0]*x1
    x3 = dq[1]*x2
    x4 = ddq[0]*x0
    x5 = -x3 - x4
    x6 = ddq[0]*x1
    x7 = dq[0]**2
    x8 = x0*x1*x7
    x9 = dq[1]**2
    x10 = -x9
    x11 = x0**2*x7
    x12 = -x8
    x13 = x1**2
    x14 = x13*x7
    x15 = numpy.cos(q[2])
    x16 = dq[1] + dq[2]
    x17 = dq[0]*x0
    x18 = x15*x17
    x19 = numpy.sin(q[2])
    x20 = x19*x2
    x21 = -x18 - x20
    x22 = x16*x21
    x23 = x15*x22
    x24 = dq[0]*dq[1]*x0 - x6
    x25 = x19*x24
    x26 = x15*x5
    x27 = x17*x19
    x28 = -x15*x2 + x27
    x29 = dq[2]*x28
    x30 = x25 + x26 + x29
    x31 = x19*x22
    x32 = x16*x28
    x33 = x30 + x32
    x34 = dq[2]*x21
    x35 = x19*x5
    x36 = -x15*x24 + x34 + x35
    x37 = -x22 - x36
    x38 = x16**2
    x39 = -x21**2
    x40 = x38 + x39
    x41 = x21*x28
    x42 = ddq[1] + ddq[2]
    x43 = x41 + x42
    x44 = x28**2
    x45 = -x38 + x44
    x46 = -x41
    x47 = x42 + x46
    x48 = dq[1]*x17
    x49 = 0.05*x11 + 0.05*x14 + 0.56*x48 - 0.28*x6
    x50 = x15*x19
    x51 = x15**2
    x52 = x11*x50
    x53 = x19**2
    x54 = x53*x8
    x55 = x14*x50 + x42 + x51*x8 - x52 - x54
    x56 = 0.05*x15
    x57 = 2*x50*x8
    x58 = 2*dq[1]*dq[2] + dq[2]**2 + x9
    x59 = -x11*x53 - x14*x51 + x57 - x58
    x60 = 0.05*x19
    x61 = 0.28*x15
    x62 = 0.28*x19
    x63 = 0.28*dq[2]
    x64 = -x49
    x65 = -x11*x51 - x14*x53 - x57 - x58
    x66 = x0*x1*x51*x7 + x13*x15*x19*x7 - x42 - x52 - x54
    x67 = -9.81*x1
    x68 = 9.81*x0
    x69 = -x39 - x44
    x70 = x30 - x32
    x71 = x22 - x36
    x72 = 0.28*ddq[1] + 0.05*x4 + x67 + 0.28*x8
    x73 = -0.28*x14 - 0.05*x6 - x68 - 0.28*x9
    x74 = x15*x72 - x19*x73
    x75 = x15*x73 + x19*x72
    return numpy.array([
        [ddq[0], x0*x3 - x0*x5, -x0*(2*dq[0]*dq[1]*x0 - x6) - x1*(-2*x3 - x4), -x0*(ddq[1] + x8) - x1*(-x10 - x11), -x0*(x10 + x14) - x1*(ddq[1] + x12), 0, 0, 0, -x0*(x15*x30 - x31) - x1*(x19*x30 + x23), -x0*(x15*x37 - x19*x33) - x1*(x15*x33 + x19*x37), -x0*(x15*x43 - x19*x40) - x1*(x15*x40 + x19*x43), -x0*(x15*x45 - x19*x47) - x1*(x15*x47 + x19*x45), -x0*(x15*x32 + x31) - x1*(x19*x32 - x23), -x0*(-x19*x49 - x55*x56 - x59*x60) - x1*(0.28*x15*x24 + x15*x49 + x18*x63 + x20*x63 + x3*x62 - 0.28*x34 - 0.28*x35 + x48*x61 - x55*x60 + x56*x59), -x0*(x15*x64 - x56*x65 - x60*x66) - x1*(0.28*dq[0]*dq[1]*x1*x15 + 0.28*dq[0]*dq[2]*x1*x15 - 0.28*dq[1]*x27 + 0.05*x15*x66 + x19*x64 - 0.28*x25 - 0.28*x26 - x27*x63 - 0.28*x29 - x60*x65)],
        [0, x12, x11 - x14, -x4, -x6, ddq[1], x67, x68, x46, x69, x70, x71, x42, x55*x61 + x59*x62 + x74, 0.28*x15*x65 + 0.28*x19*x66 - x75],
        [0, 0, 0, 0, 0, 0, 0, 0, x46, x69, x70, x71, x42, x74, -x75],
    ])


def baseparms(parms):
    """最小参数集 pi_b（用原参数表示）。

    返回形状 15x1。
    """
    x0 = (49/625)*parms[29]
    baseparms_1 = parms[13] + parms[23] + (1/10)*parms[28] + (809/10000)*parms[29] + parms[5]
    baseparms_2 = parms[10] - parms[13] - x0
    baseparms_3 = parms[11]
    baseparms_4 = parms[12] - 7/25*parms[28] - 7/500*parms[29]
    baseparms_5 = parms[14]
    baseparms_6 = parms[15] + x0
    baseparms_7 = parms[16] + (7/25)*parms[29]
    baseparms_8 = parms[17]
    baseparms_9 = parms[20] - parms[23]
    baseparms_10 = parms[21]
    baseparms_11 = parms[22]
    baseparms_12 = parms[24]
    baseparms_13 = parms[25]
    baseparms_14 = parms[26]
    baseparms_15 = parms[27]
    return numpy.array([baseparms_1, baseparms_2, baseparms_3, baseparms_4, baseparms_5, baseparms_6, baseparms_7, baseparms_8, baseparms_9, baseparms_10, baseparms_11, baseparms_12, baseparms_13, baseparms_14, baseparms_15])
