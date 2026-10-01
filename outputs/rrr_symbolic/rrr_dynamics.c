/* 由 robot_model 自动生成，请勿手改。
 *
 * 机器人: rrr_arm    自由度: 3
 * 约定: modified
 */

#include <math.h>

// parms 顺序（共 30 项，Khalil 排列）:
//   * [ 0] L_1xx
//     [ 1] L_1xy
//     [ 2] L_1xz
//     [ 3] L_1yy
//     [ 4] L_1yz
//     [ 5] L_1zz
//     [ 6] l_1x
//     [ 7] l_1y
//     [ 8] l_1z
//     [ 9] m_1
//   * [10] L_2xx
//     [11] L_2xy
//     [12] L_2xz
//     [13] L_2yy
//     [14] L_2yz
//     [15] L_2zz
//     [16] l_2x
//     [17] l_2y
//     [18] l_2z
//     [19] m_2
//   * [20] L_3xx
//     [21] L_3xy
//     [22] L_3xz
//     [23] L_3yy
//     [24] L_3yz
//     [25] L_3zz
//     [26] l_3x
//     [27] l_3y
//     [28] l_3z
//     [29] m_3

void T(const double *q, double *out);
void J(const double *q, double *out);
void M(const double *q, const double *parms, double *out);
void c(const double *q, const double *dq, const double *parms, double *out);
void C(const double *q, const double *dq, const double *parms, double *out);
void g(const double *q, const double *parms, double *out);
void tau(const double *q, const double *dq, const double *ddq, const double *parms, double *out);
void H(const double *q, const double *dq, const double *ddq, double *out);
void Hb(const double *q, const double *dq, const double *ddq, double *out);
void baseparms(const double *parms, double *out);


/* 末端位姿（齐次变换）。out: 4x4，行优先。 */
void T(const double *q, double *out)
{
    const double x0 = cos(q[0]);
    const double x1 = sin(q[1]);
    const double x2 = sin(q[2]);
    const double x3 = x1*x2;
    const double x4 = cos(q[1]);
    const double x5 = cos(q[2]);
    const double x6 = x1*x5;
    const double x7 = x2*x4;
    const double x8 = sin(q[0]);
    out[0] = -x0*x3 + x0*x4*x5;
    out[1] = -x0*x6 - x0*x7;
    out[2] = -x8;
    out[3] = 0.28000000000000003*x0*x4 - 0.050000000000000003*x8;
    out[4] = -x3*x8 + x4*x5*x8;
    out[5] = -x6*x8 - x7*x8;
    out[6] = x0;
    out[7] = 0.050000000000000003*x0 + 0.28000000000000003*x4*x8;
    out[8] = -x6 - x7;
    out[9] = x3 - x4*x5;
    out[10] = 0;
    out[11] = 0.29999999999999999 - 0.28000000000000003*x1;
    out[12] = 0;
    out[13] = 0;
    out[14] = 0;
    out[15] = 1;
}

/* 末端几何雅可比 [v; omega] = J dq。out: 6x3，行优先。 */
void J(const double *q, double *out)
{
    const double x0 = cos(q[0]);
    const double x1 = sin(q[0]);
    const double x2 = cos(q[1]);
    const double x3 = 0.050000000000000003*x0 + 0.28000000000000003*x1*x2;
    const double x4 = 0.28000000000000003*sin(q[1]);
    const double x5 = 0.28000000000000003*x0*x2 - 0.050000000000000003*x1;
    const double x6 = -x1;
    out[0] = -x3;
    out[1] = -x0*x4;
    out[2] = 0;
    out[3] = x5;
    out[4] = -x1*x4;
    out[5] = 0;
    out[6] = 0;
    out[7] = -x0*x5 - x1*x3;
    out[8] = 0;
    out[9] = 0;
    out[10] = x6;
    out[11] = x6;
    out[12] = 0;
    out[13] = x0;
    out[14] = x0;
    out[15] = 1;
    out[16] = 0;
    out[17] = 0;
}

/* 惯性矩阵。out: 3x3，行优先。 */
void M(const double *q, const double *parms, double *out)
{
    const double x0 = sin(q[1]);
    const double x1 = cos(q[1]);
    const double x2 = cos(q[2]);
    const double x3 = x0*x2;
    const double x4 = sin(q[2]);
    const double x5 = x1*x4;
    const double x6 = x3 + x5;
    const double x7 = 0.050000000000000003*x3 + 0.050000000000000003*x5;
    const double x8 = parms[28]*x6 + parms[29]*x7;
    const double x9 = x0*x4;
    const double x10 = x1*x2;
    const double x11 = -x10 + x9;
    const double x12 = -0.050000000000000003*x10 + 0.050000000000000003*x9;
    const double x13 = parms[28]*x11 + parms[29]*x12;
    const double x14 = 0.050000000000000003*x4;
    const double x15 = 0.28000000000000003*x1;
    const double x16 = -x6;
    const double x17 = parms[21]*x16 + parms[23]*x11 - parms[26]*x15 + parms[28]*x12;
    const double x18 = parms[20]*x16 + parms[21]*x11 + parms[27]*x15 - parms[28]*x7;
    const double x19 = 0.28000000000000003*parms[26];
    const double x20 = 0.28000000000000003*parms[27];
    const double x21 = 0.28000000000000003*x2;
    const double x22 = 0.28000000000000003*x4;
    const double x23 = parms[22]*x16 + parms[24]*x11 + parms[26]*x7 - parms[27]*x12;
    const double x24 = -parms[12]*x0 - parms[14]*x1 + x13*x22 + x21*x8 + x23;
    const double x25 = parms[25] + x19*x2 - x20*x4;
    out[0] = parms[5] - x0*(-parms[10]*x0 - parms[11]*x1 - x13*x14 - x17*x4 + x18*x2 - 0.050000000000000003*x2*x8) - x1*(-parms[11]*x0 - parms[13]*x1 - 0.078400000000000011*parms[29]*x1 + x11*x19 + 0.050000000000000003*x13*x2 - x14*x8 - x16*x20 + x17*x2 + x18*x4);
    out[1] = x24;
    out[2] = x23;
    out[3] = x24;
    out[4] = parms[15] + x21*(parms[26] + parms[29]*x21) + x22*(-parms[27] + 0.28000000000000003*parms[29]*x4) + x25;
    out[5] = x25;
    out[6] = x23;
    out[7] = x25;
    out[8] = parms[25];
}

/* 科氏 / 离心项。out: 3x1，行优先。 */
void c(const double *q, const double *dq, const double *parms, double *out)
{
    const double x0 = sin(q[1]);
    const double x1 = cos(q[1]);
    const double x2 = dq[0]*x1;
    const double x3 = dq[0]*x0;
    const double x4 = dq[1]*parms[14] - parms[11]*x3 - parms[13]*x2;
    const double x5 = dq[1]*parms[15] - parms[12]*x3 - parms[14]*x2;
    const double x6 = cos(q[2]);
    const double x7 = 0.28000000000000003*x6;
    const double x8 = pow(dq[0], 2);
    const double x9 = x0*x1*x8;
    const double x10 = sin(q[2]);
    const double x11 = pow(dq[1], 2);
    const double x12 = pow(x1, 2);
    const double x13 = x12*x8;
    const double x14 = -0.28000000000000003*x11 - 0.28000000000000003*x13;
    const double x15 = -x10*x14 + x7*x9;
    const double x16 = dq[0]*x6;
    const double x17 = x1*x16;
    const double x18 = dq[2]*x17;
    const double x19 = dq[0]*x10;
    const double x20 = x0*x19;
    const double x21 = -x17 + x20;
    const double x22 = dq[2]*x21;
    const double x23 = pow(x10, 2);
    const double x24 = pow(x0, 2);
    const double x25 = x24*x8;
    const double x26 = x10*x6;
    const double x27 = pow(x6, 2);
    const double x28 = x0*x1*x27*x8 + x10*x12*x6*x8 - x23*x9 - x25*x26;
    const double x29 = 2*x26*x9;
    const double x30 = x13*x23 + x25*x27;
    const double x31 = 2*dq[1];
    const double x32 = pow(dq[2], 2) + dq[2]*x31 + x11;
    const double x33 = parms[26]*x28 + parms[27]*(-x29 - x30 - x32) + parms[28]*(dq[0]*dq[2]*x0*x10 - x18 - x22) + parms[29]*x15;
    const double x34 = 0.28000000000000003*x10;
    const double x35 = x14*x6 + x34*x9;
    const double x36 = x0*x16;
    const double x37 = x1*x19;
    const double x38 = -x36 - x37;
    const double x39 = dq[2]*x38;
    const double x40 = dq[2]*x36 + dq[2]*x37;
    const double x41 = x13*x27 + x23*x25;
    const double x42 = parms[26]*(x29 - x32 - x41) + parms[27]*x28 + parms[28]*(-x39 - x40) + parms[29]*x35;
    const double x43 = 0.050000000000000003*x10;
    const double x44 = 0.050000000000000003*x8;
    const double x45 = -0.56000000000000005*dq[1]*x3 - x12*x44 - x24*x44;
    const double x46 = -dq[1]*x17 + dq[1]*x20 + x22;
    const double x47 = -x39;
    const double x48 = dq[1]*x36 + dq[1]*x37 + x47;
    const double x49 = dq[1] + dq[2];
    const double x50 = parms[20]*x38 + parms[21]*x21 + parms[22]*x49;
    const double x51 = parms[22]*x38 + parms[24]*x21 + parms[25]*x49;
    const double x52 = parms[21]*x46 + parms[23]*x48 - parms[26]*x45 + parms[28]*x35 - x38*x51 + x49*x50;
    const double x53 = parms[21]*x38 + parms[23]*x21 + parms[24]*x49;
    const double x54 = parms[20]*x46 + parms[21]*x48 + parms[27]*x45 - parms[28]*x15 + x21*x51 - x49*x53;
    const double x55 = parms[11]*x2;
    const double x56 = dq[1]*parms[12] - parms[10]*x3 - x55;
    const double x57 = parms[22]*x46 + parms[24]*x48 + parms[26]*x15 - parms[27]*x35 - x21*x50 + x38*x53;
    out[0] = -x0*(dq[0]*dq[1]*parms[11]*x0 - dq[1]*parms[10]*x2 - dq[1]*x4 - x10*x52 - x2*x5 - 0.050000000000000003*x33*x6 - x42*x43 + x54*x6) - x1*(dq[0]*dq[1]*parms[13]*x0 + dq[0]*x0*x5 - dq[1]*x55 + dq[1]*x56 - 0.28000000000000003*parms[26]*(-x31*x36 - x31*x37 - x40 - x47) - 0.28000000000000003*parms[27]*(dq[2]*x20 - x17*x31 - x18 + x20*x31 + x22) - 0.28000000000000003*parms[28]*(-x30 - x41) - 0.28000000000000003*parms[29]*x45 + x10*x54 - x33*x43 + 0.050000000000000003*x42*x6 + x52*x6);
    out[1] = -dq[1]*parms[12]*x2 + dq[1]*parms[14]*x3 + x2*x56 - x3*x4 + x33*x7 + x34*x42 + x57;
    out[2] = x57;
}

/* 科氏矩阵（Christoffel，c = C dq）。out: 3x3，行优先。 */
void C(const double *q, const double *dq, const double *parms, double *out)
{
    const double x0 = sin(q[1]);
    const double x1 = sin(q[2]);
    const double x2 = x0*x1;
    const double x3 = cos(q[1]);
    const double x4 = cos(q[2]);
    const double x5 = x3*x4;
    const double x6 = x2 - x5;
    const double x7 = -x6;
    const double x8 = 0.050000000000000003*x2 - 0.050000000000000003*x5;
    const double x9 = -x8;
    const double x10 = parms[28]*x7 + parms[29]*x9;
    const double x11 = 0.050000000000000003*x4;
    const double x12 = x10*x11;
    const double x13 = parms[28]*x6 + parms[29]*x8;
    const double x14 = x11*x13;
    const double x15 = x0*x4;
    const double x16 = x1*x3;
    const double x17 = x15 + x16;
    const double x18 = parms[21]*x6;
    const double x19 = 0.050000000000000003*x15 + 0.050000000000000003*x16;
    const double x20 = parms[28]*x19;
    const double x21 = parms[23]*x17 + x18 + x20;
    const double x22 = parms[20]*x6 + parms[21]*x17 - parms[28]*x9;
    const double x23 = 0.28000000000000003*x3;
    const double x24 = -x17;
    const double x25 = parms[21]*x24 + parms[23]*x6 - parms[26]*x23 + parms[28]*x8;
    const double x26 = x25*x4;
    const double x27 = parms[20]*x24 + parms[27]*x23 + x18 - x20;
    const double x28 = x1*x27;
    const double x29 = (1.0/2.0)*x0;
    const double x30 = 0.28000000000000003*parms[26];
    const double x31 = x24*x30;
    const double x32 = 0.28000000000000003*parms[27];
    const double x33 = x32*x6;
    const double x34 = 0.050000000000000003*x1;
    const double x35 = x10*x34;
    const double x36 = x1*x25 + x13*x34 - x27*x4;
    const double x37 = (1.0/2.0)*x3;
    const double x38 = x29*(-x1*x21 - x12 - x14 + x22*x4 - x26 - x28) + x37*(x1*x22 + x21*x4 - x31 - x33 - x35 - x36);
    const double x39 = -x38;
    const double x40 = parms[11]*x0;
    const double x41 = parms[28]*x17 + parms[29]*x19;
    const double x42 = x34*x41;
    const double x43 = x0*x30 + x21;
    const double x44 = -x0*x32 + x22;
    const double x45 = parms[11]*x3;
    const double x46 = x11*x41;
    const double x47 = 0.078400000000000011*parms[29];
    const double x48 = -1.0/2.0*x0*(-parms[13]*x3 + x14 - x24*x32 + x26 + x28 - x3*x47 - x30*x7 - x40 - x42) + x29*(-parms[10]*x3 - x1*x43 - x12 + x4*x44 + x40 - x42) + x37*(-parms[10]*x0 - x36 - x45 - x46) + x37*(parms[13]*x0 + x0*x47 + x1*x44 - x31 - x33 - x35 + x4*x43 - x45 + x46);
    const double x49 = -x48;
    const double x50 = x10*x4;
    const double x51 = 0.14000000000000001*x4;
    const double x52 = x13*x51 + 0.14000000000000001*x50;
    const double x53 = parms[22]*x6 + parms[24]*x17 + parms[26]*x9 - parms[27]*x19;
    const double x54 = x52 + x53;
    const double x55 = 0.14000000000000001*x1;
    const double x56 = parms[26]*x55 + parms[27]*x51 - 0.14000000000000001*x4*(-parms[27] + 0.28000000000000003*parms[29]*x1) + x55*(parms[26] + 0.28000000000000003*parms[29]*x4);
    const double x57 = -x56;
    const double x58 = -x52;
    out[0] = dq[1]*x49 + dq[2]*x39;
    out[1] = dq[0]*x49 + dq[1]*(-parms[12]*x3 + parms[14]*x0 + 0.28000000000000003*x1*x41 + 0.28000000000000003*x50 + x53) + dq[2]*x54;
    out[2] = dq[0]*x39 + dq[1]*x54 + dq[2]*x53;
    out[3] = dq[0]*x48 + dq[2]*x52;
    out[4] = dq[2]*x57;
    out[5] = dq[0]*x52 + dq[1]*x57 + dq[2]*(-x1*x30 - x32*x4);
    out[6] = dq[0]*x38 + dq[1]*x58;
    out[7] = dq[0]*x58 + dq[1]*x56;
    out[8] = 0;
}

/* 重力项。out: 3x1，行优先。 */
void g(const double *q, const double *parms, double *out)
{
    const double x0 = cos(q[1]);
    const double x1 = sin(q[1]);
    const double x2 = 9.8100000000000005*x1;
    const double x3 = sin(q[2]);
    const double x4 = cos(q[2]);
    const double x5 = 9.8100000000000005*x0;
    const double x6 = x2*x3 - x4*x5;
    const double x7 = parms[28]*x3;
    const double x8 = 0.050000000000000003*parms[29];
    const double x9 = x3*x8;
    const double x10 = -x2*x4 - x3*x5;
    const double x11 = 0.28000000000000003*parms[29];
    const double x12 = parms[26]*x6 - parms[27]*x10;
    out[0] = -x0*(-parms[18]*x2 + parms[28]*x10*x4 + 0.050000000000000003*parms[29]*x10*x4 - x6*x7 - x6*x9) - x1*(9.8100000000000005*parms[18]*x0 - parms[28]*x4*x6 - x10*x7 - x10*x9 - x4*x6*x8);
    out[1] = -parms[16]*x5 + parms[17]*x2 + x10*x11*x3 + x11*x4*x6 + x12;
    out[2] = x12;
}

/* 逆动力学力矩 tau = M ddq + c + g。out: 3x1，行优先。 */
void tau(const double *q, const double *dq, const double *ddq, const double *parms, double *out)
{
    const double x0 = sin(q[1]);
    const double x1 = cos(q[1]);
    const double x2 = 9.8100000000000005*x1;
    const double x3 = ddq[0]*x1;
    const double x4 = dq[0]*dq[1]*x0 - x3;
    const double x5 = ddq[0]*x0;
    const double x6 = dq[0]*dq[1];
    const double x7 = x1*x6;
    const double x8 = -x5 - x7;
    const double x9 = dq[0]*x0;
    const double x10 = dq[0]*x1;
    const double x11 = dq[1]*parms[14] - parms[11]*x9 - parms[13]*x10;
    const double x12 = dq[1]*parms[15] - parms[12]*x9 - parms[14]*x10;
    const double x13 = cos(q[2]);
    const double x14 = pow(dq[0], 2);
    const double x15 = x0*x1*x14;
    const double x16 = 0.28000000000000003*ddq[1] + 0.28000000000000003*x15 - x2 + 0.050000000000000003*x5;
    const double x17 = sin(q[2]);
    const double x18 = 9.8100000000000005*x0;
    const double x19 = pow(dq[1], 2);
    const double x20 = pow(x1, 2);
    const double x21 = x14*x20;
    const double x22 = -x18 - 0.28000000000000003*x19 - 0.28000000000000003*x21 - 0.050000000000000003*x3;
    const double x23 = x13*x16 - x17*x22;
    const double x24 = x13*x17;
    const double x25 = pow(x13, 2);
    const double x26 = pow(x0, 2);
    const double x27 = x14*x26;
    const double x28 = x24*x27;
    const double x29 = pow(x17, 2);
    const double x30 = x15*x29;
    const double x31 = ddq[1] + ddq[2];
    const double x32 = 2*x15*x24;
    const double x33 = x21*x29 + x25*x27;
    const double x34 = 2*dq[1]*dq[2] + pow(dq[2], 2) + x19;
    const double x35 = x13*x7;
    const double x36 = dq[0]*x13;
    const double x37 = x1*x36;
    const double x38 = dq[2]*x37;
    const double x39 = dq[0]*x17;
    const double x40 = x0*x39;
    const double x41 = -x37 + x40;
    const double x42 = dq[2]*x41 + x13*x8 + x17*x4;
    const double x43 = parms[26]*(x15*x25 + x21*x24 - x28 - x30 + x31) + parms[27]*(-x32 - x33 - x34) + parms[28]*(dq[0]*dq[1]*x0*x17 + dq[0]*dq[2]*x0*x17 - x35 - x38 - x42) + parms[29]*x23;
    const double x44 = 0.050000000000000003*x13;
    const double x45 = x13*x22 + x16*x17;
    const double x46 = x21*x25 + x27*x29;
    const double x47 = x17*x8;
    const double x48 = x13*x4;
    const double x49 = x0*x36;
    const double x50 = x1*x39;
    const double x51 = -x49 - x50;
    const double x52 = dq[2]*x51;
    const double x53 = x47 - x48 + x52;
    const double x54 = x0*x6;
    const double x55 = dq[2]*x49 + dq[2]*x50 + x13*x54 + x17*x7;
    const double x56 = parms[26]*(x32 - x34 - x46) + parms[27]*(x0*x1*x14*x25 + x13*x14*x17*x20 - x28 - x30 - x31) + parms[28]*(-x53 - x55) + parms[29]*x45;
    const double x57 = 0.050000000000000003*x17;
    const double x58 = 0.050000000000000003*x14;
    const double x59 = 0.28000000000000003*ddq[0]*x1 - x20*x58 - x26*x58 - 0.56000000000000005*x54;
    const double x60 = dq[1] + dq[2];
    const double x61 = parms[21]*x51 + parms[23]*x41 + parms[24]*x60;
    const double x62 = -x53;
    const double x63 = parms[22]*x51 + parms[24]*x41 + parms[25]*x60;
    const double x64 = parms[20]*x42 + parms[21]*x62 + parms[22]*x31 + parms[27]*x59 - parms[28]*x23 + x41*x63 - x60*x61;
    const double x65 = parms[20]*x51 + parms[21]*x41 + parms[22]*x60;
    const double x66 = parms[21]*x42 + parms[23]*x62 + parms[24]*x31 - parms[26]*x59 + parms[28]*x45 - x51*x63 + x60*x65;
    const double x67 = dq[1]*parms[12] - parms[10]*x9 - parms[11]*x10;
    const double x68 = parms[22]*x42 + parms[24]*x62 + parms[25]*x31 + parms[26]*x23 - parms[27]*x45 - x41*x65 + x51*x61;
    out[0] = ddq[0]*parms[5] - x0*(ddq[1]*parms[12] - dq[1]*x11 + parms[10]*x8 + parms[11]*x4 + parms[18]*x2 - x10*x12 + x13*x64 - x17*x66 - x43*x44 - x56*x57) - x1*(ddq[1]*parms[14] + dq[1]*x67 + parms[11]*x8 + parms[13]*x4 - parms[18]*x18 - 0.28000000000000003*parms[26]*(x47 - x48 + x52 - x55) - 0.28000000000000003*parms[27]*(dq[2]*x40 + x17*x54 - x35 - x38 + x42) - 0.28000000000000003*parms[28]*(-x33 - x46) - 0.28000000000000003*parms[29]*x59 + x12*x9 + x13*x66 + x17*x64 - x43*x57 + x44*x56);
    out[1] = ddq[1]*parms[15] + parms[12]*x8 + parms[14]*x4 - parms[16]*x2 + parms[17]*x18 + x10*x67 - x11*x9 + 0.28000000000000003*x13*x43 + 0.28000000000000003*x17*x56 + x68;
    out[2] = x68;
}

/* 回归矩阵 tau = H pi。out: 3x30，行优先。 */
void H(const double *q, const double *dq, const double *ddq, double *out)
{
    const double x0 = sin(q[1]);
    const double x1 = cos(q[1]);
    const double x2 = dq[0]*dq[1];
    const double x3 = x1*x2;
    const double x4 = x0*x3;
    const double x5 = ddq[0]*x0;
    const double x6 = -x3 - x5;
    const double x7 = ddq[0]*x1;
    const double x8 = pow(dq[0], 2);
    const double x9 = x0*x1*x8;
    const double x10 = pow(dq[1], 2);
    const double x11 = -x10;
    const double x12 = pow(x0, 2)*x8;
    const double x13 = dq[0]*dq[1]*x0 - x7;
    const double x14 = -x9;
    const double x15 = pow(x1, 2);
    const double x16 = x15*x8;
    const double x17 = cos(q[2]);
    const double x18 = dq[1] + dq[2];
    const double x19 = dq[0]*x17;
    const double x20 = x0*x19;
    const double x21 = sin(q[2]);
    const double x22 = dq[0]*x21;
    const double x23 = x1*x22;
    const double x24 = -x20 - x23;
    const double x25 = x18*x24;
    const double x26 = x17*x25;
    const double x27 = x13*x21;
    const double x28 = x17*x6;
    const double x29 = x0*x22;
    const double x30 = x1*x19;
    const double x31 = x29 - x30;
    const double x32 = dq[2]*x31;
    const double x33 = x27 + x28 + x32;
    const double x34 = x21*x25;
    const double x35 = x18*x31;
    const double x36 = x33 + x35;
    const double x37 = x21*x6;
    const double x38 = dq[2]*x24;
    const double x39 = -x13*x17 + x37 + x38;
    const double x40 = -x25 - x39;
    const double x41 = pow(x18, 2);
    const double x42 = -pow(x24, 2);
    const double x43 = x41 + x42;
    const double x44 = x24*x31;
    const double x45 = ddq[1] + ddq[2];
    const double x46 = x44 + x45;
    const double x47 = x21*x35;
    const double x48 = -x39;
    const double x49 = x17*x35;
    const double x50 = pow(x31, 2);
    const double x51 = -x41 + x50;
    const double x52 = -x44;
    const double x53 = x45 + x52;
    const double x54 = x0*x2;
    const double x55 = 0.050000000000000003*x12 + 0.050000000000000003*x16 + 0.56000000000000005*x54 - 0.28000000000000003*x7;
    const double x56 = x17*x21;
    const double x57 = pow(x17, 2);
    const double x58 = x12*x56;
    const double x59 = pow(x21, 2);
    const double x60 = x59*x9;
    const double x61 = x16*x56 + x45 + x57*x9 - x58 - x60;
    const double x62 = 0.050000000000000003*x17;
    const double x63 = x16*x57;
    const double x64 = x12*x59;
    const double x65 = 2*x56*x9;
    const double x66 = 2*dq[1]*dq[2] + pow(dq[2], 2) + x10;
    const double x67 = -x63 - x64 + x65 - x66;
    const double x68 = 0.050000000000000003*x21;
    const double x69 = 0.28000000000000003*x17;
    const double x70 = 0.28000000000000003*x21;
    const double x71 = dq[2]*x20;
    const double x72 = dq[2]*x23;
    const double x73 = -x55;
    const double x74 = x16*x59;
    const double x75 = x12*x57;
    const double x76 = -x65 - x66 - x74 - x75;
    const double x77 = x0*x1*x57*x8 + x15*x17*x21*x8 - x45 - x58 - x60;
    const double x78 = -9.8100000000000005*x1;
    const double x79 = 0.28000000000000003*ddq[1] + 0.050000000000000003*x5 + x78 + 0.28000000000000003*x9;
    const double x80 = 9.8100000000000005*x0;
    const double x81 = -0.28000000000000003*x10 - 0.28000000000000003*x16 - 0.050000000000000003*x7 - x80;
    const double x82 = x17*x81 + x21*x79;
    const double x83 = x21*x82;
    const double x84 = x17*x79 - x21*x81;
    const double x85 = -x84;
    const double x86 = dq[0]*dq[1]*x0*x21 + dq[0]*dq[2]*x0*x21 - dq[2]*x30 - x17*x3 - x33;
    const double x87 = -x17*x54 - x21*x3 - x39 - x71 - x72;
    const double x88 = x17*x82;
    const double x89 = -x42 - x50;
    const double x90 = x33 - x35;
    const double x91 = x25 - x39;
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = ddq[0];
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[10] = -x0*x6 + x4;
    out[11] = -x0*(2*dq[0]*dq[1]*x0 - x7) - x1*(-2*x3 - x5);
    out[12] = -x0*(ddq[1] + x9) - x1*(-x11 - x12);
    out[13] = -x1*x13 - x4;
    out[14] = -x0*(x11 + x16) - x1*(ddq[1] + x14);
    out[15] = 0;
    out[16] = 0;
    out[17] = 0;
    out[18] = 0;
    out[19] = 0;
    out[20] = -x0*(x17*x33 - x34) - x1*(x21*x33 + x26);
    out[21] = -x0*(x17*x40 - x21*x36) - x1*(x17*x36 + x21*x40);
    out[22] = -x0*(x17*x46 - x21*x43) - x1*(x17*x43 + x21*x46);
    out[23] = -x0*(-x21*x48 - x49) - x1*(x17*x48 - x47);
    out[24] = -x0*(x17*x51 - x21*x53) - x1*(x17*x53 + x21*x51);
    out[25] = -x0*(x34 + x49) - x1*(-x26 + x47);
    out[26] = -x0*(-x21*x55 - x61*x62 - x67*x68) - x1*(0.28000000000000003*x13*x17 + x17*x55 + x3*x70 - 0.28000000000000003*x37 - 0.28000000000000003*x38 + x54*x69 - x61*x68 + x62*x67 + 0.28000000000000003*x71 + 0.28000000000000003*x72);
    out[27] = -x0*(x17*x73 - x62*x76 - x68*x77) - x1*(0.28000000000000003*dq[0]*dq[1]*x1*x17 + 0.28000000000000003*dq[0]*dq[2]*x1*x17 - 0.28000000000000003*dq[2]*x29 + 0.050000000000000003*x17*x77 + x21*x73 - 0.28000000000000003*x27 - 0.28000000000000003*x28 - 0.28000000000000003*x32 - x54*x70 - x68*x76);
    out[28] = -x0*(x17*x85 - x62*x86 - x68*x87 - x83) - x1*(x21*x85 + x62*x87 + 0.28000000000000003*x63 + 0.28000000000000003*x64 - x68*x86 + 0.28000000000000003*x74 + 0.28000000000000003*x75 + x88);
    out[29] = -x0*(-x62*x84 - 0.050000000000000003*x83) - x1*(0.014000000000000002*x12 + 0.014000000000000002*x16 + 0.15680000000000002*x54 - x68*x84 - 0.078400000000000011*x7 + 0.050000000000000003*x88);
    out[30] = 0;
    out[31] = 0;
    out[32] = 0;
    out[33] = 0;
    out[34] = 0;
    out[35] = 0;
    out[36] = 0;
    out[37] = 0;
    out[38] = 0;
    out[39] = 0;
    out[40] = x14;
    out[41] = x12 - x16;
    out[42] = -x5;
    out[43] = x9;
    out[44] = -x7;
    out[45] = ddq[1];
    out[46] = x78;
    out[47] = x80;
    out[48] = 0;
    out[49] = 0;
    out[50] = x52;
    out[51] = x89;
    out[52] = x90;
    out[53] = x44;
    out[54] = x91;
    out[55] = x45;
    out[56] = x61*x69 + x67*x70 + x84;
    out[57] = 0.28000000000000003*x17*x76 + 0.28000000000000003*x21*x77 - x82;
    out[58] = x69*x86 + x70*x87;
    out[59] = x69*x84 + 0.28000000000000003*x83;
    out[60] = 0;
    out[61] = 0;
    out[62] = 0;
    out[63] = 0;
    out[64] = 0;
    out[65] = 0;
    out[66] = 0;
    out[67] = 0;
    out[68] = 0;
    out[69] = 0;
    out[70] = 0;
    out[71] = 0;
    out[72] = 0;
    out[73] = 0;
    out[74] = 0;
    out[75] = 0;
    out[76] = 0;
    out[77] = 0;
    out[78] = 0;
    out[79] = 0;
    out[80] = x52;
    out[81] = x89;
    out[82] = x90;
    out[83] = x44;
    out[84] = x91;
    out[85] = x45;
    out[86] = x84;
    out[87] = -x82;
    out[88] = 0;
    out[89] = 0;
}

/* 最小参数回归矩阵 tau = Hb pi_b。out: 3x15，行优先。 */
void Hb(const double *q, const double *dq, const double *ddq, double *out)
{
    const double x0 = sin(q[1]);
    const double x1 = cos(q[1]);
    const double x2 = dq[0]*x1;
    const double x3 = dq[1]*x2;
    const double x4 = ddq[0]*x0;
    const double x5 = -x3 - x4;
    const double x6 = ddq[0]*x1;
    const double x7 = pow(dq[0], 2);
    const double x8 = x0*x1*x7;
    const double x9 = pow(dq[1], 2);
    const double x10 = -x9;
    const double x11 = pow(x0, 2)*x7;
    const double x12 = -x8;
    const double x13 = pow(x1, 2);
    const double x14 = x13*x7;
    const double x15 = cos(q[2]);
    const double x16 = dq[1] + dq[2];
    const double x17 = dq[0]*x0;
    const double x18 = x15*x17;
    const double x19 = sin(q[2]);
    const double x20 = x19*x2;
    const double x21 = -x18 - x20;
    const double x22 = x16*x21;
    const double x23 = x15*x22;
    const double x24 = dq[0]*dq[1]*x0 - x6;
    const double x25 = x19*x24;
    const double x26 = x15*x5;
    const double x27 = x17*x19;
    const double x28 = -x15*x2 + x27;
    const double x29 = dq[2]*x28;
    const double x30 = x25 + x26 + x29;
    const double x31 = x19*x22;
    const double x32 = x16*x28;
    const double x33 = x30 + x32;
    const double x34 = dq[2]*x21;
    const double x35 = x19*x5;
    const double x36 = -x15*x24 + x34 + x35;
    const double x37 = -x22 - x36;
    const double x38 = pow(x16, 2);
    const double x39 = -pow(x21, 2);
    const double x40 = x38 + x39;
    const double x41 = x21*x28;
    const double x42 = ddq[1] + ddq[2];
    const double x43 = x41 + x42;
    const double x44 = pow(x28, 2);
    const double x45 = -x38 + x44;
    const double x46 = -x41;
    const double x47 = x42 + x46;
    const double x48 = dq[1]*x17;
    const double x49 = 0.050000000000000003*x11 + 0.050000000000000003*x14 + 0.56000000000000005*x48 - 0.28000000000000003*x6;
    const double x50 = x15*x19;
    const double x51 = pow(x15, 2);
    const double x52 = x11*x50;
    const double x53 = pow(x19, 2);
    const double x54 = x53*x8;
    const double x55 = x14*x50 + x42 + x51*x8 - x52 - x54;
    const double x56 = 0.050000000000000003*x15;
    const double x57 = 2*x50*x8;
    const double x58 = 2*dq[1]*dq[2] + pow(dq[2], 2) + x9;
    const double x59 = -x11*x53 - x14*x51 + x57 - x58;
    const double x60 = 0.050000000000000003*x19;
    const double x61 = 0.28000000000000003*x15;
    const double x62 = 0.28000000000000003*x19;
    const double x63 = 0.28000000000000003*dq[2];
    const double x64 = -x49;
    const double x65 = -x11*x51 - x14*x53 - x57 - x58;
    const double x66 = x0*x1*x51*x7 + x13*x15*x19*x7 - x42 - x52 - x54;
    const double x67 = -9.8100000000000005*x1;
    const double x68 = 9.8100000000000005*x0;
    const double x69 = -x39 - x44;
    const double x70 = x30 - x32;
    const double x71 = x22 - x36;
    const double x72 = 0.28000000000000003*ddq[1] + 0.050000000000000003*x4 + x67 + 0.28000000000000003*x8;
    const double x73 = -0.28000000000000003*x14 - 0.050000000000000003*x6 - x68 - 0.28000000000000003*x9;
    const double x74 = x15*x72 - x19*x73;
    const double x75 = x15*x73 + x19*x72;
    out[0] = ddq[0];
    out[1] = x0*x3 - x0*x5;
    out[2] = -x0*(2*dq[0]*dq[1]*x0 - x6) - x1*(-2*x3 - x4);
    out[3] = -x0*(ddq[1] + x8) - x1*(-x10 - x11);
    out[4] = -x0*(x10 + x14) - x1*(ddq[1] + x12);
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
    out[8] = -x0*(x15*x30 - x31) - x1*(x19*x30 + x23);
    out[9] = -x0*(x15*x37 - x19*x33) - x1*(x15*x33 + x19*x37);
    out[10] = -x0*(x15*x43 - x19*x40) - x1*(x15*x40 + x19*x43);
    out[11] = -x0*(x15*x45 - x19*x47) - x1*(x15*x47 + x19*x45);
    out[12] = -x0*(x15*x32 + x31) - x1*(x19*x32 - x23);
    out[13] = -x0*(-x19*x49 - x55*x56 - x59*x60) - x1*(0.28000000000000003*x15*x24 + x15*x49 + x18*x63 + x20*x63 + x3*x62 - 0.28000000000000003*x34 - 0.28000000000000003*x35 + x48*x61 - x55*x60 + x56*x59);
    out[14] = -x0*(x15*x64 - x56*x65 - x60*x66) - x1*(0.28000000000000003*dq[0]*dq[1]*x1*x15 + 0.28000000000000003*dq[0]*dq[2]*x1*x15 - 0.28000000000000003*dq[1]*x27 + 0.050000000000000003*x15*x66 + x19*x64 - 0.28000000000000003*x25 - 0.28000000000000003*x26 - x27*x63 - 0.28000000000000003*x29 - x60*x65);
    out[15] = 0;
    out[16] = x12;
    out[17] = x11 - x14;
    out[18] = -x4;
    out[19] = -x6;
    out[20] = ddq[1];
    out[21] = x67;
    out[22] = x68;
    out[23] = x46;
    out[24] = x69;
    out[25] = x70;
    out[26] = x71;
    out[27] = x42;
    out[28] = x55*x61 + x59*x62 + x74;
    out[29] = 0.28000000000000003*x15*x65 + 0.28000000000000003*x19*x66 - x75;
    out[30] = 0;
    out[31] = 0;
    out[32] = 0;
    out[33] = 0;
    out[34] = 0;
    out[35] = 0;
    out[36] = 0;
    out[37] = 0;
    out[38] = x46;
    out[39] = x69;
    out[40] = x70;
    out[41] = x71;
    out[42] = x42;
    out[43] = x74;
    out[44] = -x75;
}

/* 最小参数集 pi_b（用原参数表示）。out: 15x1，行优先。 */
void baseparms(const double *parms, double *out)
{
    const double x0 = (49.0/625.0)*parms[29];
    out[0] = parms[13] + parms[23] + (1.0/10.0)*parms[28] + (809.0/10000.0)*parms[29] + parms[5];
    out[1] = parms[10] - parms[13] - x0;
    out[2] = parms[11];
    out[3] = parms[12] - 7.0/25.0*parms[28] - 7.0/500.0*parms[29];
    out[4] = parms[14];
    out[5] = parms[15] + x0;
    out[6] = parms[16] + (7.0/25.0)*parms[29];
    out[7] = parms[17];
    out[8] = parms[20] - parms[23];
    out[9] = parms[21];
    out[10] = parms[22];
    out[11] = parms[24];
    out[12] = parms[25];
    out[13] = parms[26];
    out[14] = parms[27];
}
