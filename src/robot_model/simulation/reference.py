"""把 MuJoCo 模型当作动力学真值：惯性参数提取 + mj_inverse 力矩。

用途是给符号推导找一个**外部**参照。库内的 Park / Khalil / 拉格朗日互相对拍
只能发现"实现之间不一致"，发现不了"几套实现按同一个错误理解写成"——
参数约定、惯性张量参考点、重力符号这类错误会一致地错。

前提：MJCF 里 body 帧与 MDH 帧逐帧对齐，且关节为裸刚体
（armature / damping / frictionloss 均为 0、无接触、无限位），
否则 ``mj_inverse`` 的结果会混入本库不建模的项。
``rrr_arm.xml`` 本来就这么写；真实机型的 MJCF（如 Panda）带电机惯量和阻尼，
用 :func:`strip_to_rigid_body` 就地退化即可。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracking import _require_mujoco


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    mujoco = _require_mujoco()

    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def strip_to_rigid_body(model):
    """就地去掉 MJCF 里本库不建模的项，让 ``mj_inverse`` 只剩纯刚体力矩。

    面向真实机型的 MJCF：它们通常带 ``armature``（电机惯量）、``damping``
    与 ``frictionloss``，还有碰撞几何和关节限位。这些都会进 ``qfrc_inverse``，
    使外部真值不再是 ``M ddq + C dq + g``。清零 + 关约束后才能与符号模型对拍。

    也可以反过来——让 ``Robot`` 带上 ``frictionmodel`` / ``driveinertiamodel``
    去匹配 MJCF，但那样比对的就不是刚体动力学本身了。
    """
    mujoco = _require_mujoco()

    model.dof_armature[:] = 0.0
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0
    model.opt.disableflags |= (
        mujoco.mjtDisableBit.mjDSBL_CONTACT
        | mujoco.mjtDisableBit.mjDSBL_LIMIT
    )
    return model


@dataclass(frozen=True)
class LinkInertia:
    """单连杆的 barycentric 参数（均在该连杆自身坐标系下）。"""

    m: float
    r: np.ndarray  # 质心位置
    l: np.ndarray  # 一阶矩 m*r
    I: np.ndarray  # 绕质心的惯性张量
    L: np.ndarray  # 绕连杆系原点的惯性张量

    @property
    def Le(self) -> np.ndarray:
        """``L`` 的 6 分量形式，顺序同 ``RobotSymbols.Le``。"""
        return np.array(
            [
                self.L[0, 0],
                self.L[0, 1],
                self.L[0, 2],
                self.L[1, 1],
                self.L[1, 2],
                self.L[2, 2],
            ]
        )


class MujocoReference:
    """MuJoCo 模型的动力学真值视图。"""

    def __init__(self, model, body_names):
        mujoco = _require_mujoco()

        self.model = model
        self.data = mujoco.MjData(model)
        self.body_names = list(body_names)
        self.body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in self.body_names
        ]
        if any(i < 0 for i in self.body_ids):
            missing = [
                n for n, i in zip(self.body_names, self.body_ids) if i < 0
            ]
            raise ValueError(f"模型中找不到 body: {missing}")
        self.links = [self._read_link(i) for i in self.body_ids]

    def _read_link(self, bid: int) -> LinkInertia:
        m = float(self.model.body_mass[bid])
        r = np.array(self.model.body_ipos[bid], dtype=float)
        # body_inertia 是主轴惯性，body_iquat 给出主轴相对 body 系的朝向
        R = _quat_to_mat(self.model.body_iquat[bid])
        I = R @ np.diag(np.asarray(self.model.body_inertia[bid], float)) @ R.T
        sk = _skew(r)
        return LinkInertia(m=m, r=r, l=m * r, I=I, L=I + m * sk.T @ sk)

    @property
    def gravity(self) -> np.ndarray:
        return np.array(self.model.opt.gravity, dtype=float)

    def inertia_args(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(m, l, Le)``，形状分别为 ``(n,)``、``(n,3)``、``(n,6)``。"""
        return (
            np.array([lk.m for lk in self.links]),
            np.array([lk.l for lk in self.links]),
            np.array([lk.Le for lk in self.links]),
        )

    def dynparms(self) -> np.ndarray:
        """按 Khalil 顺序展平：每连杆 ``[Le(6), l(3), m]``。"""
        return np.concatenate(
            [np.concatenate([lk.Le, lk.l, [lk.m]]) for lk in self.links]
        )

    def inverse_dynamics(self, q, dq, ddq) -> np.ndarray:
        """``mj_inverse`` 给出的关节力矩真值。"""
        mujoco = _require_mujoco()

        d = self.data
        d.qpos[:] = np.asarray(q, float)
        d.qvel[:] = np.asarray(dq, float)
        d.qacc[:] = np.asarray(ddq, float)
        mujoco.mj_inverse(self.model, d)
        return np.array(d.qfrc_inverse, dtype=float)

    def sensor(self, name: str) -> np.ndarray:
        """按名字读传感器；需先调用 ``body_poses`` / ``inverse_dynamics`` 更新状态。"""
        mujoco = _require_mujoco()

        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise ValueError(f"模型中找不到 sensor: {name!r}")
        adr = self.model.sensor_adr[sid]
        return np.array(self.data.sensordata[adr : adr + self.model.sensor_dim[sid]])

    def forward(self, q, dq=None) -> None:
        """设定状态并跑一次前向计算，之后可读位姿与传感器。"""
        mujoco = _require_mujoco()

        self.data.qpos[:] = np.asarray(q, float)
        self.data.qvel[:] = 0.0 if dq is None else np.asarray(dq, float)
        mujoco.mj_forward(self.model, self.data)

    def body_poses(self, q, dq=None) -> list[np.ndarray]:
        """各 body 的 4×4 世界位姿，用于运动学对拍。"""
        self.forward(q, dq)
        d = self.data
        out = []
        for bid in self.body_ids:
            T = np.eye(4)
            T[:3, :3] = d.xmat[bid].reshape(3, 3)
            T[:3, 3] = d.xpos[bid]
            out.append(T)
        return out
