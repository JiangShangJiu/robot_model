"""演示共享的任务空间（末端位置）阻抗控制原语。

只处理 3 自由度位置子任务（RRR 的 J 为 3x3 方阵）；姿态不在演示范围内。
重力补偿使用名义模型 ``mj_inverse``，不读取对象侧的接触或摩擦信息。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImpedanceGains:
    """任务空间刚度/阻尼（N/m、N·s/m），各轴独立。"""

    kp: tuple = (800.0, 800.0, 800.0)
    kd: tuple = (80.0, 80.0, 80.0)

    def arrays(self):
        return np.asarray(self.kp, float), np.asarray(self.kd, float)


class TaskSpaceArm:
    """包一层 MuJoCo：工具位置/速度、位置雅可比与名义重力补偿。"""

    def __init__(self, model, site_name="tool"):
        import mujoco
        from copy import deepcopy

        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        # 重力补偿必须在"无接触"的模型副本上算：mj_inverse 的结果包含
        # 当前状态下的约束力，直接用接触场景会把墙的反力也当作需要
        # 补偿的项，等于主动往墙里压。
        self._gravity_model = deepcopy(model)
        self._gravity_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self._gravity_data = mujoco.MjData(self._gravity_model)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"模型缺少 site {site_name!r}")
        self.dof = model.nv
        self._jacp = np.zeros((3, self.dof))
        self._jacr = np.zeros((3, self.dof))

    def reset(self, q0):
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.asarray(q0, float)
        self.mujoco.mj_forward(self.model, self.data)

    def tool_state(self):
        """当前工具位置与速度（世界系）；调用前须已 mj_forward。"""
        x = self.data.site_xpos[self.site_id].copy()
        self.mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        dx = self._jacp @ self.data.qvel
        return x, dx, self._jacp.copy()

    def gravity_torque(self, q):
        """名义重力 + 科氏为零的静态补偿项 g(q)。"""
        d = self._gravity_data
        d.qpos[:] = q
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        self.mujoco.mj_inverse(self._gravity_model, d)
        return d.qfrc_inverse.copy()

    def step(self, tau):
        self.data.ctrl[:] = tau
        self.mujoco.mj_step(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)


def impedance_force(x, dx, x_d, dx_d, gains: ImpedanceGains):
    """任务空间 PD 虚拟力 F = Kp (x_d - x) + Kd (dx_d - dx)。"""
    kp, kd = gains.arrays()
    return kp * (np.asarray(x_d) - x) + kd * (np.asarray(dx_d) - dx)


def clamp_torque(tau, limit):
    return np.clip(tau, -np.asarray(limit, float), np.asarray(limit, float))
