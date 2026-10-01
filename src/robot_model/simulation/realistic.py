"""实机近似的自由空间辨识仿真：对象、名义控制器和测量链分离。

支持固定基座、每关节一个 gear=1 的直接力矩 motor，按关节顺序排列。
不支持浮动基座/欠驱动/位置执行器；这些模型需要另外定义传动映射。
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field

import numpy as np

from ..data.dataset import MotionData
from ..data.filtering import differentiate
from ..utils.validation import integer, vector
from .hardware import (
    ActuatorConfig, OnlineSensorConfig, OnlineSensors, TorqueActuator,
)
from .tracking import _require_mujoco


@dataclass(frozen=True)
class RealismConfig:
    """可复现实验假设，默认量级面向小型 3R 臂，不代表标定后的真机。"""

    seed: int = 0
    physics_dt: float = 0.001
    control_dt: float = 0.004
    kp: float = 20.0
    kd: float = 0.5
    mass_relative_error: float = 0.10
    inertia_relative_error: float = 0.10
    com_error_m: float = 0.003
    damping: float = 0.20
    frictionloss: float = 0.15
    armature: float = 0.04
    # 额外平滑低速摩擦峰：仅用于制造 sign(dq) 模型之外的失配，非完整静摩擦模型。
    low_speed_friction: float = 0.04
    friction_velocity: float = 0.08
    disturbance_amplitude: float = 0.02
    disturbance_frequency_hz: float = 0.7
    joint_lower: tuple | None = None
    joint_upper: tuple | None = None
    velocity_limit: float = 1.5
    acceleration_limit: float = 5.0
    sensors: OnlineSensorConfig = field(default_factory=OnlineSensorConfig)
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "sensors" in values:
            values["sensors"] = OnlineSensorConfig(**values["sensors"])
        if "actuator" in values:
            values["actuator"] = ActuatorConfig(**values["actuator"])
        return cls(**values)


def _validate_motor_model(model):
    mj = _require_mujoco()
    n = model.nv
    if n == 0 or not (model.nq == model.nu == model.njnt == n):
        raise ValueError("需要固定基座、每关节一个直接力矩 motor（nq=nv=nu=njnt）")
    if not np.all(model.jnt_type == mj.mjtJoint.mjJNT_HINGE):
        raise ValueError("当前实机近似模式仅支持转动关节")
    if not (np.array_equal(model.jnt_qposadr, np.arange(n))
            and np.array_equal(model.jnt_dofadr, np.arange(n))
            and np.array_equal(model.actuator_trnid[:, 0], np.arange(n))
            and np.all(model.actuator_trntype == mj.mjtTrn.mjTRN_JOINT)):
        raise ValueError("motor 必须按关节顺序直接驱动各关节")
    expected_gear = np.zeros((n, 6))
    expected_gear[:, 0] = 1
    expected_gain = np.zeros_like(model.actuator_gainprm)
    expected_gain[:, 0] = 1
    if not (np.allclose(model.actuator_gear, expected_gear)
            and np.all(model.actuator_dyntype == mj.mjtDyn.mjDYN_NONE)
            and np.all(model.actuator_gaintype == mj.mjtGain.mjGAIN_FIXED)
            and np.all(model.actuator_biastype == mj.mjtBias.mjBIAS_NONE)
            and np.allclose(model.actuator_gainprm, expected_gain)):
        raise ValueError("需要 gear=1、无内置动态的单位增益力矩 motor")


@dataclass(frozen=True)
class SimulationRun:
    """measured 用于辨识；truth 仅供离线评价，绝不传给控制器/辨识器。"""

    measured: MotionData
    truth: MotionData
    desired_q: np.ndarray
    command: np.ndarray
    diagnostics: dict

    def save(self, path):
        """NPZ 同时保存测量与标明 truth_ 的仿真真值。"""
        payload = {key: getattr(self.measured, key) for key in ("t", "q", "dq", "ddq", "tau")}
        payload.update({f"truth_{key}": getattr(self.truth, key) for key in ("q", "dq", "ddq", "tau")})
        np.savez_compressed(path, **payload, desired_q=self.desired_q, command=self.command)


class RealisticMujocoSource:
    """名义模型前馈 + 测量反馈；物理高频积分，控制/传感低频采样。

    nominal_model 被复制，调用方的模型不变。对象惯性被随机扰动，摩擦、
    armature、限位在对象中启用；前馈只用未扰动的名义模型。
    """

    def __init__(self, nominal_model, config=None):
        self.mujoco = mj = _require_mujoco()
        _validate_motor_model(nominal_model)
        self.config = c = config or RealismConfig()
        self.dof = n = nominal_model.nv
        integer(c.seed, "seed")
        if not np.isfinite([c.physics_dt, c.control_dt]).all() or c.physics_dt <= 0 or c.control_dt <= 0:
            raise ValueError("physics_dt/control_dt 必须为正数")
        ratio = c.control_dt / c.physics_dt
        self.control_steps = int(round(ratio))
        if self.control_steps < 1 or not np.isclose(ratio, self.control_steps, atol=1e-10, rtol=0):
            raise ValueError("control_dt 必须是 physics_dt 的整数倍")
        for key in ("mass_relative_error", "inertia_relative_error"):
            if not 0 <= getattr(c, key) < 1:
                raise ValueError(f"{key} 必须位于 [0,1)")
        if not np.isfinite(c.com_error_m) or c.com_error_m < 0:
            raise ValueError("com_error_m 必须非负")
        self.kp = vector(c.kp, n, "kp", nonnegative=True)
        self.kd = vector(c.kd, n, "kd", nonnegative=True)
        self.low_friction = vector(c.low_speed_friction, n, "low_speed_friction", nonnegative=True)
        self.friction_velocity = vector(c.friction_velocity, n, "friction_velocity", positive=True)
        self.disturbance = vector(c.disturbance_amplitude, n, "disturbance_amplitude", nonnegative=True)
        vector(c.disturbance_frequency_hz, 1, "disturbance_frequency_hz", nonnegative=True)
        self.velocity_limit = vector(c.velocity_limit, n, "velocity_limit", positive=True)
        self.acceleration_limit = vector(c.acceleration_limit, n, "acceleration_limit", positive=True)
        self.nominal_model = deepcopy(nominal_model)
        self.model = deepcopy(nominal_model)
        self.model.opt.timestep = c.physics_dt
        self.model.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
        rng = np.random.default_rng(c.seed)
        # 整体缩放每个惯性张量，保持主惯量正定及三角不等式。
        for bid in range(1, self.model.nbody):
            if self.model.body_mass[bid] <= 0:
                continue
            self.model.body_mass[bid] *= rng.uniform(1-c.mass_relative_error, 1+c.mass_relative_error)
            self.model.body_inertia[bid] *= rng.uniform(1-c.inertia_relative_error, 1+c.inertia_relative_error)
            self.model.body_ipos[bid] += rng.uniform(-c.com_error_m, c.com_error_m, 3)
        self.model.dof_damping[:] = vector(c.damping, n, "damping", nonnegative=True)
        self.model.dof_frictionloss[:] = vector(c.frictionloss, n, "frictionloss", nonnegative=True)
        self.model.dof_armature[:] = vector(c.armature, n, "armature", nonnegative=True)
        if (c.joint_lower is None) != (c.joint_upper is None):
            raise ValueError("joint_lower/upper 必须同时提供")
        if c.joint_lower is not None:
            lo = vector(c.joint_lower, n, "joint_lower")
            hi = vector(c.joint_upper, n, "joint_upper")
            if np.any(hi <= lo):
                raise ValueError("关节上限必须大于下限")
            self.model.jnt_limited[:] = True
            self.model.jnt_range[:] = np.column_stack([lo, hi])
        self.limit = vector(c.actuator.torque_limit, n, "torque_limit", positive=True)
        # 不覆盖源模型更严格的已有力矩/控制范围。
        for limited, ranges in ((self.model.actuator_ctrllimited, self.model.actuator_ctrlrange),
                                (self.model.actuator_forcelimited, self.model.actuator_forcerange)):
            lo = np.where(limited, np.maximum(ranges[:, 0], -self.limit), -self.limit)
            hi = np.where(limited, np.minimum(ranges[:, 1], self.limit), self.limit)
            if np.any(lo >= hi):
                raise ValueError("配置力矩范围与模型执行器范围不相交")
            limited[:] = True
            ranges[:] = np.column_stack([lo, hi])
        mj.mj_setConst(self.model, mj.MjData(self.model))
        self.data = mj.MjData(self.model)
        self._ff = mj.MjData(self.nominal_model)
        # 提前验证硬件配置，避免跑到中途才发现参数无效。
        TorqueActuator(c.actuator, n, c.physics_dt)
        OnlineSensors(c.sensors, n, c.control_dt, np.zeros(n), np.random.default_rng(c.seed))

    def feedforward(self, q, dq, ddq):
        """只用名义模型；与对象真值惯性无关。"""
        self._ff.qpos[:] = q
        self._ff.qvel[:] = dq
        self._ff.qacc[:] = ddq
        self.mujoco.mj_inverse(self.nominal_model, self._ff)
        return self._ff.qfrc_inverse.copy()

    def run(self, trajectory, *, periods=1, warmup_periods=1, seed=None):
        """从首个期望位置静止启动，预热若干周期后记录。

        每个采样时刻先测位置并控制，再读取该时刻的实际执行器力矩；
        mj_forward 后、积分前同步保存真值。返回的 ddq 只从测量 dq 微分。
        延迟保留在测量数据里；不使用仿真真值暗中对齐或去噪。
        """
        integer(periods, "periods", 1)
        integer(warmup_periods, "warmup_periods")
        if trajectory.dof != self.dof or not np.isfinite(trajectory.period) or trajectory.period <= 0:
            raise ValueError("轨迹自由度/周期无效")
        mj, c, d = self.mujoco, self.config, self.data
        run_seed = c.seed if seed is None else integer(seed, "seed")
        mj.mj_resetData(self.model, d)
        mj.mj_resetData(self.nominal_model, self._ff)
        q0, _, _ = trajectory.evaluate(0.0)
        d.qpos[:] = q0
        sensors = OnlineSensors(c.sensors, self.dof, c.control_dt, q0, np.random.default_rng(run_seed))
        actuator = TorqueActuator(c.actuator, self.dof, c.physics_dt)
        warmup = warmup_periods * trajectory.period
        total_steps = int(np.ceil((warmup_periods + periods) * trajectory.period / c.physics_dt))
        command = np.zeros(self.dof)
        rows = []
        saturation = np.zeros(self.dof)
        slew = np.zeros(self.dof)
        limits = np.zeros(self.dof)
        max_speed = np.zeros(self.dof)
        max_acceleration = np.zeros(self.dof)
        contact_steps = 0
        counted = 0
        for k in range(total_steps):
            t = k * c.physics_dt
            tick = k % self.control_steps == 0
            if tick:
                qm, dqm = sensors.feedback(d.qpos)
                qd, dqd, ddqd = trajectory.evaluate(t)
                command = self.feedforward(qd, dqd, ddqd) + self.kp * (qd-qm) + self.kd * (dqd-dqm)
            d.ctrl[:] = actuator.step(command)
            v = d.qvel.copy()
            extra_friction = self.low_friction * np.exp(-(v/self.friction_velocity)**2) * np.tanh(v/0.005)
            disturbance = self.disturbance * np.sin(2*np.pi*c.disturbance_frequency_hz*t + np.arange(self.dof))
            d.qfrc_applied[:] = disturbance - extra_friction
            mj.mj_forward(self.model, d)
            if not np.isfinite(np.concatenate([d.qpos, d.qvel, d.qacc])).all():
                raise RuntimeError(f"仿真数值无效，t={t:.6f}")
            if tick:
                taum = sensors.torque(d.qfrc_actuator.copy())
                if t >= warmup - 1e-12:
                    rows.append((t, qm.copy(), dqm.copy(), taum, d.qpos.copy(),
                                 d.qvel.copy(), d.qacc.copy(), d.qfrc_actuator.copy(),
                                 qd.copy(), command.copy()))
            if t >= warmup - 1e-12:
                counted += 1
                saturation += actuator.saturated | (np.abs(d.ctrl-d.qfrc_actuator) > 1e-9)
                slew += actuator.slew_limited
                limits += self.model.jnt_limited.astype(bool) & (
                    (d.qpos <= self.model.jnt_range[:, 0] + self.model.jnt_margin)
                    | (d.qpos >= self.model.jnt_range[:, 1] - self.model.jnt_margin))
                max_speed = np.maximum(max_speed, np.abs(d.qvel))
                max_acceleration = np.maximum(max_acceleration, np.abs(d.qacc))
                contact_steps += d.ncon > 0
            mj.mj_step(self.model, d)
            if not np.isclose(d.time, (k+1)*c.physics_dt, atol=1e-8, rtol=0) or np.any(d.warning.number):
                raise RuntimeError(f"MuJoCo 出现警告或自动重置，t={t:.6f}；请降低增益/激励或减小步长")
        if len(rows) < 3:
            raise ValueError("采样点不足，至少需要 3 个控制周期")
        t, q, dq, tau, qt, dqt, ddqt, taut, desired, commands = map(np.asarray, zip(*rows))
        measured = MotionData(t, q, dq, differentiate(dq, c.control_dt), tau)
        truth = MotionData(t, qt, dqt, ddqt, taut)
        diagnostics = {
            "seed": run_seed,
            "samples": len(t),
            "tracking_rms_rad": np.sqrt(np.mean((qt-desired)**2, axis=0)).tolist(),
            "tracking_max_rad": np.max(np.abs(qt-desired), axis=0).tolist(),
            "saturation_fraction": (saturation/counted).tolist(),
            "slew_limit_fraction": (slew/counted).tolist(),
            "joint_limit_fraction": (limits/counted).tolist(),
            "contact_fraction": contact_steps/counted,
            "max_speed_rad_s": max_speed.tolist(),
            "max_acceleration_rad_s2": max_acceleration.tolist(),
            "speed_limit_exceeded": (max_speed > self.velocity_limit).tolist(),
            "acceleration_limit_exceeded": (max_acceleration > self.acceleration_limit).tolist(),
        }
        return SimulationRun(measured, truth, desired, commands, diagnostics)
