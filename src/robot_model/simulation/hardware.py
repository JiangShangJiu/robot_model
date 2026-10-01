"""因果的采样测量链与关节侧力矩执行器；不依赖 MuJoCo。

参数是实验假设，必须用实机日志/规格书校准后才能代表具体设备。
延迟单位为各自调用周期的整数步；所有状态在每次实验重新创建。
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


from ..utils.validation import integer, vector


class DelayLine:
    """延迟 d 步：第 k 次返回第 k-d 次输入；启动时保持 initial。"""

    def __init__(self, steps, initial):
        steps = integer(steps, "delay")
        self.values = deque([np.array(initial, float).copy() for _ in range(steps)])

    def push(self, value):
        self.values.append(np.array(value, float).copy())
        return self.values.popleft()


@dataclass(frozen=True)
class OnlineSensorConfig:
    """位置 rad、速度 rad/s、力矩 N·m；延迟以控制/采样周期计。"""

    encoder_bits: int | None = 17
    encoder_span: float = 2 * np.pi
    position_noise_std: float = 1e-5
    position_bias: float = 0.0002
    encoder_delay_steps: int = 1
    velocity_cutoff_hz: float = 30.0
    torque_noise_std: float = 0.02
    torque_gain: float = 1.01
    torque_bias: float = 0.01
    torque_deadband: float = 0.005
    torque_delay_steps: int = 2


class OnlineSensors:
    """位置量化后向后差分测速，再做因果一阶低通；绝不读取真实速度。"""

    def __init__(self, config, dof, dt, q_initial, rng):
        self.config, self.dt, self.rng = config, float(dt), rng
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("采样周期必须为正")
        if config.encoder_bits is not None:
            integer(config.encoder_bits, "encoder_bits", 1)
            if config.encoder_bits > 52:
                raise ValueError("encoder_bits 必须 <= 52")
        self.span = vector(config.encoder_span, 1, "encoder_span", positive=True)[0]
        self.q_std = vector(config.position_noise_std, dof, "position_noise_std", nonnegative=True)
        self.q_bias = vector(config.position_bias, dof, "position_bias")
        self.t_std = vector(config.torque_noise_std, dof, "torque_noise_std", nonnegative=True)
        self.t_gain = vector(config.torque_gain, dof, "torque_gain", positive=True)
        self.t_bias = vector(config.torque_bias, dof, "torque_bias")
        self.deadband = vector(config.torque_deadband, dof, "torque_deadband", nonnegative=True)
        cutoff = float(config.velocity_cutoff_hz)
        if not 0 < cutoff < 0.5 / dt:
            raise ValueError("velocity_cutoff_hz 必须在 (0, Nyquist) 内")
        self.alpha = -np.expm1(-2 * np.pi * cutoff * dt)
        initial = self._position(q_initial)
        self.q_delay = DelayLine(config.encoder_delay_steps, initial)
        self.t_delay = DelayLine(config.torque_delay_steps, np.zeros(dof))
        self.previous = initial
        self.velocity = np.zeros(dof)

    def _position(self, q):
        q = np.asarray(q, float) + self.q_bias + self.rng.normal(size=self.q_std.size) * self.q_std
        if self.config.encoder_bits is not None:
            lsb = self.span / 2**self.config.encoder_bits
            q = np.round(q / lsb) * lsb
        return q

    def feedback(self, q_true):
        q = self.q_delay.push(self._position(q_true))
        derivative = (q - self.previous) / self.dt
        self.velocity += self.alpha * (derivative - self.velocity)
        self.previous = q.copy()
        return q.copy(), self.velocity.copy()

    def torque(self, applied):
        value = self.t_gain * applied + self.t_bias
        value = value + self.rng.normal(size=self.t_std.size) * self.t_std
        value = np.where(np.abs(value) < self.deadband, 0.0, value)
        return self.t_delay.push(value)


@dataclass(frozen=True)
class ActuatorConfig:
    """关节侧近似：通信延迟 + 力矩上限 + 一阶响应 + 力矩变化率限制。

    delay_steps 以物理步长计；time_constant_s 为秒；slew_rate 为 N·m/s。
    这不是完整电气/齿轮模型，armature 在 MuJoCo 对象侧另外设置。
    """

    torque_limit: float = 20.0
    time_constant_s: float = 0.008
    slew_rate: float = 400.0
    gain: float = 0.99
    delay_steps: int = 2


class TorqueActuator:
    def __init__(self, config, dof, dt):
        self.dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("物理步长必须为正")
        self.limit = vector(config.torque_limit, dof, "torque_limit", positive=True)
        self.tc = vector(config.time_constant_s, dof, "time_constant_s", nonnegative=True)
        self.slew = vector(config.slew_rate, dof, "slew_rate", positive=True)
        self.gain = vector(config.gain, dof, "gain", positive=True)
        self.delay = DelayLine(config.delay_steps, np.zeros(dof))
        self.output = np.zeros(dof)
        self.alpha = np.ones(dof)
        mask = self.tc > 0
        self.alpha[mask] = -np.expm1(-dt / self.tc[mask])
        self.saturated = np.zeros(dof, dtype=bool)
        self.slew_limited = np.zeros(dof, dtype=bool)

    def step(self, command):
        requested = self.gain * self.delay.push(command)
        self.saturated = np.abs(requested) > self.limit
        target = np.clip(requested, -self.limit, self.limit)
        delta = self.alpha * (target - self.output)
        self.slew_limited = np.abs(delta) > self.slew * self.dt
        self.output += np.clip(delta, -self.slew * self.dt, self.slew * self.dt)
        self.output = np.clip(self.output, -self.limit, self.limit)
        return self.output.copy()
