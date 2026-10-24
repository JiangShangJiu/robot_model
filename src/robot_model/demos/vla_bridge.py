"""演示 3：VLA → 底层控制桥接（仿真桩）——延迟/丢包下护栏的价值。

"VLA 桩"不是真模型：它以低频（2 Hz）输出动作块（8 个 0.1 s 间隔的末端
位置目标），目标从当前观测收敛到参考轨迹——这正是 ACT/π0 一类分块策略
的接口形态。桥接层以 1 kHz 把动作块变成阻抗环的设定点，并可叠加：

* 推理+通信延迟（动作块按过期时间戳到达）
* 随机丢块 + 一段强制"模型沉默"窗口
* 护栏：设定点速度钳位、工作空间盒钳位、超时保持（hold-on-timeout）

三种场景对照：理想链路 / 退化链路（裸桥接）/ 退化链路 + 护栏。
结论不是"护栏恢复了跟踪精度"，而是"护栏把退化变成有界、安全的退化"。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .common import ensure_output_dir, rms, rounded, save_metrics, use_headless_matplotlib
from .task_space import ImpedanceGains, TaskSpaceArm, clamp_torque, impedance_force

HOME_Q = np.array([0.0, -0.6, 1.2])


def freespace_model_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "assets" / "models" / "rrr_arm" / "rrr_arm.xml"


@dataclass(frozen=True)
class VlaBridgeDemoConfig:
    seed: int = 0
    dt: float = 0.001
    duration_s: float = 20.0
    # 参考路径：y-z 平面内的圆（法向 x），周期 8 s
    circle_center: tuple = (0.40, 0.05, 0.30)
    circle_radius_m: float = 0.06
    circle_period_s: float = 8.0
    # VLA 桩：低频动作块
    chunk_period_s: float = 0.5
    action_dt_s: float = 0.1
    chunk_len: int = 8
    convergence_decay: float = 0.5   # 动作块从观测位置向参考收敛的衰减率
    # 链路退化
    latency_s: float = 0.25
    dropout_prob: float = 0.15
    silence_window_s: tuple = (12.0, 14.5)
    # 护栏
    setpoint_speed_max_m_s: float = 0.25
    workspace_lower: tuple = (0.30, -0.10, 0.15)
    workspace_upper: tuple = (0.52, 0.20, 0.45)
    # 必须大于 chunk_period + latency，否则健康链路也会触发保持
    hold_timeout_s: float = 0.8
    # 底层阻抗环
    gains: ImpedanceGains = ImpedanceGains()
    torque_limit: tuple = (15.0, 20.0, 8.0)
    eval_start_s: float = 2.0

    def to_dict(self):
        values = asdict(self)
        values["gains"] = {"kp": list(self.gains.kp), "kd": list(self.gains.kd)}
        return values


class CirclePath:
    """参考轨迹：工作空间内一个匀速圆。"""

    def __init__(self, config):
        self.center = np.asarray(config.circle_center, float)
        self.radius = config.circle_radius_m
        self.omega = 2 * np.pi / config.circle_period_s

    def position(self, t):
        phase = self.omega * np.asarray(t, float)
        return self.center + self.radius * np.stack(
            [np.zeros_like(phase), np.cos(phase) - 1.0, np.sin(phase)], axis=-1)


class VlaStub:
    """动作块桩：观测当前末端位置，输出向参考收敛的未来目标序列。

    不含任何学习成分；它只定义接口与时序，供桥接层压力测试。
    """

    def __init__(self, config, path):
        self.config = config
        self.path = path

    def infer(self, t_now, x_observed):
        """返回 [(目标应执行的时刻, 末端目标位置), ...]。"""
        c = self.config
        chunk = []
        offset = x_observed - self.path.position(t_now)
        for j in range(1, c.chunk_len + 1):
            t_exec = t_now + j * c.action_dt_s
            target = self.path.position(t_exec) + offset * c.convergence_decay ** j
            chunk.append((t_exec, target))
        return chunk


class Bridge:
    """把异步动作块转成 1 kHz 阻抗设定点；护栏可开关。

    naive 模式：按时间戳线性插值，块耗尽后沿最后速度外推（乐观执行器）。
    guarded 模式：超时保持 + 设定点速度钳位 + 工作空间盒钳位。
    """

    def __init__(self, config, x_init, *, guarded):
        self.config = config
        self.guarded = guarded
        self.waypoints = []           # [(t_exec, target)]，时间递增
        self.last_data_time = 0.0
        self.setpoint = np.asarray(x_init, float).copy()
        self.velocity = np.zeros(3)
        self.holding = False
        self.clamp_events = 0
        self.hold_steps = 0

    def push_chunk(self, t_now, chunk):
        self.waypoints = [(t, target) for t, target in chunk if t > t_now]
        self.last_data_time = t_now

    def _raw_target(self, t):
        """按时间戳插值动作块；块耗尽时 naive 外推、guarded 报告无数据。"""
        while len(self.waypoints) > 1 and self.waypoints[1][0] <= t:
            self.waypoints.pop(0)
        if not self.waypoints:
            return None
        t0, x0 = self.waypoints[0]
        if t <= t0:
            # 下一个目标还在未来：向它匀速推进
            span = max(t0 - t, self.config.action_dt_s * 0.1)
            direction = (x0 - self.setpoint) / span
            return self.setpoint + direction * self.config.dt
        if len(self.waypoints) >= 2:
            t1, x1 = self.waypoints[1]
            ratio = np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
            return x0 + ratio * (x1 - x0)
        # 只剩最后一个目标：naive 沿最近速度外推（这正是风险来源）
        if self.guarded:
            return None
        return self.setpoint + self.velocity * self.config.dt

    def update(self, t):
        """返回 (x_d, dx_d)。"""
        c = self.config
        target = self._raw_target(t)
        stale = (t - self.last_data_time) > c.hold_timeout_s
        self.holding = self.guarded and (target is None or stale)
        if self.holding:
            # 超时保持：冻结设定点，速度归零，等待新数据
            self.hold_steps += 1
            self.velocity = np.zeros(3)
            return self.setpoint.copy(), np.zeros(3)
        if target is None:  # naive 且完全无数据：维持外推
            target = self.setpoint + self.velocity * c.dt
        step = np.asarray(target, float) - self.setpoint
        if self.guarded:
            max_step = c.setpoint_speed_max_m_s * c.dt
            norm = float(np.linalg.norm(step))
            if norm > max_step:
                step *= max_step / norm
                self.clamp_events += 1
            new_setpoint = np.clip(self.setpoint + step,
                                   np.asarray(c.workspace_lower), np.asarray(c.workspace_upper))
        else:
            new_setpoint = self.setpoint + step
        self.velocity = (new_setpoint - self.setpoint) / c.dt
        self.setpoint = new_setpoint
        return self.setpoint.copy(), self.velocity.copy()


def run_scenario(config: VlaBridgeDemoConfig, *, degraded, guarded) -> dict:
    """跑一种链路/护栏组合；三种场景共用种子与参考路径。"""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(freespace_model_path()))
    arm = TaskSpaceArm(model)
    arm.reset(HOME_Q)
    path = CirclePath(config)
    stub = VlaStub(config, path)
    rng = np.random.default_rng(config.seed + 909)

    x0 = arm.tool_state()[0]
    bridge = Bridge(config, x0, guarded=guarded)
    steps = int(round(config.duration_s / config.dt))
    chunk_every = int(round(config.chunk_period_s / config.dt))
    latency_steps = int(round((config.latency_s if degraded else 0.0) / config.dt))
    silence = config.silence_window_s

    pending = []  # [(到达步, chunk)]
    log = {key: np.zeros(steps) for key in ("t", "error", "speed", "holding", "data_age")}
    log["x"] = np.zeros((steps, 3))
    log["x_ref"] = np.zeros((steps, 3))
    dropped = 0
    for k in range(steps):
        t = k * config.dt
        x, dx, jacobian = arm.tool_state()

        if k % chunk_every == 0:
            in_silence = degraded and silence[0] <= t < silence[1]
            lost = degraded and rng.uniform() < config.dropout_prob
            if not in_silence and not lost:
                pending.append((k + latency_steps, stub.infer(t, x.copy())))
            else:
                dropped += 1
        while pending and pending[0][0] <= k:
            _, chunk = pending.pop(0)
            bridge.push_chunk(t, chunk)

        x_d, dx_d = bridge.update(t)
        force = impedance_force(x, dx, x_d, dx_d, config.gains)
        tau = clamp_torque(jacobian.T @ force + arm.gravity_torque(arm.data.qpos),
                           config.torque_limit)
        arm.step(tau)

        log["t"][k] = t
        log["x"][k] = x
        log["x_ref"][k] = path.position(t)
        log["error"][k] = float(np.linalg.norm(x - log["x_ref"][k]))
        log["speed"][k] = float(np.linalg.norm(dx))
        log["holding"][k] = float(bridge.holding)
        log["data_age"][k] = t - bridge.last_data_time
    log["dropped_chunks"] = dropped
    log["clamp_events"] = bridge.clamp_events
    return log


def _scenario_metrics(config, log) -> dict:
    mask = log["t"] >= config.eval_start_s
    inside = np.all((log["x"] >= np.asarray(config.workspace_lower)) &
                    (log["x"] <= np.asarray(config.workspace_upper)), axis=1)
    return {
        "tracking_rmse_m": float(rms(log["error"][mask])),
        "tracking_max_m": float(np.max(log["error"][mask])),
        "ee_speed_max_m_s": float(np.max(log["speed"])),
        "dropped_chunks": int(log["dropped_chunks"]),
        "hold_fraction": float(np.mean(log["holding"][mask])),
        "speed_clamp_events": int(log["clamp_events"]),
        "outside_workspace_fraction": float(np.mean(~inside[mask])),
        "max_data_age_s": float(np.max(log["data_age"])),
    }


def run_demo(output_dir, config: VlaBridgeDemoConfig | None = None) -> dict:
    config = config or VlaBridgeDemoConfig()
    out = ensure_output_dir(output_dir)
    scenarios = {}
    for name, degraded, guarded in (("ideal_link", False, False),
                                    ("degraded_naive", True, False),
                                    ("degraded_guarded", True, True)):
        print(f"VLA 桥接场景：{name} …", flush=True)
        scenarios[name] = run_scenario(config, degraded=degraded, guarded=guarded)

    metrics = {
        "demo": "vla_low_level_bridge",
        "settings": config.to_dict(),
        "scenarios": {name: _scenario_metrics(config, log) for name, log in scenarios.items()},
        "notes": [
            "VLA 桩只定义动作块接口与时序（2 Hz、8 步块、观测收敛），无学习成分。",
            "退化链路：250 ms 延迟 + 15% 随机丢块 + 12–14.5 s 强制沉默。",
            "护栏不恢复精度——它把无界漂移变成有界保持，并限制设定点速度与工作空间。",
        ],
    }
    save_metrics(out / "metrics.json", rounded(metrics))
    _plot_paths(out, config, scenarios)
    _plot_timeline(out, config, scenarios, metrics)
    print(f"产物已写入 {out.resolve()}")
    return metrics


_COLORS = {"ideal_link": "#55a868", "degraded_naive": "#c44e52", "degraded_guarded": "#4c72b0"}
_LABELS = {"ideal_link": "ideal link", "degraded_naive": "degraded, naive bridge",
           "degraded_guarded": "degraded, guarded bridge"}


def _plot_paths(out, config, scenarios):
    plt = use_headless_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharex=True, sharey=True)
    reference = scenarios["ideal_link"]["x_ref"]
    for axis, (name, log) in zip(axes, scenarios.items()):
        axis.plot(reference[:, 1], reference[:, 2], lw=1.2, color="0.35", ls="--",
                  label="reference circle")
        axis.plot(log["x"][:, 1], log["x"][:, 2], lw=0.9, color=_COLORS[name])
        axis.set_title(_LABELS[name], fontsize=10)
        axis.set_xlabel("y [m]")
        axis.set_aspect("equal")
    axes[0].set_ylabel("z [m]")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("End-effector path (y-z plane): action chunks at 2 Hz driving a 1 kHz impedance loop")
    fig.tight_layout()
    fig.savefig(out / "paths.png", dpi=130)
    plt.close(fig)


def _plot_timeline(out, config, scenarios, metrics):
    plt = use_headless_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True)
    silence = config.silence_window_s
    for name, log in scenarios.items():
        axes[0].plot(log["t"], 1e3 * log["error"], lw=0.9, color=_COLORS[name],
                     label=f"{_LABELS[name]} (RMSE {1e3 * metrics['scenarios'][name]['tracking_rmse_m']:.1f} mm)")
        axes[1].plot(log["t"], log["speed"], lw=0.9, color=_COLORS[name])
    guarded = scenarios["degraded_guarded"]
    hold = guarded["holding"] > 0
    axes[0].fill_between(guarded["t"], 0, 1, where=hold, transform=axes[0].get_xaxis_transform(),
                         color="#4c72b0", alpha=0.12, label="guarded bridge holding")
    for axis in axes:
        axis.axvspan(*silence, color="0.85", zorder=0)
    axes[0].text(0.5 * (silence[0] + silence[1]), 0.96, "forced VLA silence",
                 transform=axes[0].get_xaxis_transform(), ha="center", va="top", fontsize=8)
    axes[0].set_ylabel("tracking error [mm]")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].axhline(config.setpoint_speed_max_m_s, color="0.3", lw=0.9, ls="--",
                    label="guarded setpoint speed limit")
    axes[1].set_ylabel("EE speed [m/s]")
    axes[1].set_xlabel("t [s]")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[0].set_title("Latency 250 ms + 15% chunk dropout + forced silence: naive drifts, guardrails bound the damage")
    fig.tight_layout()
    fig.savefig(out / "timeline.png", dpi=130)
    plt.close(fig)
