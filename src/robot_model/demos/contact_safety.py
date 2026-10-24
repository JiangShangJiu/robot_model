"""演示 2：接触力控失稳回放——能量罐护栏前后的同一任务对照。

任务：RRR 工具端压在 x=0.46 的刚性墙上，跟踪 15 N 法向力并做小幅横向
"抛光"滑动。外环是把力误差积分进法向位置设定点的导纳式力跟踪；
力测量含延迟、噪声和低通。延迟 + 高积分增益 + 硬接触使无护栏版本
进入接触弹跳极限环；能量罐限制设定点运动可注入的能量，保持稳定。

能量账本（离散步）：
  弹簧储能 V = 1/2 (x_d-x)^T Kp (x_d-x)；设定点移动注入 P_act = F_spring · dx_d。
  罐 E 以阻尼耗散补充、被 P_act 消耗；E 低于下限时冻结法向设定点运动，
  横向滑动与阻抗内环不受影响。这是时域无源性的工程化近似，不是严格证明。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..simulation.hardware import DelayLine
from .common import ensure_output_dir, rms, rounded, save_metrics, use_headless_matplotlib
from .task_space import ImpedanceGains, TaskSpaceArm, clamp_torque, impedance_force

WALL_X = 0.46        # 与 rrr_contact_scene.xml 的墙面一致
TOOL_RADIUS = 0.012  # 工具球半径：球心在 WALL_X - TOOL_RADIUS 处表面接触
# 接近段目标：轻触（约 2 mm 虚拟穿透），力控段再由积分器建立目标压力
CONTACT_POINT = np.array([WALL_X - TOOL_RADIUS + 0.002, 0.05, 0.30])
HOME_Q = np.array([0.0, -0.6, 1.2])


def contact_scene_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "assets" / "models" / "rrr_arm" / "rrr_contact_scene.xml"


@dataclass(frozen=True)
class ContactDemoConfig:
    """失稳场景是"激进调参"假设：高积分增益 + 20 ms 力延迟 + 硬接触。"""

    seed: int = 0
    dt: float = 0.001
    duration_s: float = 12.0
    approach_end_s: float = 2.0
    eval_start_s: float = 8.0
    force_target_N: float = 15.0
    force_delay_steps: int = 30
    force_noise_std_N: float = 0.3
    force_filter_hz: float = 60.0
    ki_force: float = 0.04           # 导纳积分增益 m/(N·s)，故意调高
    normal_offset_max_m: float = 0.05
    lateral_amplitude_m: float = 0.02
    lateral_freq_hz: float = 0.2
    gains: ImpedanceGains = ImpedanceGains()
    torque_limit: tuple = (15.0, 20.0, 8.0)
    # 罐初值即首个失稳瞬态的能量预算：越小护栏介入越早、初始峰值越低
    tank_init_J: float = 0.07
    tank_max_J: float = 0.6
    tank_min_J: float = 0.05
    # 阻尼耗散只按小比例回充：全额回充会在受迫振荡时掩盖净注入，
    # 护栏将永远不触发（振荡时耗散本身就大）。
    tank_refill_fraction: float = 0.05
    # 小额恒定回充 = 允许的稳态活动预算：罐耗尽后力跟踪能缓慢恢复，
    # 而不是永久冻结在失稳瞬间的设定点上。
    tank_trickle_W: float = 0.08

    def to_dict(self):
        values = asdict(self)
        values["gains"] = {"kp": list(self.gains.kp), "kd": list(self.gains.kd)}
        return values


def _desired_lateral(config, t):
    """接触后的横向滑动（y 方向正弦），模拟抛光式运动。"""
    if t <= config.approach_end_s:
        return 0.0, 0.0
    phase = 2 * np.pi * config.lateral_freq_hz * (t - config.approach_end_s)
    amp = config.lateral_amplitude_m
    return amp * np.sin(phase), amp * 2 * np.pi * config.lateral_freq_hz * np.cos(phase)


def run_case(config: ContactDemoConfig, *, use_tank: bool) -> dict:
    """跑一种护栏设置，返回时序与事件计数；两种设置共用对象与噪声种子。"""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(contact_scene_path()))
    arm = TaskSpaceArm(model)
    arm.reset(HOME_Q)
    rng = np.random.default_rng(config.seed)
    tool_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tool_tip")

    steps = int(round(config.duration_s / config.dt))
    start = arm.tool_state()[0]
    delay = DelayLine(config.force_delay_steps, np.zeros(1))
    alpha = -np.expm1(-2 * np.pi * config.force_filter_hz * config.dt)
    force_filtered = 0.0
    normal_offset = 0.0
    tank = config.tank_init_J
    kp, kd = config.gains.arrays()

    log = {key: np.zeros(steps) for key in
           ("t", "force_true", "force_meas", "normal_pos", "setpoint_x",
            "tank", "scale", "contact")}
    tau_log = np.zeros((steps, arm.dof))
    for k in range(steps):
        t = k * config.dt
        x, dx, jacobian = arm.tool_state()

        # 法向真实接触力：对工具球上的所有接触求和（墙压工具为正）。
        force_true = _tool_normal_force(arm, tool_geom)
        noisy = force_true + rng.normal(0.0, config.force_noise_std_N)
        delayed = delay.push(np.array([noisy]))[0]
        force_filtered += alpha * (delayed - force_filtered)

        # 设定点：接近段直线插补到墙面；力控段 = 墙面 + 法向偏置 + 横向滑动
        if t < config.approach_end_s:
            ratio = t / config.approach_end_s
            x_d = start + ratio * (CONTACT_POINT - start)
            dx_d = (CONTACT_POINT - start) / config.approach_end_s
            offset_rate = 0.0
        else:
            offset_rate = config.ki_force * (config.force_target_N - force_filtered)
            lateral, lateral_rate = _desired_lateral(config, t)
            x_d = CONTACT_POINT + np.array([normal_offset, lateral, 0.0])
            dx_d = np.array([offset_rate, lateral_rate, 0.0])

        spring = kp * (x_d - x)
        scale = 1.0
        if use_tank and t >= config.approach_end_s:
            # 罐只核算法向设定点通道：横向滑动是预先验证过的名义任务，
            # 力跟踪积分器才是会被延迟激励成主动源的部分。
            injected = float(spring[0] * offset_rate)
            dissipated = float(dx @ (kd * dx))      # 阻尼耗散，按比例回充
            tank = min(config.tank_max_J,
                       tank + config.dt * (config.tank_trickle_W
                                           + config.tank_refill_fraction * dissipated))
            if injected > 0:
                available = max(0.0, tank - config.tank_min_J)
                scale = min(1.0, available / (injected * config.dt + 1e-12))
                tank -= scale * injected * config.dt
            offset_rate *= scale
            dx_d = np.array([offset_rate, dx_d[1], dx_d[2]])

        normal_offset = float(np.clip(normal_offset + offset_rate * config.dt,
                                      -config.normal_offset_max_m,
                                      config.normal_offset_max_m))
        force = impedance_force(x, dx, x_d, dx_d, config.gains)
        tau = clamp_torque(jacobian.T @ force + arm.gravity_torque(arm.data.qpos),
                           config.torque_limit)
        arm.step(tau)

        log["t"][k] = t
        log["force_true"][k] = force_true
        log["force_meas"][k] = force_filtered
        log["normal_pos"][k] = x[0]
        log["setpoint_x"][k] = x_d[0]
        log["tank"][k] = tank
        log["scale"][k] = scale
        log["contact"][k] = float(arm.data.ncon > 0)
        tau_log[k] = tau
    log["tau"] = tau_log
    return log


def _tool_normal_force(arm, tool_geom) -> float:
    """工具球所有接触的法向力之和；几何上法向即墙面法向。

    真机上这一读数来自腕部力/力矩传感器；仿真直接取接触求解器结果，
    测量误差（噪声、延迟、低通）在调用处叠加。
    """
    total = 0.0
    result = np.zeros(6)
    for i in range(arm.data.ncon):
        contact = arm.data.contact[i]
        if tool_geom in (contact.geom1, contact.geom2):
            arm.mujoco.mj_contactForce(arm.model, arm.data, i, result)
            total += abs(result[0])
    return total


def _case_metrics(config, log) -> dict:
    mask = log["t"] >= config.eval_start_s
    if not np.any(mask):
        raise ValueError("评估窗为空：duration_s 必须大于 eval_start_s")
    force = log["force_true"][mask]
    contact = log["contact"]
    first = np.argmax(contact > 0) if np.any(contact > 0) else len(contact)
    # 只统计 >= 5 ms 的接触丢失：求解器级的微颤不算一次"弹开"
    open_runs, run = [], 0
    for value in contact[first:]:
        if value == 0:
            run += 1
        elif run:
            open_runs.append(run)
            run = 0
    if run:
        open_runs.append(run)
    breaks = int(sum(length >= max(1, int(0.005 / config.dt)) for length in open_runs))
    return {
        "force_rmse_N": float(rms(force - config.force_target_N)),
        "force_mean_N": float(np.mean(force)),
        "force_peak_N": float(np.max(log["force_true"])),
        "contact_loss_events_after_touch": breaks,
        "eval_window_s": [config.eval_start_s, config.duration_s],
        "tank_min_J": float(np.min(log["tank"])),
        "scale_active_fraction": float(np.mean(log["scale"] < 0.999)),
    }


def run_demo(output_dir, config: ContactDemoConfig | None = None) -> dict:
    config = config or ContactDemoConfig()
    out = ensure_output_dir(output_dir)
    print("接触力控演示：无护栏（允许失稳）…", flush=True)
    unguarded = run_case(config, use_tank=False)
    print("接触力控演示：能量罐护栏…", flush=True)
    guarded = run_case(config, use_tank=True)

    metrics = {
        "demo": "contact_force_safety",
        "settings": config.to_dict(),
        "without_tank": _case_metrics(config, unguarded),
        "with_tank": _case_metrics(config, guarded),
        "notes": [
            "失稳机理：20 ms 力测量延迟 + 高导纳积分增益 + 硬接触 → 接触弹跳极限环。",
            "能量罐只限制法向设定点运动的注入能量；阻抗内环与横向滑动不受影响。",
            "scale_active_fraction 是护栏实际干预的时间占比。",
        ],
    }
    save_metrics(out / "metrics.json", rounded(metrics))
    _plot_force(out, config, unguarded, guarded, metrics)
    _plot_tank(out, config, unguarded, guarded)
    print(f"产物已写入 {out.resolve()}")
    return metrics


def _plot_force(out, config, unguarded, guarded, metrics):
    plt = use_headless_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, sharey=True)
    for axis, log, label, key in ((axes[0], unguarded, "without energy tank", "without_tank"),
                                  (axes[1], guarded, "with energy tank", "with_tank")):
        axis.plot(log["t"], log["force_true"], lw=0.9, color="#4c72b0", label="normal force (true)")
        axis.plot(log["t"], log["force_meas"], lw=0.8, color="0.6", ls=":",
                  label="measured (delayed + filtered)")
        axis.axhline(config.force_target_N, color="#55a868", lw=1.0, ls="--", label="target 15 N")
        axis.axvspan(config.eval_start_s, config.duration_s, color="0.92", zorder=0)
        m = metrics[key]
        axis.text(0.99, 0.95, f"RMSE {m['force_rmse_N']:.2f} N | peak {m['force_peak_N']:.1f} N | "
                  f"contact losses {m['contact_loss_events_after_touch']}",
                  transform=axis.transAxes, ha="right", va="top", fontsize=8)
        axis.set_ylabel("F_n [N]")
        axis.set_title(label, fontsize=10)
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")
    axes[1].set_xlabel("t [s]")
    fig.suptitle("Force tracking on a stiff wall: delayed force loop without vs with energy tank")
    fig.tight_layout()
    fig.savefig(out / "force_tracking.png", dpi=130)
    plt.close(fig)


def _plot_tank(out, config, unguarded, guarded):
    plt = use_headless_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True)
    axes[0].plot(unguarded["t"], unguarded["normal_pos"], lw=0.9, color="#c44e52",
                 label="tool x, without tank")
    axes[0].plot(guarded["t"], guarded["normal_pos"], lw=0.9, color="#4c72b0",
                 label="tool x, with tank")
    axes[0].plot(unguarded["t"], unguarded["setpoint_x"], lw=0.8, ls=":", color="#c44e52",
                 label="setpoint x, without tank")
    axes[0].plot(guarded["t"], guarded["setpoint_x"], lw=0.8, ls=":", color="#4c72b0",
                 label="setpoint x, with tank")
    axes[0].axhline(WALL_X, color="0.3", lw=1.0, ls="--", label="wall surface")
    axes[0].set_ylabel("x [m]")
    axes[0].legend(ncol=2, fontsize=8, loc="lower right")
    axes[0].set_title("Normal position and setpoint: the unguarded integrator digs into the wall")

    axes[1].plot(guarded["t"], guarded["tank"], lw=1.0, color="#4c72b0", label="tank energy")
    axes[1].axhline(config.tank_min_J, color="0.3", lw=0.8, ls="--", label="tank lower bound")
    twin = axes[1].twinx()
    twin.plot(guarded["t"], guarded["scale"], lw=0.8, color="#dd8452", alpha=0.8, label="setpoint scale")
    twin.set_ylabel("scale", color="#dd8452")
    twin.set_ylim(-0.05, 1.1)
    twin.grid(False)
    axes[1].set_ylabel("E_tank [J]")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title("Energy tank state and guard activation (guarded run)")

    for log, color, label in ((unguarded, "#c44e52", "without tank"),
                              (guarded, "#4c72b0", "with tank")):
        axes[2].plot(log["t"], log["contact"] + (0.0 if label == "without tank" else 1.2),
                     lw=0.8, color=color, label=label)
    axes[2].set_yticks([0, 1, 1.2, 2.2], ["open", "contact", "open", "contact"])
    axes[2].set_ylabel("contact state")
    axes[2].set_xlabel("t [s]")
    axes[2].legend(fontsize=8, loc="center right")
    axes[2].set_title("Contact on/off: bouncing appears as rapid toggling")
    fig.tight_layout()
    fig.savefig(out / "tank_and_contact.png", dpi=130)
    plt.close(fig)
