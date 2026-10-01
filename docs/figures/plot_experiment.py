"""由实机近似实验的输出目录生成 README 用图。

只读 ``--output-dir`` 里的文件，不重新跑实验；对象扰动由 ``config.json`` 的种子
复现，仅用于画真值对照。用法（仓库根目录）::

    PYTHONPATH=src python docs/figures/plot_experiment.py \
        --robot franka --output-dir outputs/showcase-prior/franka \
        --figure-dir docs/figures/franka

出图遵循 ``README.md`` 的“展示图表约定”：激励筛选、辨识用数据（连续滤波曲线与有效样本区间）、
验证力矩、惯性参数（基参数 + 可选的逐连杆质量、质心、质心惯量）。
同时保存原始报告副本与机器可读的惯性对照，便于不带 outputs/ 分享 README。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib import import_module
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from robot_model import Robot  # noqa: E402
from robot_model.data import MotionData  # noqa: E402
from robot_model.experiments.realistic import parameters  # noqa: E402
from robot_model.experiments.pipeline import prepare_measurements  # noqa: E402
from robot_model.identification import predict_torque  # noqa: E402
from robot_model.simulation import RealisticMujocoSource, RealismConfig  # noqa: E402


def load(output_dir: Path):
    def read(name):
        return json.loads((output_dir / name).read_text(encoding="utf-8"))

    recovery = output_dir / "inertial-parameters.json"
    return {
        "report": read("report.json"),
        "screening": read("screening.json"),
        "config": RealismConfig.from_dict(read("config.json")),
        "recovery": json.loads(recovery.read_text(encoding="utf-8")) if recovery.is_file() else None,
    }


def motion(path: Path, prefix: str = "") -> MotionData:
    with np.load(path) as values:
        return MotionData(values["t"], *(values[prefix + key] for key in ("q", "dq", "ddq", "tau")))


def retained_mask(t, kept_t):
    """把保存的有效样本映射回完整时间轴，要求每个时间戳均能匹配。"""
    indices = np.searchsorted(t, kept_t)
    if np.any(indices >= len(t)):
        raise ValueError("保留样本超出完整时间轴")
    np.testing.assert_allclose(t[indices], kept_t, rtol=0, atol=1e-10)
    mask = np.zeros(len(t), dtype=bool)
    mask[indices] = True
    if np.count_nonzero(mask) != len(kept_t):
        raise ValueError("保留样本存在重复时间戳")
    return mask


def continuous_measurements(raw, kept, config, *, period=None, cycles=None):
    """重做同样的测量预处理，只在展示副本中保留低速段；不插值或重新拟合。"""
    full = prepare_measurements(
        raw, encoder_delay_steps=config.sensors.encoder_delay_steps,
        torque_delay_steps=config.sensors.torque_delay_steps,
        period=period, cycles=cycles, min_speed=0.0,
    )
    np.testing.assert_allclose(np.diff(full.t), np.diff(full.t)[0], rtol=1e-8, atol=1e-10)
    mask = retained_mask(full.t, kept.t)
    # 核对保留样本与实际辨识、评分数据一致；连续展示不改变原评分区间。
    for field in ("q", "dq", "ddq", "tau"):
        np.testing.assert_allclose(getattr(full, field)[mask], getattr(kept, field),
                                   rtol=0, atol=1e-10)
    return full


def shade_excluded(axes, t, mask):
    """灰底表示未参与辨识/原报告评分的区间，曲线仍显示真实连续测量。"""
    edges = np.flatnonzero(np.diff(np.r_[False, ~mask, False]))
    half_step = np.median(np.diff(t)) / 2
    for axis in np.atleast_1d(axes):
        for start, stop in zip(edges[::2], edges[1::2]):
            axis.axvspan(max(t[0], t[start] - half_step),
                         min(t[-1], t[stop - 1] + half_step), color="0.90", zorder=0)
        axis.set_xlim(t[0], t[-1])


def plot_screening(screening, path: Path):
    """候选的实测条件数，淘汰的单独标色；分数来自预处理后的测量数据。"""
    records = screening["candidates"]
    ids = [r["candidate_id"] + 1 for r in records]
    conds = [(r["measured"] or {}).get("condition_number", np.nan) for r in records]
    accepted = [r["accepted"] for r in records]
    chosen = screening["selected_candidate_id"] + 1
    colors = ["#c44e52" if not ok else ("#4c72b0" if i != chosen else "#55a868")
              for i, ok in zip(ids, accepted)]
    figure, axis = plt.subplots(figsize=(8, 3.4))
    axis.bar(ids, conds, color=colors)
    axis.set_yscale("log")
    axis.set_xticks(ids)
    axis.set_xlim(min(ids) - .6, max(ids) + .6)
    axis.set_xlabel("candidate")
    axis.set_ylabel("measured cond(W)")
    axis.set_title(f"closed-loop screening: {screening['accepted_count']}/{len(records)} accepted; "
                   f"selected #{chosen} (green), rejected (red)")
    for i, (c, ok) in enumerate(zip(conds, accepted)):
        if np.isfinite(c):
            axis.text(ids[i], c, f"{c:.3g}", ha="center", va="bottom", fontsize=7)
        else:
            axis.text(ids[i], .03, "rejected\nbefore\nscoring", ha="center", va="bottom",
                      fontsize=7, color="#c44e52", transform=axis.get_xaxis_transform())
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_trajectory(trajectories, config, dof, path: Path, title: str):
    from robot_model.excitation import FourierTrajectory

    trajectory = FourierTrajectory(
        trajectories["train_q0"], trajectories["train_a"], trajectories["train_b"],
        float(trajectories["train_freq"]))
    t = trajectory.timestamps(rate=100.0)
    q, dq, ddq = trajectory.evaluate(t)
    figure, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i in range(dof):
        axes[0].plot(t, q[:, i], lw=1.0, label=f"q{i+1}")
        axes[1].plot(t, dq[:, i], lw=1.0)
        axes[2].plot(t, ddq[:, i], lw=1.0)
    for i in range(dof):
        axes[0].axhline(config.joint_lower[i], color=f"C{i%10}", ls=":", lw=.7, alpha=.6)
        axes[0].axhline(config.joint_upper[i], color=f"C{i%10}", ls=":", lw=.7, alpha=.6)
    for axis, limit in ((axes[1], config.velocity_limit), (axes[2], config.acceleration_limit)):
        for i in range(dof):
            axis.axhline(limit[i], color=f"C{i%10}", ls=":", lw=.7, alpha=.5)
            axis.axhline(-limit[i], color=f"C{i%10}", ls=":", lw=.7, alpha=.5)
    for axis, label in zip(axes, ("q [rad]", "dq [rad/s]", r"ddq [rad/s$^2$]")):
        axis.set_ylabel(label)
    axes[2].set_xlabel("t [s]")
    axes[0].set_title(title)
    axes[0].legend(ncol=min(dof, 7), fontsize=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_motion(data: MotionData, dof, path: Path, title: str, kept_t=None):
    """完整时间轴的 q/dq/ddq/tau；给 kept_t 时把不参与辨识的区间涂灰。"""
    figure, axes = plt.subplots(4, 1, figsize=(10, 8.5), sharex=True)
    series = ((data.q, "q [rad]"), (data.dq, "dq [rad/s]"),
              (data.ddq, r"ddq [rad/s$^2$]"), (data.tau, "tau [N·m]"))
    for axis, (values, label) in zip(axes, series):
        for i in range(dof):
            axis.plot(data.t, values[:, i], lw=.8, label=f"{i+1}" if label.startswith("q ") else None)
        axis.set_ylabel(label)
    if kept_t is not None:
        shade_excluded(axes, data.t, retained_mask(data.t, kept_t))
    axes[-1].set_xlabel("t [s]")
    axes[0].set_title(title)
    axes[0].legend(ncol=min(dof, 7), fontsize=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def plot_validation_torque(data, kept, Hb, fitted, nominal, dof, path: Path, title: str, expected):
    """连续验证曲线 + 原评分掩码；原报告 RMSE 与完整区间 RMSE 分开记录。"""
    mask = retained_mask(data.t, kept.t)
    tau_fitted = predict_torque(Hb, data, fitted)
    tau_nominal = predict_torque(Hb, data, nominal)
    scored, full_scores = {}, {}
    for key, prediction in (("measured_fitted", tau_fitted), ("measured_nominal", tau_nominal)):
        scored[key] = np.sqrt(np.mean((kept.tau - prediction[mask]) ** 2, axis=0))
        np.testing.assert_allclose(scored[key], expected[key], rtol=1e-8, atol=1e-10)
        full_scores[key] = np.sqrt(np.mean((data.tau - prediction) ** 2, axis=0)).tolist()
    figure, axes = plt.subplots(dof, 1, figsize=(10, 1.45 * dof + .4), sharex=True)
    axes = np.atleast_1d(axes)
    shade_excluded(axes, data.t, mask)
    for i, axis in enumerate(axes):
        axis.plot(data.t, data.tau[:, i], lw=1.2, color="0.25", label="measured")
        axis.plot(data.t, tau_fitted[:, i], lw=1.0, ls="--", color="#55a868", label="identified")
        axis.plot(data.t, tau_nominal[:, i], lw=.9, ls=":", color="#c44e52", label="nominal")
        rms_fitted, rms_nominal = scored["measured_fitted"][i], scored["measured_nominal"][i]
        axis.set_ylabel(f"tau{i+1} [N m]", fontsize=8)
        axis.text(.995, .90, f"scored RMSE: identified {rms_fitted:.3f} / nominal {rms_nominal:.3f} N m",
                  transform=axis.transAxes, ha="right", va="top", fontsize=7,
                  bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 1.5})
    axes[0].set_title(title + "\nGray = excluded from reported RMSE; continuous filtered measurements")
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("t [s]")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return {"samples": len(data), "scored_samples": len(kept),
            "scope": "Aligned, filtered, edge-trimmed full timeline including low-speed samples; display only.",
            "rms_Nm": full_scores,
            "overall_rms_Nm": {key: float(np.sqrt(np.mean(np.square(values))))
                               for key, values in full_scores.items()}}


def plot_base_parameters(fitted, plant, path: Path, title: str):
    index = np.arange(len(fitted))
    figure, axes = plt.subplots(2, 1, figsize=(11, 5.5),
                                gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    width = .42
    axes[0].bar(index - width/2, fitted, width, label="identified", color="#4c72b0")
    axes[0].bar(index + width/2, plant, width, label="plant truth projection", color="#dd8452")
    axes[0].set_ylabel("base parameter value")
    axes[0].legend(fontsize=8)
    axes[0].set_title(title)
    axes[1].bar(index, fitted - plant, color="0.4")
    axes[1].set_ylabel("difference")
    axes[1].set_xlabel("base parameter index")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_recovered_masses(recovery, nominal_masses, plant_masses, path: Path, title: str):
    masses = np.asarray(recovery["masses"], float)
    config = recovery["report"]["config"]
    lower = np.broadcast_to(np.asarray(config["mass_lower"], float), masses.shape)
    upper = np.broadcast_to(np.asarray(config["mass_upper"], float), masses.shape)
    pinned = (np.abs(masses - lower) < 1e-3) | (np.abs(masses - upper) < 1e-3)
    index = np.arange(len(masses))
    width = .25
    figure, axis = plt.subplots(figsize=(9, 3.8))
    axis.bar(index - width, nominal_masses, width, label="nominal MJCF", color="0.65")
    axis.bar(index, plant_masses, width, label="plant truth (scoring only)", color="#dd8452")
    axis.bar(index + width, masses, width, label="recovered (one feasible solution)", color="#4c72b0")
    axis.vlines(index + width, lower, upper, color="0.3", lw=1, label="mass bounds")
    for i in index[pinned]:
        axis.annotate("at bound", (i + width, masses[i]), textcoords="offset points",
                      xytext=(0, 6), ha="center", fontsize=8, color="#c44e52")
    axis.set_xticks(index, [f"link{i+1}" for i in index])
    axis.set_ylabel("mass [kg]")
    axis.set_title(title)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_inertial_components(comparison, path: Path, *, inertia=False):
    """连杆坐标系中的质心/完整对称质心惯量；对象真值只用于展示。"""
    field = "inertias_com" if inertia else "centers_of_mass"
    components = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)] if inertia else range(3)
    labels = ["Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"] if inertia else ["cx", "cy", "cz"]
    figure, axes = plt.subplots(2 if inertia else 1, 3,
                                figsize=(11, 6.2 if inertia else 3.6), squeeze=False)
    index = np.arange(1, len(comparison["recovered"][field]) + 1)
    for axis, component, label in zip(axes.flat, components, labels):
        for source, legend, color, style in (
            ("nominal", "nominal MJCF", "0.55", ":"),
            ("plant_truth_scoring_only", "plant truth (scoring only)", "#dd8452", "--"),
            ("recovered", "one feasible solution", "#4c72b0", "-"),
        ):
            values = np.asarray(comparison[source][field])
            y = values[:, component[0], component[1]] if inertia else values[:, component]
            axis.plot(index, y, style, marker="o", ms=3, color=color, label=legend)
        if not inertia:
            for contact in comparison["acceptance"].get("com_bound_contacts", []):
                if contact["component"] == "xyz"[component]:
                    axis.plot(contact["link"], contact["value"], "D", color="#c44e52", ms=5)
                    axis.annotate("at COM bound", (contact["link"], contact["value"]),
                                  xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7,
                                  color="#c44e52")
        axis.set_xticks(index)
        axis.set_xlabel("link")
        axis.set_ylabel(label + (r" [kg m$^2$]" if inertia else " [m]"))
        axis.grid(alpha=.2)
    title = "COM inertia tensor" if inertia else "center of mass"
    figure.suptitle(f"Per-link {title}: one feasible solution, not unique truth\n"
                   "Expressed in link coordinate axes")
    handles, legends = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legends, loc="lower center", ncol=3, fontsize=8)
    figure.tight_layout(rect=(0, .07, 1, .90))
    figure.savefig(path, dpi=140)
    plt.close(figure)


def link_values(model, body_names):
    from robot_model.simulation import MujocoReference

    links = MujocoReference(model, body_names).links
    return {"masses": [link.m for link in links],
            "centers_of_mass": [link.r.tolist() for link in links],
            "inertias_com": [link.I.tolist() for link in links]}


def main():
    parser = argparse.ArgumentParser(description="由实验输出生成 README 用图")
    parser.add_argument("--robot", choices=("rrr", "franka"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    # 图上一律用英文标签：中文字形依赖本机字体，缺字时 matplotlib 会画成方框
    matplotlib.rcParams["axes.unicode_minus"] = False
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    import mujoco

    loaded = load(args.output_dir)
    report, config = loaded["report"], loaded["config"]
    if report["status"] != "completed":
        raise ValueError("只为完成且验收通过的实验生成展示；请先检查 report.json")
    module = import_module(f"robot_model.experiments.{args.robot}")
    description = module.setup()
    dof = description.dof

    robot = Robot.from_mdh(description.name, description.mdh_parms,
                           frictionmodel={"viscous", "Coulomb"}, driveinertiamodel="simplified")
    dynamics = robot.dynamics
    dynamics.calc_base_parms(samples=description.base_parms_samples)
    dynamics.gen_base_regressor()
    Hb = dynamics.Hb_func
    # 用同一个种子复现对象扰动，仅供真值对照，不参与任何估计
    source = RealisticMujocoSource(mujoco.MjModel.from_xml_path(str(description.xml_path)), config)
    nominal_parms = parameters(robot, source.nominal_model, description.body_names)
    plant_parms = parameters(robot, source.model, description.body_names)
    fitted = np.asarray(report["fit"]["parameters"], float)
    np.testing.assert_allclose(
        np.linalg.norm(fitted - plant_parms) / np.linalg.norm(plant_parms),
        report["fit"]["relative_error_to_rigid_friction_component"], rtol=1e-8, atol=1e-10)

    plot_screening(loaded["screening"], args.figure_dir / "screening.png")
    periods = {}
    with np.load(args.output_dir / "trajectories.npz") as trajectories:
        if report["enhancements"]["periodic_average"]:
            periods = {phase: 1 / float(trajectories[phase + "_freq"])
                       for phase in ("train", "validation")}
        plot_trajectory(trajectories, config, dof, args.figure_dir / "trajectory.png",
                        f"selected training excitation (f0={float(trajectories['train_freq']):.2f} Hz, "
                        f"dotted = design limits)")
    train = motion(args.output_dir / "train.npz")
    train_kept = motion(args.output_dir / "train-processed.npz")
    plot_motion(train, dof, args.figure_dir / "train_data.png",
                f"training measurements, closed loop (N={len(train)}; gray = discarded)",
                kept_t=train_kept.t)
    train_full = continuous_measurements(train, train_kept, config,
        period=periods.get("train"), cycles=report["experiment"]["recorded_periods"])
    plot_motion(train_full, dof, args.figure_dir / "train_kept.png",
                f"continuous filtered training measurements (N={len(train_full)})\n"
                f"white = {len(train_kept)} identification samples; gray = excluded low-speed intervals",
                kept_t=train_kept.t)
    validation = motion(args.output_dir / "validation-processed.npz")
    validation_full = continuous_measurements(motion(args.output_dir / "validation.npz"), validation, config,
        period=periods.get("validation"), cycles=report["experiment"]["recorded_periods"])
    continuous_validation = plot_validation_torque(validation_full, validation, Hb, fitted, nominal_parms, dof,
                           args.figure_dir / "validation_torque.png",
                           "independent validation: measured vs identified vs nominal",
                           expected=report["validation_rms_Nm"])
    plot_base_parameters(fitted, plant_parms, args.figure_dir / "base_params.png",
                         f"{len(fitted)} base parameters: identified vs plant-truth projection "
                         f"(relative error {report['fit']['relative_error_to_rigid_friction_component']:.3f})")
    summary = {
        "robot": report.get("robot", description.name),
        "mujoco_version": report["mujoco_version"],
        "source_report_sha256": hashlib.sha256((args.output_dir / "report.json").read_bytes()).hexdigest(),
        "experiment": report["experiment"],
        "recovery_update": report.get("recovery_update"),
        "enhancements": report["enhancements"],
        "n_base": report["fit"]["n_parms"],
        "rank": report["fit"]["rank"],
        "condition_number": report["fit"]["condition_number"],
        "residual_rms_Nm": report["fit"]["residual_rms_Nm"],
        "samples": report["fit"]["samples_after_preprocessing"],
        "raw_samples": len(train),
        "base_parameter_relative_error": report["fit"]["relative_error_to_rigid_friction_component"],
        "validation_rms_Nm": report["validation_rms_Nm"],
        "continuous_validation": continuous_validation,
        "display_preprocessing": "Saved raw measurements, same alignment/filtering/edge trim, low-speed intervals retained for continuous display only.",
        "screening": {
            "accepted": loaded["screening"]["accepted_count"],
            "total": len(loaded["screening"]["candidates"]),
            "selected": loaded["screening"]["selected_candidate_id"] + 1,
        },
        "warnings": report["warnings"],
    }
    overall = {key: float(np.sqrt(np.mean(np.square(report["validation_rms_Nm"][key]))))
               for key in ("measured_fitted", "measured_nominal")}
    overall["reduction_fraction"] = 1 - overall["measured_fitted"] / overall["measured_nominal"]
    summary["validation_overall_rms_Nm"] = overall
    if loaded["recovery"] is not None:
        recovery = loaded["recovery"]
        comparison = {
            "interpretation": "One physically feasible solution; full link inertias are not uniquely identifiable.",
            "units": recovery["units"], "inertia_frame": recovery["inertia_frame"],
            "truth_source": "Reconstructed from saved config seed and repository MJCF; scoring only.",
            "nominal": link_values(source.nominal_model, description.body_names),
            "plant_truth_scoring_only": link_values(source.model, description.body_names),
            "recovered": {key: recovery[key] for key in ("masses", "centers_of_mass", "inertias_com")},
            "acceptance": {key: recovery["report"][key] for key in
                           ("solver_success", "physical_feasible", "matches_base", "base_error",
                            "geometry_enabled", "geometry_feasible", "nominal_prior_enabled", "nominal_prior_weight",
                            "mass_bound_contacts", "com_bound_contacts", "inertia_bound_contacts")
                           if key in recovery["report"]},
        }
        nominal_masses = comparison["nominal"]["masses"]
        plot_recovered_masses(recovery, nominal_masses, comparison["plant_truth_scoring_only"]["masses"],
                              args.figure_dir / "inertia_mass.png",
                              "per-link mass: nominal, plant truth and one feasible solution")
        plot_inertial_components(comparison, args.figure_dir / "inertia_com.png")
        plot_inertial_components(comparison, args.figure_dir / "inertia_tensor.png", inertia=True)
        (args.figure_dir / "inertia-comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        summary["recovery"] = {
            key: recovery["report"][key]
            for key in ("success", "solver_success", "physical_feasible", "matches_base", "base_error",
                        "geometry_enabled", "geometry_feasible", "nominal_prior_enabled", "nominal_prior_weight",
                        "mass_bound_contacts", "com_bound_contacts", "inertia_bound_contacts")
            if key in recovery["report"]
        }
        summary["recovery"]["masses"] = recovery["masses"]
        summary["recovery"]["nominal_masses"] = nominal_masses
    (args.figure_dir / "report.json").write_bytes((args.output_dir / "report.json").read_bytes())
    (args.figure_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"figures and summary.json written to {args.figure_dir}")


if __name__ == "__main__":
    main()
