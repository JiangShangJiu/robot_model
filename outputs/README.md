# 实验产物

这里保存可复查的实验结果和符号导出。Git 只保留下列四个目录；新实验、冒烟验证和临时对照默认忽略。Python 缓存与 Notebook 检查点也不提交。

## 保存的结果

- `showcase-fresh/rrr/`、`showcase-fresh/franka/`：闭环采集、轨迹筛选、基参数辨识及最初的惯性恢复，作为原始对照保留。
- `showcase-prior/rrr/`、`showcase-prior/franka/`：在相同的基参数拟合上，使用名义先验与几何/名义惯性包络重新恢复完整惯性参数。这是根 README 当前图表的数据来源。
- `rrr_symbolic/`：RRR 的 Python、C 和 LaTeX 动力学导出，由 [RRR 建模 Notebook](../notebooks/rrr/modeling.ipynb) 生成。
- `franka_symbolic/`：Franka 的 Python、C 和 LaTeX 动力学导出，以及 `identification/` 中的理想数据辨识、`excitation/` 中的激励轨迹对照。分别由 [Franka 建模](../notebooks/franka/modeling.ipynb) 和 [激励轨迹 Notebook](../notebooks/franka/excitation.ipynb) 生成；这些结果与闭环展示实验使用的模型配置不同。

`showcase-prior/` 保留完整的数据副本，便于直接绘图和复查。它与 `showcase-fresh/` 的采集数据、轨迹、筛选及基参数拟合一致，只更新恢复配置、恢复参数及对应评分。求解没有使用验证数据选择参数。

## 一次闭环实验保存什么

- `config.json`、`screening-config.json`、`recovery-config.json`：实际使用的仿真、筛选和恢复设置。
- `train.npz`、`validation.npz`：原始测量及仿真真值，用于重建预处理和独立评分。
- `train-processed.npz`、`validation-processed.npz`：滤波、对齐和低速剔除后实际用于辨识或评分的数据。
- `trajectories.npz`、`screening.json`、`optimization.json`：激励系数、候选筛选与优化记录。
- `report.json`、`inertial-parameters.json`：基参数拟合、完整惯性恢复、物理审计及独立验证结果。

离线恢复报告中的 `recovery_update.source_directory` 记录生成时的原始路径，仅用于溯源；在其他机器上请使用本仓库的 `outputs/showcase-fresh/<机器人>/`。`source_report_sha256` 可用于核对原始报告。

## 更新与新增

每次重跑使用新的 `outputs/<机器人>/<实验名>/` 或 `outputs/reproduce/<机器人>/`，先保留已有结果。命令和环境版本见 [根 README](../README.md#复现本页实验与图表)。

确认要保存后，在根 `.gitignore` 的 `outputs` 规则中添加对应一级目录的例外，例如 `!/outputs/paper-experiment/`，并在本页记录目录用途和生成入口。保留整套配置、数据和报告；更新展示结果时，同时重建 `docs/figures/` 中的图片和 JSON，并核对根 README 数值。

提交前查看候选文件：

```bash
git status --short --untracked-files=all -- outputs
git status --short --ignored -- outputs
```

目前 `franka/smoke/`、`rrr/recovery-smoke/` 继续留在本地，未删除。运行 Notebook 会改写符号目录中的同名文件，提交前应检查这些变化是否来自本次需要保存的运行。
