# 辅助实验脚本

核心辨识流程使用 `robot-identify`，见 [项目说明](../README.md)。本目录存放调参、约束检查和性能基准的辅助脚本。实验数据与生成文件写入 `outputs/`。脚本不是自动化测试，也不会被安装为库的一部分。

## Franka 历史调参

`franka_tuning/` 保存控制增益与轨迹幅值的探索脚本；仿真结果见
[Franka 的配置假设](../configs/README.md#franka-的配置假设)：

- `sweep.py`：扫描控制增益和激励速度比例，检查跟踪、饱和和关节约束。
- `retention.py`：比较轨迹幅度对应的有效样本保留率与跟踪误差。
- `envelope.py`：检查不同幅度下的实际速度、加速度包络与样本保留率。

这些脚本保留当时的候选设置和验收阈值；当前实验配置见 `configs/franka/`。
从仓库根目录运行（需要安装仿真与辨识依赖）：

```bash
PYTHONPATH=src python scripts/franka_tuning/sweep.py
PYTHONPATH=src python scripts/franka_tuning/retention.py
PYTHONPATH=src python scripts/franka_tuning/envelope.py
```

## 恢复约束来源检查

`inspect_recovery_geometry.py` 读取未扰动 MJCF 的 capsule/mesh，生成每个连杆的几何范围、名义惯性半径和建议包络；不会覆盖配置。

```bash
PYTHONPATH=src python scripts/inspect_recovery_geometry.py --output /tmp/recovery-geometry.json
```

只重做已有数据的惯性恢复使用 `python -m robot_model.cli.recover_inertia`，具体命令见 [恢复配置与输出](../configs/README.md#恢复配置与输出)。

## 数值回归基准

数值回归按连杆合批参数列，同一状态共用正向运动学；摩擦与电机惯量列直接构造，不改变采样、筛选或最小二乘。基准保留原逐列算法作为对照，见 [基准脚本](../tests/franka/benchmark_numeric_regressor.py)。

2026-10-01 本机使用全部 2255 个 Franka 训练样本、`seed=0` 和 400 点基参数映射：回归求值 56.380 s → 1.916 s，基映射 8.192 s → 0.349 s，秩保持 62；完整回归矩阵、基映射和辨识参数最大绝对差均为 0.0。详细值见 [基准记录](../docs/figures/franka/numeric-benchmark.json)。

复现需要保留的采集数据；单次耗时受负载和 BLAS 线程数影响。记录基准的环境使用 MKL，命令中同时限制 OpenBLAS、MKL 和 OpenMP 的线程数：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m tests.franka.benchmark_numeric_regressor \
  --data outputs/showcase-fresh/franka/train-processed.npz
```

2026-10-01 加入名义先验与包络后的历史测试记录为 290 passed, 7 deselected（227.94 s），及 Franka 慢速闭环冒烟测试 1 passed, 6 deselected（21.69 s）：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest -m "not slow" -q
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest -m slow tests/franka/test_franka_realistic.py -q
```

其中数值一致性覆盖 Park/Khalil、DH/MDH/PoE、混合关节及两种参数顺序；恢复与离线入口覆盖梯度、先验/包络、候选排序、原数据保留和失败退出。更早同日、加入先验前曾跑完整慢测 7 passed, 264 deselected（251.91 s），包括符号导出；先验更新后仅重跑上面的 1 项慢测。以上记录对应当时的代码状态；后续修改仍需运行相应测试。
