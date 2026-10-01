# 实验配置

从仓库根目录运行：

```bash
robot-identify --robot rrr \
  --config configs/rrr/realistic.json \
  --screening-config configs/rrr/screening.json \
  --trials 12 --periods 2 --output-dir outputs/rrr/baseline

robot-identify --robot franka \
  --config configs/franka/realistic.json \
  --screening-config configs/franka/screening.json \
  --recover-inertia --recovery-config configs/franka/recovery.json \
  --trials 12 --periods 2 --output-dir outputs/franka/baseline
```

- `rrr/realistic.json`：与 `experiments.rrr.default_config()` 一致，包含仿真步长、控制周期、PD、模型扰动、电机和测量参数。
- `rrr/screening.json`：与 `ExcitationCriteria()` 一致，包含轨迹可执行性与信息量门槛。
- `rrr/recovery.json`：启用名义软先验与惯性包络；质量取未扰动 MJCF 名义值的 0.5～2 倍，总质量取名义值 ±30%，质心取 capsule 几何包围盒加 10% 裕量。这些界限用于限制不可辨识方向上的参数偏移。
- `franka/realistic.json`：与 `experiments.franka.default_config()` 相同。力矩/速度上限取公开规格量级，PD 增益与摩擦是仿真假设，见 [Franka 的配置假设](#franka-的配置假设)。
- `franka/screening.json`：使用 `ExcitationCriteria()` 默认值，当前七轴仿真实验沿用这组门槛。
- `franka/recovery.json`：启用同样的名义软先验与惯性包络；质量取各连杆名义值的 0.5～2 倍、总质量取名义值 ±30%，质心取各连杆 mesh 包围盒加 10% 裕量。

## Franka 的配置假设

闭环采集使用 `panda_arm_motor.xml`（每关节一个 `gear=1` 的力矩 motor）；`panda_arm.xml` 是 position servo，只用于运动学与逆动力学对照，采集器会拒收。两者几何与惯性一致。

仿真对象由名义模型复制后扰动，摩擦和电机惯量由配置覆盖；控制器前馈仍用未扰动模型的 `armature=0.1`、`damping=1`、`frictionloss=0`，由此形成控制模型与对象之间的失配。恢复初值和先验只来自名义模型。

力矩上限为 `[87,87,87,87,12,12,12]` N·m；关节限位取 MJCF，`joint4` 的轨迹中心需取负半轴限位中点。

PD 为 `kp=(120,180,120,180,80,80,40)`、`kd=kp/10`。在 8 ms 电机滞后、2 步通信延迟下，历史 3 个种子 × 3 条轨迹扫描的最差跟踪 RMS 为 0.0090 rad，零饱和、接触和限位；整体增益加倍曾出现约 85% 饱和并发散。更改延迟后需重扫增益。

低速剔除要求全部 7 轴同时 `|dq|≥0.08 rad/s`。早期 `vel_scale=0.4`、加速度上限 4 rad/s² 时仅保留 1.4%～16%；改用当前包络、`vel_scale=1.2` 后保留约 23%～59%，因此幅值须兼顾有效样本量。

历史 6 条候选中 5 条通过，条件数约 225～1350；未通过项保留率 0.211，故沿用 RRR 的 0.25 保留率及其他筛选门槛。探索脚本见 [Franka 历史调参](../scripts/README.md#franka-历史调参)。

以上是当前仿真假设下的记录，PD、摩擦和传感器参数未经实机标定。Franka 默认关闭候选优化与解析搜索精炼；展示使用 12 条随机候选，增大搜索预算需另行验证。

## 恢复配置与输出

```bash
robot-identify --robot rrr --recover-inertia \
  --recovery-config configs/rrr/recovery.json \
  --output-dir outputs/rrr/recovered
```

两套配置的 `nominal_prior_weight=0.001`，只约束质量、质心、惯量，不把名义摩擦/电机参数固定。
`geometry_center` 取名义质心；`geometry_radius = 1.10 * max(几何包围半径, sqrt(trace(I_nominal)/(2*m_nominal)))`。
这约束质量、质心与惯量共同对应的二阶矩，称为“几何/名义惯性包络”。Franka 的 link1/2/7 名义惯量不容于纯 mesh 球，
因此半径同时考虑模型惯性，不能据此证明质量分布符合精确 CAD 几何。配置只依据未扰动模型，未使用对象真值或验证误差调节。

```bash
# 只检查配置来源；可选择 --output /tmp/recovery-geometry.json 保存详细证据
PYTHONPATH=src python scripts/inspect_recovery_geometry.py

# 已有辨识结果无需重新仿真；输出必须为新目录
PYTHONPATH=src python -m robot_model.cli.recover_inertia --robot rrr \
  --input-dir outputs/showcase-fresh/rrr \
  --recovery-config configs/rrr/recovery.json \
  --output-dir outputs/recovered-again/rrr
```

完整字段和默认值见 [RecoveryConfig](../src/robot_model/identification/recovery.py#L54)。库默认 `nominal_prior_weight=0`；提供名义参数时第一轮用其初始化，其余轮随机，未提供则全部随机。正权重必须同时提供名义参数。

变量界限的单位为质量 kg、质心 m、惯量 kg·m²；`mass_*`、`com_*`、`inertia_*` 和可选 `total_mass_*` 界限不能代替物理检查。`runs`、退火及局部迭代次数控制搜索预算，不保证全局最优。

`feasibility_tolerance=1e-6` 对质量/惯量检查沿用物理单位，对归一化包络违反量用无量纲比较；`max_base_error=0.05` 限制 `||Bπ−π_b||/max(||π_b||,1)`，并非每个参数的百分比误差或置信区间。

输出 `inertial-parameters.json` 保存质量、质心、惯量及完整/基参数和诊断，`recovery-config.json` 保存实际配置，`report.json` 保存恢复状态与独立验证。

`success` 同时要求 `solver_success`、`physical_feasible`、`matches_base`；启用包络后物理检查也要求 `geometry_feasible`，`*_bound_contacts` 另行记录参数达到边界的情况。失败结果仍保存用于诊断，命令返回非零退出码。

离线入口要求输出为输入目录外、尚不存在的新目录，并核对基参数定义与名义模型；只重做恢复，保留原始 `fit`、筛选、轨迹和测量数据。验证及真值在求解结束后读取，不进入求解与候选选择；Franka 将上面命令的机器人名和路径一并替换即可。

恢复得到的完整参数一般不唯一；初值、参数界限和搜索预算会影响各连杆数值，查看结果时应同时检查
求解状态、物理可行性、基参数匹配和独立验证，详见[惯性参数恢复原理](../docs/principles.md#8-完整惯性参数如何恢复)。

新实验可复制配置文件后修改；仿真种子可以用 `--seed` 覆盖，恢复器的随机种子由 `recovery.json` 的 `seed` 单独控制。轨迹数量、周期、优化、多场景和周期平均由 CLI 选项决定。
输出目录会保存实际使用的配置、轨迹、测量数据和报告。配置中的物理参数是仿真假设，详细单位及限制见 dataclass 与 [工程原理](../docs/principles.md)。
