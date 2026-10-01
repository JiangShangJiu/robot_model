# 工程原理：从机器人模型到动力学参数辨识

本项目在已知关节结构和连杆几何的前提下，用运动与力矩数据辨识动力学参数。先估计能预测关节力矩的基参数，再结合名义模型和物理约束，恢复各连杆的质量、质心和惯量。

验证模型是 RRR 三轴机械臂和 Franka Panda 七轴机械臂。MuJoCo 提供被控对象，控制器跟踪激励轨迹，测量模块记录数据，辨识程序处理采集结果。

本文说明模型、实验与辨识算法之间的关系。安装、运行和实验结果见 [README](../README.md)；接口细节见本文的 [建模接口与公式导出](#建模接口与公式导出) 和 [辨识接口与实验选项](#辨识接口与实验选项)；文件分工见 [目录导航](../README.md#目录导航)。

## 1. 先分清三个问题

建模是正向问题：给定几何、质量、质心、惯量，以及某一时刻的位置、速度、加速度，计算需要多少关节力矩。

基参数辨识是反向问题：给定一段测得的位置、速度、加速度和力矩，估计决定力矩的独立参数组合。

完整惯性参数恢复是在基参数辨识之后，寻找各连杆的质量、质心和惯量，使它们既能解释辨识结果，又满足给定的物理条件与先验。多组完整参数可能产生相同力矩，因此这一步通常没有唯一答案。

三个问题的未知量不同。这里默认连杆几何和关节坐标约定已经确定，当前流程不同时估计 DH 尺寸、关节零位或传感器延迟。

~~~mermaid
flowchart TD
    A["机器人几何与模型选项"] --> B["逆动力学与完整回归矩阵 H"]
    B --> C["基参数映射 B 与基回归矩阵 Hb"]
    C --> D["候选激励轨迹：信息量与运动约束"]
    D --> E["闭环试跑、测量、预处理与筛选"]
    E --> F["用选中训练数据估计基参数"]
    F --> G["可选：完整惯性参数恢复"]
    N["未扰动名义模型与尺寸信息"] --> G
    F --> V["独立验证轨迹上的力矩预测"]
    G --> V
~~~

验证轨迹在筛选训练候选之前固定。验证误差不参与候选排名、参数拟合或恢复约束调整。

## 2. 从几何得到动力学

### 2.1 几何决定坐标变换

关节位置记为 $q$，速度、加速度分别为 $\dot q$、$\ddot q$。DH、MDH 和 PoE 都是在描述连杆如何相连、关节绕哪根轴运动。

仓库把这些描述转成统一的连杆坐标变换 `FrameChain`，再计算正运动学、雅可比和动力学。RRR、Franka 的实验入口使用 MDH；连杆坐标系要与 MJCF 中对应的 body 帧保持一致。

质心坐标随参考系改变，惯量也取决于参考点和坐标轴。对比不同模型的参数前，需要先对齐这些约定。

实现入口：[robot.py](../src/robot_model/robot.py)、[geometry/](../src/robot_model/geometry/)、[robots/](../src/robot_model/robots/)。

### 2.2 逆动力学算的是关节力矩

当前辨识模型可以写成：

$$
\tau =
M_{\mathrm{rb}}(q)\ddot q
+h_{\mathrm{rb}}(q,\dot q)
+g(q)
+\operatorname{diag}(I_a)\ddot q
+f_v\odot\dot q
+f_c\odot\operatorname{sign}(\dot q).
$$

这里 $M_{\mathrm{rb}}$ 是刚体惯性矩阵，$h_{\mathrm{rb}}$ 是科氏与离心力矩，$g$ 是重力力矩；$I_a$ 是简化的关节侧等效电机惯量，$f_v$、$f_c$ 是粘性和库仑摩擦系数，$\odot$ 表示逐元素相乘。若关闭摩擦或电机选项，相应项不进入模型。对象侧的额外外扰、低速摩擦峰和传感器误差不能完全由这些参数表达，会体现在模型残差中。

库支持递归牛顿–欧拉的 Park/Khalil 实现，以及拉格朗日实现。同一模型下可交叉检查三种实现的结果。较小的 RRR 适合生成符号表达式；七轴 Franka 日常辨识采用数值回归，避免把庞大的符号式全部展开。完整符号导出仍然可用。

实现入口：[dynamics/model.py](../src/robot_model/dynamics/model.py)、[dynamics/numeric.py](../src/robot_model/dynamics/numeric.py)、[dynamics/actuator.py](../src/robot_model/dynamics/actuator.py)。

## 3. 为什么要把参数改写成线性形式

动力学对关节运动是非线性的，但使用适当的惯性参数表示后，力矩对参数呈线性关系：

$$
\tau=H(q,\dot q,\ddot q)\pi.
$$

$H$ 是随运动状态变化的回归矩阵，$\pi$ 是待辨识的常数向量。这里的线性只针对参数。

对第 $i$ 根连杆，用 $m_i$ 表示质量、$r_i$ 表示连杆坐标系中的质心、$I_i$ 表示以质心为参考点且沿连杆坐标轴表达的惯量。构造：

$$
l_i=m_i r_i,\qquad
L_i=I_i+m_i\left((r_i^\mathsf{T}r_i)\mathbf{1}_3-r_ir_i^\mathsf{T}\right).
$$

$\mathbf{1}_3$ 是三维单位矩阵；$l_i$ 是一阶质量矩，$L_i$ 是移到连杆坐标原点后的惯量。平行轴定理使完整参数可以用 $L_i$ 的六个独立分量、$l_i$ 的三个分量和 $m_i$ 表示，共十个刚体参数。

当前实验再加每关节的 $I_a,f_v,f_c$，因此每根连杆对应十三个完整参数。代码默认按连杆排列：

~~~text
Lxx, Lxy, Lxz, Lyy, Lyz, Lzz, lx, ly, lz, m, Ia, fv, fc
~~~

实际顺序以 `robot.symbols.dynparms()` 为准，构造参数数组时需与所用模型一致。恢复结果中的 `inertias_com` 保存的是质心惯量 $I_i$，不是上述 $L_i$。

### 3.1 完整参数为什么不能直接唯一辨识

连杆之间的动力学耦合会让 $H$ 的部分列线性相关。增加采样量也无法分离这些参数，只能辨识它们的组合。

仓库先从多组运动状态构造回归矩阵，通过秩判断与 QR 分解找出独立列，形成基回归矩阵 $H_b$ 和基参数 $\pi_b$：

$$
H_b=HP_b,\qquad H_d=HP_d=H_bK_d,\qquad
\pi_b=(P_b^\mathsf{T}+K_dP_d^\mathsf{T})\pi=B\pi,
\qquad
\tau=H_b\pi_b.
$$

$H_d$ 表示依赖列组成的矩阵，$P_b$、$P_d$ 分别选择独立列和依赖列，$K_d$ 表示依赖关系；在实现中，这些关系通过数值采样和容差确定。因此，单个基参数往往是多个物理参数的组合。

当前几何和摩擦/电机选项下，RRR 是 39 个完整参数 → 22 个基参数；Franka 是 91 → 62。只考虑刚体的 Franka 则是 43 个基参数。参数数量随模型选项和几何变化，不能跨模型直接复用基参数数组。

结构冗余和激励不足会分别造成秩不足。基参数消除了前者；要区分所有基参数，还需要训练轨迹提供足够的激励。

实现入口：[dynamics/base_parms.py](../src/robot_model/dynamics/base_parms.py)。

## 4. 为什么机械臂要执行专门的激励轨迹

重复缓慢移动或只运动一个关节，可能使不同参数对力矩的影响很难区分。激励设计就是在满足运动约束的前提下，让数据包含足够多的独立信息。

激励轨迹用有限项傅里叶级数表示。对某一关节，令 $\omega=2\pi f$：

$$
q(t)=q_0+\sum_{k=1}^{K}
\left[\frac{a_k}{k\omega}\sin(k\omega t)
-\frac{b_k}{k\omega}\cos(k\omega t)\right],
$$

$$
\dot q(t)=\sum_{k=1}^{K}[a_k\cos(k\omega t)+b_k\sin(k\omega t)],
$$

$$
\ddot q(t)=\sum_{k=1}^{K}k\omega[-a_k\sin(k\omega t)+b_k\cos(k\omega t)].
$$

周期便于重复采样；频率和谐波数控制运动带宽；参考位置、速度、加速度都有解析式。这些解析量用于生成期望轨迹，闭环辨识使用实际测量量。

候选首先满足位置、速度、加速度约束，再比较回归矩阵的信息量。将多个时刻的 $H_b$ 按行堆叠为 $W$，其条件数为 $\kappa(W)=\sigma_{\max}/\sigma_{\min}$。条件数大意味着某些参数方向难以区分，测量误差更容易被放大。

当前流程还要求候选实际闭环试跑，检查跟踪、饱和、接触、限位、实际速度/加速度以及预处理后的有效样本。合格候选默认按实测回归矩阵条件数排名，同时检查满秩和最小奇异值。筛选不按力矩拟合残差或真值参数误差挑选。

最小奇异值使用 $W/\sqrt{N}$，其中 $N$ 是采样时刻数。这减少记录长度对评分的直接影响，但没有消除参数量纲差异；应在同一机器人、同一基参数定义下比较。

通用 CLI 对 RRR 默认增加 SLSQP 系数优化候选，对 Franka 默认关闭；解析评分改善后的轨迹仍须经过闭环检查。多场景筛选、周期平均是另行开启的选项，详见 [多场景筛选与周期平均](#多场景筛选与周期平均)。

实现入口：[trajectory.py](../src/robot_model/excitation/trajectory.py)、[screening.py](../src/robot_model/excitation/screening.py)、[optimization.py](../src/robot_model/excitation/optimization.py)。

## 5. MuJoCo 如何代替被辨识的机器人

实验保留一个未扰动的名义模型，再复制出一个被控对象模型。对象的质量、质心和惯量带扰动，还叠加摩擦、执行器动态、测量误差和外扰；控制器前馈只使用名义模型。

控制指令是：

$$
\tau_{\mathrm{cmd}}=
\tau_{\mathrm{nominal}}(q_d,\dot q_d,\ddot q_d)
+K_{\mathrm{P}}(q_d-q_m)+K_{\mathrm{D}}(\dot q_d-\dot q_m).
$$

下标 $d$ 表示期望量，$m$ 表示测量量；$K_{\mathrm{P}}$、$K_{\mathrm{D}}$ 是位置和速度误差的反馈增益，对应配置中的 `kp`、`kd`。

指令随后经过电机延迟、饱和、一阶滞后和变化率限制，才成为实际作用力矩。因此 `command`、`ctrl` 与测得的力矩不是同一回事。力矩测量通道读取实际 `qfrc_actuator`，再施加通道误差。

位置测量经过编码器量化、噪声和延迟；反馈速度由测量位置差分并因果滤波得到。反馈不读取仿真真实速度。当前默认物理积分为 1000 Hz，控制与采样为 250 Hz。

这样采集的数据包含模型失配、跟踪误差和测量误差。项目另外保留理想逆动力学数据路径，用于检查公式和算法的数值一致性。

仿真真值用于两处：运动状态用于检查候选的可执行性；真实惯性和 `truth_*` 数据用于离线评分。它们不进入测量预处理、最小二乘、测量反馈或恢复先验。

实现入口：[simulation/realistic.py](../src/robot_model/simulation/realistic.py)、[hardware.py](../src/robot_model/simulation/hardware.py)、[sensors.py](../src/robot_model/simulation/sensors.py)。

## 6. 数据为什么需要预处理

进入辨识的数据统一表示为 `MotionData(t, q, dq, ddq, tau)`。当前预处理顺序是：

1. 补偿已知整数通道延迟：重新索引位置/速度和力矩，使它们对应同一物理时刻。未知延迟没有在这里自动估计。
2. 可选周期平均：同一对象重复执行同一轨迹时，按相位对齐并平均，减小随机波动；系统偏置仍会保留。
3. 离线零相位低通：默认 5 Hz，处理位置、速度、力矩。加速度由速度滤波后微分，再滤波得到。
4. 去掉滤波边缘：默认两端各 0.3 s，减轻边缘效应。
5. 剔除低速样本：只要任一关节 $|\dot q|<0.08$ rad/s，整个时刻的数据就不参与拟合。

微分会放大高频噪声，因此速度先滤波，再用于估计加速度。零相位滤波用于离线数据，不进入需要实时输出的反馈控制器；它也不补偿在线测速滤波器本身的相位延迟。

低速剔除针对的是模型适用范围：简单的 $f_c\operatorname{sign}(\dot q)$ 无法充分表达过零附近的摩擦。七轴要求所有关节同时达标，因此 Franka 对激励幅度和样本保留率更敏感。

剔除低速样本后时间戳会有间隔，原始采集则是连续的。README 图中画的是完整滤波序列，灰底标出未用于拟合或原报告评分的低速区间；灰底不是插值得到的数据。原报告 RMSE 按保留样本计算，整段误差另行报告。

实现入口：[experiments/pipeline.py](../src/robot_model/experiments/pipeline.py)、[data/dataset.py](../src/robot_model/data/dataset.py)、[data/filtering.py](../src/robot_model/data/filtering.py)。

## 7. 基参数具体怎么辨识

设保留了 $N$ 个时刻、机器人有 $n$ 个关节、基参数有 $p_b$ 个。对每个时刻构造 $n\times p_b$ 的 $H_b$，堆叠成：

$$
W=\begin{bmatrix}H_b(t_1)\\ \vdots\\ H_b(t_N)\end{bmatrix},
\qquad
y=\begin{bmatrix}\tau(t_1)\\ \vdots\\ \tau(t_N)\end{bmatrix},
\qquad
\hat\pi_b=\arg\min_{\pi_b}\|W\pi_b-y\|_2^2.
$$

$W$ 的大小是 $(Nn)\times p_b$。实现调用 `numpy.linalg.lstsq`，没有显式求 $(W^\mathsf{T}W)^{-1}$。接口支持每关节权重，当前闭环展示实验没有传入权重，也没有在这一步加入名义参数正则。

估计后检查秩、条件数和残差。满秩意味着这批数据在数值意义上能区分所选基参数，不意味着没有偏差。运动数据本身含噪，会让 $W$ 也含误差；滤波、未建模摩擦和外扰也会影响估计。

训练残差衡量模型对训练数据的拟合程度；参数精度、其他轨迹上的预测能力以及逐连杆物理参数，需要分别验证。

实现入口：[identification/estimator.py](../src/robot_model/identification/estimator.py)。

## 8. 完整惯性参数如何恢复

### 8.1 恢复器解决另一个优化问题

辨识得到 $\hat\pi_b$ 后，恢复器把每根连杆的质量、质心、对称质心惯量及启用的附加参数作为变量 $x$。先用平行轴定理得到完整参数 $\pi(x)$，再用同一个映射 $B$ 得到基参数。

当前带先验的目标是：

$$
J(x)=\|B\pi(x)-\hat\pi_b\|_2^2
+s^2\alpha R(x)+\lambda P(x),
\qquad s=\max(\|\hat\pi_b\|_2,1).
$$

第一项要求恢复模型匹配已辨识的基参数；$R(x)$ 是相对名义惯性参数的偏离；$P(x)$ 是物理条件违反量，$\lambda$ 随求解阶段逐渐增加。参数盒界限由有界优化器直接施加。

这一步没有重新用训练力矩做最小二乘，也不使用验证误差挑选参数。命令行入口 [recover_inertia.py](../src/robot_model/cli/recover_inertia.py) 因而可以读取已保存的基参数，单独重做恢复。

### 8.2 名义软先验有什么作用

仅匹配基参数时，一些不可辨识方向仍然自由，可能出现极小质量配较大惯量，或者质量在不同连杆间重新分配。名义先验限制这些方向上的偏移，使解靠近已有模型的参数量级。

当前 $R(x)$ 将每根连杆的质量偏差、质心距离、惯量 Frobenius 偏差分别按名义质量、长度尺度和惯量尺度归一化，再求平方和并除以 $3n$。它只作用于质量、质心、惯量，不约束摩擦和电机项。

两套实验的恢复配置均取 $\alpha=0.001$；库接口 `RecoveryConfig()` 默认取 0。先验使用未扰动名义模型，对象真值仅用于离线评分。

软先验允许结果偏离名义值，也会影响基参数拟合，并非只在严格零空间内选解。结果接近名义参数可能来自先验的作用，不能据此判断真值恢复精度。

### 8.3 尺寸与物理条件约束什么

除质量为正外，质心惯量还需要满足半正定及主惯量三角条件，例如 $I_1+I_2\ge I_3$；总质量、质心位置和惯量元素也有配置范围。

此外，选定球心 $c$、半径 $R$ 的质量分布包络，要求：

$$
\frac{\operatorname{tr}(I_{\mathrm{com}})}{2}
+m\|r-c\|^2\le mR^2.
$$

$\operatorname{tr}(I_{\mathrm{com}})/2$ 是相对质心的质量二阶矩；加上质心偏移项，得到相对球心的质量二阶矩。如果实际质量分布都在这个球内，就必须满足该关系。代码检查的是二阶矩约束，不是逐点检查质量分布的几何位置；它把质量、尺寸和惯量联系起来，限制“小质量、大惯量”的组合。

当前质心盒来自名义几何包围盒加 10% 裕量。球心取名义质心，半径取几何包围半径与名义惯性所需半径中较大者，再加 10% 裕量。Franka 部分连杆的名义惯量超出纯外观网格球对应的范围，因此使用兼顾几何和名义惯性的包络。这一约束不能证明质量分布符合精确 CAD 几何。

尺寸来源可以用 [inspect_recovery_geometry.py](../scripts/inspect_recovery_geometry.py) 复查；尺度公式与配置解释见 [configs/README.md](../configs/README.md)。

### 8.4 求解器返回之后还要验收

恢复器采用多初值、模拟退火和有界 SLSQP，分阶段增大物理罚项，并使用解析梯度。物理条件通过罚项加入目标函数，求解结束后还需独立检查：

- `solver_success`：局部求解器收敛。
- `physical_feasible`：质量、惯量、总质量等检查通过；开启包络时也要求 `geometry_feasible`。
- `matches_base`：$\|B\pi-\hat\pi_b\|/\max(\|\hat\pi_b\|,1)$ 不超过配置门槛，当前为 0.05。

三者同时满足才是 `success`。多个最终候选优先选择通过全部验收的，再比较目标值；质量、质心和惯量贴界情况另行记录。

通过检查表示该模型满足指定容差和约束，不保证全局最优或解的唯一性。逐连杆参数与真值的差异需另外评估。

实现入口：[identification/recovery.py](../src/robot_model/identification/recovery.py)。

## 9. 怎样判断结果有用

用另一条没有参与训练的轨迹，比较同一批测量数据上的三种预测：

- 名义模型：未辨识前已有参数的预测。
- 基参数模型：直接使用 $\hat\pi_b$ 的预测。
- 恢复模型：使用 $B\pi_{\mathrm{recovered}}$ 的预测。

恢复时有先验与物理条件折中，第三种预测不必与第二种完全相同。逐关节 RMSE 是该关节误差的均方根；总体 RMSE 对所有关节、所有所选样本一起计算，等于各关节 RMSE 平方平均后开根号。

仿真还可以把对象完整真值投影到同一个 $B$，评价基参数误差。实机通常没有完整参数真值，主要通过独立轨迹上的预测误差、误差分布和重复实验评估模型。

当前 `base_error` 与基参数相对真值误差都混合了不同物理单位的参数，没有按不确定度加权。它们是固定参数表示下的整体比较指标，不代表每个参数都具有同样百分比精度，也不是置信区间。

实验结果和局限见 [README](../README.md#辨识结果与图表)。自由空间仿真验证也不能替代实机验证；当前没有真实设备驱动，以及完整的柔性、背隙、电气或热模型。

### 9.1 只做力矩补偿时，是否必须恢复完整参数

给定 $H_b$ 和辨识基参数即可预测力矩，无需先恢复完整参数。通过特定状态下的求值，还能分解出等效惯性矩阵、速度项和静态项。

[identification/decompose.py](../src/robot_model/identification/decompose.py) 的 `DynamicsDecomposer` 实现了这条路径：静止状态得到静态项，逐个施加单位加速度得到惯性矩阵各列，零加速度时扣除静态项得到速度项。

启用摩擦后，速度项包含摩擦；启用电机惯量后，惯性矩阵包含等效电机惯量；若启用恒定偏置，静态项也包含偏置。因此这些等效项不能一概解释为纯刚体的 $M,C,g$。

## 10. 顺着代码看一遍

实际闭环主流程在 [experiments/realistic.py](../src/robot_model/experiments/realistic.py)：

1. `experiments/rrr.py` 或 `experiments/franka.py` 提供模型路径、MDH、连杆名称和实验配置。
2. 构造 `Robot`，计算基参数映射，生成 `Hb_func`。
3. 固定验证轨迹；生成训练候选，按选项优化并闭环筛选。
4. 复用选中候选的测量数据，预处理后交给 `identify()`。
5. 按选项调用 `recover_inertial_parameters()`。
6. 在独立验证数据上评分，保存参数、配置、筛选记录、原始和处理后数据。
7. [plot_experiment.py](figures/plot_experiment.py) 从保存结果生成展示图及报告副本。

`cli/identify_robot.py` 负责参数解析与选机器人；`experiments/` 编排实验流程；`identification/` 负责估计和恢复算法。MuJoCo 数据采集与辨识层分开。接入实机时，需要实现设备控制与采集，并将数据整理为相同的 `MotionData` 格式。

## 建模接口与公式导出

### 小模型与七轴数值路径

`Robot` 是建模入口，按需构建坐标变换、运动学、动力学和导出器。两关节模型的符号计算示例：

```python
from robot_model import Robot

robot = Robot.from_mdh("planar2r", [
    ("0", "1", "0", "q"),
    ("0", "1", "0", "q"),
])
T = robot.forward_kinematics()        # 末端位姿
T1 = robot.frames.T[0]               # 第 1 连杆相对基座，Python 下标从 0 开始
J = robot.kinematics.J[-1]            # 末端雅可比
D = robot.dynamics
tau = D.gen_invdyn()                 # MDH 默认 Khalil
H = D.gen_regressor()                # 完整符号回归矩阵
pi_b = D.calc_base_parms()           # 基参数表达式
Hb = D.gen_base_regressor()          # 已有符号 H 时返回符号 Hb
```

Franka 日常辨识用数值回归，避免展开完整符号矩阵；以下状态数组只示范接口，实际辨识需使用采集数据：

```python
import numpy as np
from robot_model.robots.franka import PANDA_MDH_PARMS

panda = Robot.from_mdh("franka_panda_arm", PANDA_MDH_PARMS)
q = np.zeros(7)
dq = np.zeros(7)
ddq = np.zeros(7)
Hn = panda.dynamics.regressor_numpy(q, dq, ddq, method="park")
panda.dynamics.calc_base_parms(numeric=True, method="park", samples=400)
Hb_func = panda.dynamics.gen_base_regressor()  # 未生成符号 H 时返回数值函数
Hb_n = Hb_func(q, dq, ddq)
```

### 坐标约定与动力学分解

`from_dh()` 是标准 DH，`from_mdh()` 是 Craig MDH；两者参数行都按 `(alpha, a, d, theta)` 输入，用字符串 `q` 标记关节变量。两种约定的连杆坐标系不同，同一张参数表不能直接混用。

`from_poe(..., frame="space")` 默认用基座系旋量，满足 $T=e^{[S_1]q_1}\cdots e^{[S_n]q_n}M$；`frame="body"` 用末端零位系旋量，满足 $T=Me^{[B_1]q_1}\cdots e^{[B_n]q_n}$。互转入口为 `to_dh()`、`to_mdh()`、`to_poe()`；PoE 转 DH/MDH 需要可按目标约定分解的逐连杆零位 $M_i$，`to_poe()` 会保留这些信息。

符号动力学的 `method="park"` 和 `method="lagrange"` 支持三种约定；`method="khalil"` 支持 DH/MDH。标准 DH 和 PoE 默认 Park，MDH 默认 Khalil。数值动力学路径只支持 Park/Khalil，PoE 数值路径使用 Park。

`robot.frames` 提供 `T`、`T_rel`、`R_rel`、`p_rel`、`S_body`，无需按坐标约定更换属性名。动力学分解接口如下：

```python
M = D.gen_inertiamatrix()
c = D.gen_coriolisterm()
g = D.gen_gravityterm()
C = D.gen_coriolismatrix()            # Christoffel 形式，刚体项 c = C dq
S = D.gen_skew_symmetry()             # S = Mdot - 2C，应满足 S + S.T = 0
```

上述两关节示例没有开启摩擦或电机选项。包含附加项时，应按具体接口定义解释各项，不能把所有速度相关力矩都归为刚体科氏力。

### 导出公式与可调用代码

```python
robot.export.show(["M"])                         # Notebook 中渲染
robot.export.latex("planar2r.tex")                # 可用 pdflatex 编译
robot.export.python("planar2r_model.py")          # NumPy 函数
robot.export.c("planar2r_model.c")                # C99 函数
text = robot.export.latex(keys=["M", "g"])       # 省略路径则返回文本
```

可选 `keys` 为 `T J M c C g tau H Hb baseparms`，还支持逐连杆的 `T_i`、`T_rel_i`、`J_i`，其中下标从 1 开始。不指定时导出 `T J M c g tau`；完整回归矩阵 `H` 及基回归矩阵 `Hb` 需要显式选择。导出 `H/Hb` 会构造符号表达式，与前面的数值回归路径不同。

LaTeX 默认 `simplify="auto"`，仅化简表达式树节点数不超过 400 的量；排版按长度选择矩阵、逐元素或公共子表达式。默认文档用 `pdflatex`，`cjk=True` 保留中文正文并要求 `xelatex` 与 `ctex`。Python/C 默认 `simplify=False, cse=True`，避免昂贵化简并提取公共子表达式。

生成的代码按需要接收一维数组 `q`、`dq`、`ddq`、`parms`，具体函数签名见生成文件。`parms` 是完整的线性动力学参数，不是直接拼接的质量、质心和质心惯量；其顺序与 `robot.symbols.dynparms()` 一致，文件头逐项列出下标。参数表示见本文第 3 节。

实现入口：[export/exporter.py](../src/robot_model/export/exporter.py)、[export/collect.py](../src/robot_model/export/collect.py)。

### 与 MuJoCo 对照及完整符号导出

两轴 `planar` 测试验证约定互转及不同动力学方法的一致性；RRR、Franka 还用 MuJoCo 做独立的运动学与动力学对照。比较纯刚体力矩前，必须对齐连杆帧，并去掉电机惯量、阻尼、摩擦和接触等附加影响。

RRR 的模型 [rrr_arm.xml](../src/robot_model/assets/models/rrr_arm/rrr_arm.xml) 已按裸刚体配置；Franka 的 [panda_arm.xml](../src/robot_model/assets/models/franka_panda/panda_arm.xml) 通过 `strip_to_rigid_body()` 处理。[MujocoReference](../src/robot_model/simulation/reference.py) 把主轴惯量转换为连杆坐标轴下的质心惯量，再构造线性参数，并封装 `mj_inverse`。

[Franka 符号导出测试](../tests/franka/test_franka_export.py) 检查生成代码的力矩、惯性矩阵、运动学及完整/基参数回归形式。该模块标记为 `slow`，常用的 `-m "not slow"` 会跳过它；单独运行：

```bash
python -m pytest -q -m slow tests/franka/test_franka_export.py
```

[Franka 完整流程 Notebook](../notebooks/franka/modeling.ipynb) 包含相同导出链路，生成文件放在 `outputs/franka_symbolic/`。这条路径验证七轴符号公式可用，日常辨识仍优先使用数值回归。

## 辨识接口与实验选项

### 在代码中调用辨识与恢复

下面假定已准备好同一套 RRR 模型的 `robot`、`Hb_func`、处理后训练数据 `train_data`，以及从未扰动模型取得的完整名义参数 `nominal_full`。Franka 需同时换成对应模型与配置。

~~~python
import json
from pathlib import Path
from robot_model.identification import (
    identify, recover_inertial_parameters, RecoveryConfig,
)

fit = identify(Hb_func, train_data)
config = RecoveryConfig(json.loads(
    Path("configs/rrr/recovery.json").read_text(encoding="utf-8")
))
recovered = recover_inertial_parameters(
    robot, fit.parms, nominal_parameters=nominal_full, config=config,
)
if recovered.success:
    masses = recovered.masses
    centers = recovered.centers_of_mass
    inertias = recovered.inertias_com
report = recovered.to_dict()  # 包含失败时的诊断信息
~~~

`identify()` 接收基回归函数；`nominal_full` 必须按同一模型的 `robot.symbols.dynparms()` 顺序排列。通用 `RecoveryConfig()` 默认不启用名义正则，以上显式加载仓库配置才会同时启用先验与包络。字段与运行命令见 [配置说明](../configs/README.md)。

恢复算法改编自 FrankaEmikaPandaDynModel 的两步参数恢复分支；Python/SciPy 求解器、通用映射与本仓库的先验/包络扩展不保证复现原 MATLAB 优化数值。

### 多场景筛选与周期平均

CLI 支持以下选项，实际设置保存在报告中。完整命令见 [README](../README.md#从哪里开始)。

- `--optimize-excitation` 用 SLSQP 优化傅里叶系数的 `log(cond(W))`，默认两个初值、各最多 20 次迭代。通过连续时间约束和独立网格复查后，改善的候选才加入闭环试跑；达到迭代上限仍报告未收敛。RRR 默认开、Franka 默认关。
- `--robust-seeds 0 1 2` 让所有候选接受相同的多个对象扰动与噪声场景；全部场景通过才合格，按最坏场景排名。至少两个不同种子，第一个须等于实验 `seed`。最终只用第一个场景辨识，不能混合不同对象的数据估计一组参数。
- `--average-periods` 至少需要记录两个周期。先补偿已知延迟，再用共同相位区间插值对齐并平均，不外推；每周期至少 20 点，共同覆盖不少于 95%。随后滤波、微分和剔除低速。样本保留率折算周期数，样本数量门槛仍按真实剩余相位点数检查。
- 多场景筛选和周期平均默认关闭；关闭闭环筛选时不会运行候选优化，也不能使用多场景筛选。有限场景全部通过不代表对任意扰动都成立，周期平均也不能消除系统偏置或漂移。

`optimization.json` 保存优化诊断，`screening.json` 保存候选及场景评分；`train-processed.npz` 和 `validation-processed.npz` 才是进入辨识与测量域验证的数据。启用平均时另存 `*-averaged.npz`，其中 `std_*` 是周期间离散度，不是参数置信区间。

原始 `train.npz` / `validation.npz` 的时间戳是接收时刻，包含测量量、`truth_*`、`command`、`desired_q`；已知延迟只在预处理时重索引。`trajectories.npz` 保存可重放的系数。`report.json` 的运动诊断统计预热后的全部物理步，区别于降采样的测量日志。
