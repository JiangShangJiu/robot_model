# Notebooks

这些 Notebook 用于公式推导、模型对照和理想数据辨识。README 的闭环实验由 `robot-identify` 运行，包含对象扰动、传感器误差和执行器动态，两类结果需要分开看。

## 从哪里开始

1. [两轴符号建模](planar/symbolic_basics.ipynb)：查看 MDH、正运动学、雅可比、惯性参数和回归矩阵。无需 MuJoCo，也不写文件。
2. [RRR 基参数辨识](rrr/identification.ipynb)：比较符号/数值回归，用理想逆动力学和力矩噪声数据拟合参数，再分解 `M/c/g`。
3. [Franka 理想数据辨识](franka/identification.ipynb)：七轴数值回归，模型到 `link7`，不含夹爪。符号对照默认关闭。

## 建模、导出与专项检查

- [RRR 建模与导出](rrr/modeling.ipynb)：运动学、动力学公式、Python/C/LaTeX 导出、MuJoCo 对照和沿轨迹的力矩分解。写入 `outputs/rrr_symbolic/`。
- [Franka 建模与导出](franka/modeling.ipynb)：七轴完整符号推导与代码导出，计算较慢。写入 `outputs/franka_symbolic/`，也会在其 `identification/` 子目录保存理想辨识数据。
- [Franka 激励轨迹](franka/excitation.ipynb)：比较随机候选、论文 N=5 系数与 SLSQP 优化。这是 43 个刚体基参数上的解析对照，区别于 README 的 62 参数闭环实验；优化单元较慢，结果写入 `outputs/franka_symbolic/excitation/`。
- [Franka 运动学](franka/kinematics.ipynb)：MDH/DH/PoE 转换、逐连杆位姿、几何与解析雅可比，以及 MuJoCo 对照。
- [Franka 可视化](franka/visualization.ipynb)：离屏渲染、多个视角和连杆帧对照，不打开交互窗口。需要可用的 OpenGL 后端；无桌面环境可在启动内核前设置 `MUJOCO_GL=egl`。

两份 `modeling.ipynb` 的完整惯性恢复默认 `RUN_RECOVERY=False`。开启后读取各机器人的恢复配置，用独立加载的名义 MJCF 提供初值和先验，并缩短求解预算。这些是同模型理想数据示例；正式闭环结果与恢复记录见 [README](../README.md)。

## 运行

在仓库根目录安装依赖，Jupyter 或编辑器选用同一个 Python 环境：

```bash
python -m pip install -e ".[sim,ident,viz,dev]"
```

从仓库或 Notebook 所在目录启动均可，初始化单元会向上查找项目根目录。重新开始时先重启内核，再按顺序运行；不要依赖另一份 Notebook 中的变量。

符号导出和激励优化耗时较长，运行前先看该节说明和开关。建模、激励本会覆盖各自输出目录中的同名文件，要保留一次实验时先更换输出目录。原有图表和有参考价值的计算输出已保留；修改过代码的单元清除了旧输出，需执行后查看当前结果。

完整闭环采集使用 `robot-identify --robot rrr` 或 `robot-identify --robot franka`，运行示例见 [项目入口](../README.md#从哪里开始)，方法见 [工程原理](../docs/principles.md)。
