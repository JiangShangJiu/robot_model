# 演示产物

本目录保存三个仿真演示的指标与图表（`metrics.json` + PNG），纳入 Git，
供个人主页等外部仓库直接引用。叙述性文档见 [docs/demos.md](../docs/demos.md)。

| 目录 | 演示 | 重新生成命令 |
| --- | --- | --- |
| `hardware_id/` | 仿真实机辨识闭环（辨识前后对照） | `robot-demo-hardware-id --output-dir artifacts/hardware_id` |
| `contact_safety/` | 接触力控失稳回放（无/有能量罐） | `robot-demo-contact --output-dir artifacts/contact_safety` |
| `vla_bridge/` | VLA 动作块 → 底层阻抗环桥接 | `robot-demo-vla --output-dir artifacts/vla_bridge` |

未安装入口脚本时，可用 `PYTHONPATH=src python -m robot_model.cli.demo_hardware_id` 等方式运行。
三个演示都是确定种子的：相同依赖版本下重跑应得到相同指标；
跨 MuJoCo/NumPy 版本数值可能有微小差异。

更新约定：修改演示代码或默认参数后，必须重新生成对应目录的全部产物，
并同步 `docs/demos.md` 与 README 中引用的数字。
