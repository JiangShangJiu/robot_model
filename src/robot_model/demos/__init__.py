"""面向展示的端到端仿真演示：辨识闭环、接触力控稳定化、VLA 桥接。

每个演示都是独立可运行的流程（见 ``robot_model.cli.demo_*``），
复用库内的激励、仿真、辨识模块；产物（PNG + metrics.json）默认
写入仓库根目录 ``artifacts/<demo>/``，便于个人主页直接引用或重新生成。
导入本模块不加载 MuJoCo。
"""

__all__ = ["hardware_id", "contact_safety", "vla_bridge"]
