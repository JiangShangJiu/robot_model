"""MuJoCo 数据采集、传感器/执行器模型与动力学参考工具。

导入本模块不会加载 MuJoCo；只有创建或调用仿真对象时才需要可选依赖。
"""

from .hardware import ActuatorConfig, OnlineSensorConfig, OnlineSensors, TorqueActuator
from .realistic import RealismConfig, RealisticMujocoSource, SimulationRun
from .reference import LinkInertia, MujocoReference, strip_to_rigid_body
from .sensors import SensorModel, TYPICAL_SERVO
from .tracking import MujocoTrackingSource

__all__ = [
    "ActuatorConfig", "OnlineSensorConfig", "OnlineSensors", "TorqueActuator",
    "RealismConfig", "RealisticMujocoSource", "SimulationRun",
    "LinkInertia", "MujocoReference", "strip_to_rigid_body",
    "SensorModel", "TYPICAL_SERVO", "MujocoTrackingSource",
]
