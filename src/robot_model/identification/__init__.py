"""动力学参数辨识：估计、分解与完整惯性参数恢复。

轨迹、激励与仿真分别从 ``robot_model.data``、``robot_model.excitation``、
``robot_model.simulation`` 导入，不要从本包再取。
"""

from .decompose import DynamicsDecomposer, RigidBodyTerms
from .estimator import (
    IdentificationResult,
    identify,
    parameter_error,
    predict_torque,
    stack_regressor,
)
from .recovery import (
    InertialRecoveryResult,
    RecoveryConfig,
    pack_inertial_parameters,
    recover_inertial_parameters,
)

__all__ = [
    "DynamicsDecomposer",
    "IdentificationResult",
    "InertialRecoveryResult",
    "RecoveryConfig",
    "RigidBodyTerms",
    "identify",
    "pack_inertial_parameters",
    "parameter_error",
    "predict_torque",
    "recover_inertial_parameters",
    "stack_regressor",
]
