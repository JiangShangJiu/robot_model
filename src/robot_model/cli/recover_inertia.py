"""使用已保存的基参数重做惯性恢复，保留采集、筛选与拟合结果。

用法::

    python -m robot_model.cli.recover_inertia --robot rrr \
        --input-dir outputs/showcase-fresh/rrr \
        --recovery-config configs/rrr/recovery.json \
        --output-dir outputs/showcase-prior/rrr

仅从未扰动 MJCF 构造名义先验。验证/真值数据在求解完成后才读取，
只用于评分；输出必须为新目录，原实验不会被覆盖。
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path
import shutil

import numpy as np

from robot_model import Robot
from robot_model.data import MotionData
from robot_model.experiments.realistic import nominal_recovery_parameters
from robot_model.experiments.pipeline import torque_rms
from robot_model.identification import RecoveryConfig, predict_torque, recover_inertial_parameters


def _read_motion(path, *, truth=False):
    with np.load(path) as saved:
        prefix = 'truth_' if truth else ''
        return MotionData(saved['t'], *(saved[prefix + name] for name in ('q', 'dq', 'ddq', 'tau')))


def run_saved_recovery(robot_name, input_dir, output_dir, config):
    """仅替换完整惯性恢复与其评分，沿用原实验的独立验证集。"""
    import mujoco

    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir == input_dir or input_dir in output_dir.parents or output_dir.exists():
        raise ValueError('输出必须为输入目录之外、尚不存在的新目录，以保留原实验')
    raw_report = (input_dir / 'report.json').read_bytes()
    report = json.loads(raw_report)
    if 'fit' not in report or not report['fit']['identifiable']:
        raise ValueError('输入必须包含满秩的已辨识基参数')
    for name in ('validation-processed.npz', 'validation.npz'):
        if not (input_dir / name).is_file():
            raise ValueError(f'缺少独立验证数据：{name}')
    setup = import_module(f'robot_model.experiments.{robot_name}').setup()
    robot = Robot.from_mdh(setup.name, setup.mdh_parms,
        frictionmodel={'viscous', 'Coulomb'}, driveinertiamodel='simplified')
    dyn = robot.dynamics
    dyn.calc_base_parms(samples=setup.base_parms_samples)
    dyn.gen_base_regressor()
    if report['fit']['base_parameter_expressions'] != [str(p) for p in dyn.baseparms]:
        raise ValueError('当前模型的基参数定义与保存实验不一致，不能复用参数')
    nominal_model = mujoco.MjModel.from_xml_path(str(setup.xml_path))
    nominal = nominal_recovery_parameters(robot, nominal_model, setup.body_names)
    previous = report.get('inertial_recovery', {})
    if 'nominal_parameters' in previous:
        if not np.allclose(nominal, previous['nominal_parameters'], rtol=1e-10, atol=1e-12):
            raise ValueError('当前名义模型与保存实验不一致，不能复用名义先验')

    def progress(record):
        print(f"run {record['run']+1}, stage {record['stage']+1}: "
              f"base residual={record['base_residual_norm']:.6g}, "
              f"physical={record['physical_feasible']}, solver={record['solver_success']}", flush=True)

    result = recover_inertial_parameters(robot, np.asarray(report['fit']['parameters']),
        nominal_parameters=nominal, config=config, progress=progress)
    payload = result.to_dict()
    payload['prior_source'] = 'unperturbed repository MJCF (nominal model only)'
    payload['nominal_parameters'] = nominal.tolist()
    # 真值和留出验证数据只在求解结束后读取；不参与初值、尺度、约束或候选选择。
    validation = _read_motion(input_dir / 'validation-processed.npz')
    truth = _read_motion(input_dir / 'validation.npz', truth=True)
    scores = {
        'measured_recovered': torque_rms(predict_torque(dyn.Hb_func, validation, result.base_parameters), validation.tau),
        'truth_recovered': torque_rms(predict_torque(dyn.Hb_func, truth, result.base_parameters), truth.tau),
    }
    payload['validation_rms_Nm'] = scores
    report['inertial_recovery'] = payload
    report['validation_rms_Nm'].update(scores)
    report['enhancements']['inertial_recovery'] = True
    report['status'] = 'completed' if result.success else 'inertial_recovery_failed'
    report['recovery_update'] = {
        'source_directory': str(input_dir),
        'source_report_sha256': hashlib.sha256(raw_report).hexdigest(),
        'operation': 'inertia recovery only; acquisition, screening, base fit and validation trajectory unchanged',
        'config': config.to_dict(),
    }
    report['warnings'] = [warning for warning in report.get('warnings', [])
                          if not warning.startswith('惯性参数恢复未通过')]
    if not result.success:
        report['warnings'].append('惯性参数恢复未通过验收；候选仅供诊断，原辨识基参数保留')
    # 独立副本包含完整原始测量，便于现有绘图/分析脚本直接复用。
    shutil.copytree(input_dir, output_dir)
    for name, data in (('report.json', report), ('inertial-parameters.json', payload),
                       ('recovery-config.json', config.to_dict())):
        (output_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f"恢复{'通过' if result.success else '未通过'}；报告保存到 {output_dir}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description='从保存的基参数重做惯性恢复，不重新采集或拟合')
    parser.add_argument('--robot', choices=('rrr', 'franka'), required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--recovery-config', type=Path, required=True)
    args = parser.parse_args()
    try:
        config = RecoveryConfig(**json.loads(args.recovery_config.read_text(encoding='utf-8')))
        report = run_saved_recovery(args.robot, args.input_dir, args.output_dir, config)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    if report['status'] != 'completed':
        parser.exit(3, '惯性恢复未通过验收；候选和原基参数已保存至输出目录\n')


if __name__ == '__main__':
    main()
