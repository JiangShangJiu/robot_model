"""惯性恢复是基参数辨识后的独立步骤，先验与评分真值必须隔离。"""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robot_model.data import MotionData
from robot_model.identification import RecoveryConfig, predict_torque
from robot_model.robots import RRR_BODY_NAMES
from robot_model.experiments import realistic, rrr


@pytest.mark.parametrize('arguments,message', [
    (['--recover-inertia'], '仅用于 --realistic'),
    (['--realistic', '--recovery-config', 'missing.json'], '需要 --recover-inertia'),
])
def test_recovery_cli_requires_explicit_mode(monkeypatch, capsys, arguments, message):
    from robot_model.cli.identify_rrr import main
    monkeypatch.setattr(sys, 'argv', ['robot-identify-rrr', *arguments])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_reports_failed_recovery_with_nonzero_exit(monkeypatch, capsys, tmp_path):
    pytest.importorskip('mujoco')
    from robot_model.cli.identify_rrr import main
    captured = {}
    def failed(config, **kwargs):
        captured.update(kwargs)
        return {'status': 'inertial_recovery_failed'}
    monkeypatch.setattr(rrr, 'run_experiment', failed)
    config = tmp_path / 'recover.json'
    config.write_text(json.dumps(RecoveryConfig().to_dict()))
    monkeypatch.setattr(sys, 'argv', [
        'robot-identify-rrr', '--realistic', '--recover-inertia',
        '--recovery-config', str(config), '--output-dir', str(tmp_path),
    ])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 3
    assert captured['recover']
    assert isinstance(captured['recovery_config'], RecoveryConfig)
    assert '原辨识基参数保留' in capsys.readouterr().err


def test_recovery_config_cannot_be_silently_ignored():
    pytest.importorskip('mujoco')
    with pytest.raises(ValueError, match='recover=True'):
        rrr.run_experiment(rrr.default_config(), recovery_config=RecoveryConfig())


@pytest.mark.parametrize('success', [True, False])
def test_pipeline_preserves_fit_and_saves_recovery_diagnostics(monkeypatch, tmp_path, success):
    pytest.importorskip('mujoco')
    pytest.importorskip('scipy')
    captured = {}
    original_source = realistic.RealisticMujocoSource
    def source_factory(*args, **kwargs):
        source = original_source(*args, **kwargs)
        captured['source'] = source
        return source
    monkeypatch.setattr(realistic, 'RealisticMujocoSource', source_factory)

    # 核心求解器有单独的数值测试；这里故意改变候选基参数，检查下游是否正确评分、保存。
    def recovery(robot, target, *, nominal_parameters, config):
        captured.update(robot=robot, target=target.copy(), nominal=nominal_parameters.copy())
        base = target * .9
        payload = dict(success=success, full_parameters=nominal_parameters.tolist(),
            base_parameters=base.tolist(), masses=[1., 1., 1.],
            centers_of_mass=[[0., 0., 0.]]*3,
            inertias_com=[np.eye(3).tolist()]*3, extras={}, links=[],
            report={'success': success, 'solver_success': success,
                    'physical_feasible': True, 'matches_base': success,
                    'base_error': .1, 'message': 'pipeline fixture'})
        return SimpleNamespace(success=success, base_parameters=base,
                               to_dict=lambda: payload)
    monkeypatch.setattr(realistic, 'recover_inertial_parameters', recovery)
    config = RecoveryConfig()
    report = rrr.run_experiment(rrr.default_config(), trials=1, periods=1,
        screen=False, recover=True, recovery_config=config, output_dir=tmp_path)
    assert report['status'] == ('completed' if success else 'inertial_recovery_failed')
    assert np.array_equal(report['fit']['parameters'], captured['target'])
    assert report['inertial_recovery']['success'] is success
    assert report['inertial_recovery']['prior_source'].startswith('source.nominal_model')
    robot, source = captured['robot'], captured['source']
    assert np.array_equal(captured['nominal'], realistic.nominal_recovery_parameters(robot, source.nominal_model, RRR_BODY_NAMES))
    assert not np.allclose(captured['nominal'], realistic.nominal_recovery_parameters(robot, source.model, RRR_BODY_NAMES))
    names = [str(symbol) for symbol in robot.symbols.dynparms()]
    for name in ('Ia_1', 'fv_1', 'fc_1'):
        assert captured['nominal'][names.index(name)] == 0
    # 改变仿真对象的隐藏质量/摩擦也不会改变恢复使用的名义先验。
    source.model.body_mass[:] *= 2
    source.model.dof_damping[:] = 99
    assert np.array_equal(captured['nominal'], realistic.nominal_recovery_parameters(robot, source.nominal_model, RRR_BODY_NAMES))

    saved = json.loads((tmp_path / 'inertial-parameters.json').read_text())
    assert saved == report['inertial_recovery']
    assert json.loads((tmp_path / 'recovery-config.json').read_text()) == json.loads(json.dumps(config.to_dict()))
    assert json.loads((tmp_path / 'report.json').read_text()) == json.loads(json.dumps(report))
    with np.load(tmp_path / 'validation-processed.npz') as values:
        validation = MotionData(**{key: values[key] for key in ('t','q','dq','ddq','tau')})
    expected = np.sqrt(np.mean((predict_torque(
        robot.dynamics.Hb_func, validation, captured['target']*.9)-validation.tau)**2, axis=0))
    assert np.allclose(report['validation_rms_Nm']['measured_recovered'], expected)
    with np.load(tmp_path / 'validation.npz') as values:
        truth = MotionData(t=values['t'], **{
            key: values['truth_'+key] for key in ('q','dq','ddq','tau')})
    expected_truth = np.sqrt(np.mean((predict_torque(
        robot.dynamics.Hb_func, truth, captured['target']*.9)-truth.tau)**2, axis=0))
    assert np.allclose(report['validation_rms_Nm']['truth_recovered'], expected_truth)
    assert any('惯性参数恢复未通过' in warning for warning in report['warnings']) is not success
