"""Saved recovery keeps the experiment immutable and scores only after solving."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robot_model.cli import recover_inertia as cli
from robot_model.identification import RecoveryConfig


def _snapshot(directory):
    return {path.relative_to(directory): path.read_bytes()
            for path in directory.rglob('*') if path.is_file()}


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    source = tmp_path / 'original'
    source.mkdir()
    destination = tmp_path / 'recovered'
    nominal = np.array([10.0, 20.0, 30.0])
    report = {
        'status': 'inertial_recovery_failed',
        'robot': 'test_robot',
        'fit': {'identifiable': True, 'rank': 2, 'n_parms': 2,
                'parameters': [3.0, -1.0],
                'base_parameter_expressions': ['base_a', 'base_b'],
                'residual_rms_Nm': 0.0123},
        'screening': {'selected_candidate_id': 4, 'candidates': [1, 4, 7]},
        'optimization': {'enabled': False},
        'train': {'retained_samples': 3},
        'validation': {'trajectory_seed': 99},
        'config': {'seed': 17},
        'experiment': {'periods': 2},
        'enhancements': {'inertial_recovery': False, 'periodic_average': False},
        'inertial_recovery': {'nominal_parameters': nominal.tolist(),
                             'success': False, 'old_candidate': [999.0]},
        'validation_rms_Nm': {'measured_fitted': [0.1, 0.2],
                              'measured_nominal': [4.0, 5.0],
                              'measured_recovered': [99.0, 99.0],
                              'truth_recovered': [88.0, 88.0]},
        'warnings': ['保留这条采集提示', '惯性参数恢复未通过旧验收'],
    }
    (source / 'report.json').write_text(json.dumps(report, ensure_ascii=False))
    (source / 'inertial-parameters.json').write_text('{"old_candidate": true}')
    (source / 'recovery-config.json').write_text('{"old_config": true}')
    (source / 'nested').mkdir()
    (source / 'nested' / 'notes.txt').write_text('Keep this original artifact.')
    t = np.arange(3.0)
    q = np.column_stack((np.arange(3.0), np.zeros(3)))
    dq = np.column_stack((np.zeros(3), np.arange(1.0, 4.0)))
    ddq = np.full_like(q, 0.25)
    # Hb = [[1, q[0]], [dq[1], 1]], recovered parameters = [2, -0.5].
    # These known observations give per-joint RMS errors [1, 2] and [3, 4].
    measured_tau = np.array([[3.0, 3.5], [0.5, 1.5], [2.0, 7.5]])
    truth_q, truth_dq = q + [3.0, 0.0], dq + [0.0, 3.0]
    truth_tau = np.array([[-2.5, 11.5], [3.0, 13.5], [-3.5, 7.5]])
    np.savez(source / 'validation-processed.npz', t=t, q=q, dq=dq, ddq=ddq, tau=measured_tau)
    np.savez(source / 'validation.npz', t=t, q=q * 100, dq=dq * 100,
             ddq=ddq * 100, tau=measured_tau * 100,
             truth_q=truth_q, truth_dq=truth_dq, truth_ddq=ddq, truth_tau=truth_tau)
    np.savez(source / 'train.npz', q=q, tau=measured_tau)
    np.savez(source / 'train-processed.npz', q=q, tau=measured_tau)
    baseline = _snapshot(source)
    events = []
    state = {'solved': False, 'solver_calls': 0}
    nominal_xml = tmp_path / 'unperturbed.xml'
    nominal_xml.write_text('<mujoco model="unperturbed-nominal"/>')
    setup = SimpleNamespace(name='test_robot', mdh_parms=[('geometry',)],
                            body_names=['link_a', 'link_b'],
                            base_parms_samples=23, xml_path=nominal_xml)
    model = SimpleNamespace(full_parameters=nominal.copy())
    dynamics = SimpleNamespace(
        baseparms=['base_a', 'base_b'],
        Hb_func=lambda q, dq, ddq: np.array([[1.0, q[0]], [dq[1], 1.0]]),
    )

    def calc_base_parms(*, samples):
        assert samples == setup.base_parms_samples
        events.append('base_mapping')

    dynamics.calc_base_parms = calc_base_parms
    dynamics.gen_base_regressor = lambda: events.append('base_regressor')
    robot = SimpleNamespace(dynamics=dynamics)

    def make_robot(name, parms, **kwargs):
        assert name == setup.name and parms == setup.mdh_parms
        assert kwargs == {'frictionmodel': {'viscous', 'Coulomb'},
                          'driveinertiamodel': 'simplified'}
        return robot

    def load_setup(module_name):
        assert module_name == 'robot_model.experiments.rrr'
        return SimpleNamespace(setup=lambda: setup)

    def load_nominal(path):
        assert Path(path) == nominal_xml
        assert nominal_xml.read_text() == '<mujoco model="unperturbed-nominal"/>'
        events.append('load_nominal_mjcf')
        return model

    def pack_nominal(actual_robot, actual_model, body_names):
        assert actual_robot is robot and actual_model is model
        assert body_names == setup.body_names
        events.append('pack_nominal')
        return actual_model.full_parameters.copy()

    class Candidate:
        success = True
        base_parameters = np.array([2.0, -0.5])

        def to_dict(self):
            return {'success': self.success, 'new_candidate': True,
                    'base_parameters': self.base_parameters.tolist()}

    candidate = Candidate()
    config = RecoveryConfig(runs=1)

    def solve(actual_robot, base, *, nominal_parameters, config, progress):
        assert actual_robot is robot
        np.testing.assert_array_equal(base, report['fit']['parameters'])
        np.testing.assert_array_equal(nominal_parameters, nominal)
        assert not state['solved']
        state['solver_calls'] += 1
        events.append('solver_finished')
        state['solved'] = True
        return candidate

    original_load = np.load

    def load_after_solving(path, *args, **kwargs):
        assert state['solved'], 'Validation/truth data must not enter solver construction or priors'
        events.append(('read_npz', Path(path).name))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(cli, 'Robot', SimpleNamespace(from_mdh=make_robot))
    monkeypatch.setattr(cli, 'import_module', load_setup)
    monkeypatch.setattr(cli, 'nominal_recovery_parameters', pack_nominal)
    monkeypatch.setattr(cli, 'recover_inertial_parameters', solve)
    monkeypatch.setattr(cli.np, 'load', load_after_solving)
    monkeypatch.setitem(sys.modules, 'mujoco', SimpleNamespace(
        MjModel=SimpleNamespace(from_xml_path=load_nominal)))
    return SimpleNamespace(source=source, destination=destination, report=report,
                           baseline=baseline, config=config, candidate=candidate,
                           nominal=nominal, state=state, events=events)


@pytest.mark.parametrize('success', [True, False])
def test_recovery_preserves_source_and_fit_and_scores_independent_data(saved_run, success):
    run = saved_run
    run.candidate.success = success
    result = cli.run_saved_recovery('rrr', run.source, run.destination, run.config)
    assert _snapshot(run.source) == run.baseline
    assert run.state['solver_calls'] == 1
    assert run.events == ['base_mapping', 'base_regressor', 'load_nominal_mjcf',
                          'pack_nominal', 'solver_finished',
                          ('read_npz', 'validation-processed.npz'),
                          ('read_npz', 'validation.npz')]
    for key in ('fit', 'screening', 'optimization', 'train', 'validation', 'config', 'experiment'):
        assert result[key] == run.report[key]
    assert result['validation_rms_Nm']['measured_fitted'] == [0.1, 0.2]
    assert result['validation_rms_Nm']['measured_nominal'] == [4.0, 5.0]
    np.testing.assert_allclose(result['validation_rms_Nm']['measured_recovered'], [1.0, 2.0])
    np.testing.assert_allclose(result['validation_rms_Nm']['truth_recovered'], [3.0, 4.0])
    assert result['status'] == ('completed' if success else 'inertial_recovery_failed')
    assert result['warnings'][0] == '保留这条采集提示'
    assert '惯性参数恢复未通过旧验收' not in result['warnings']
    assert len(result['warnings']) == (1 if success else 2)
    assert result['recovery_update']['source_report_sha256'] == hashlib.sha256(
        run.baseline[Path('report.json')]).hexdigest()
    assert result['recovery_update']['config'] == run.config.to_dict()
    assert result['inertial_recovery']['nominal_parameters'] == run.nominal.tolist()
    assert result['inertial_recovery']['prior_source'] == 'unperturbed repository MJCF (nominal model only)'
    assert json.loads((run.destination / 'report.json').read_text()) == result
    assert json.loads((run.destination / 'inertial-parameters.json').read_text()) == result['inertial_recovery']
    assert json.loads((run.destination / 'recovery-config.json').read_text()) == run.config.to_dict()
    replaced = {'report.json', 'inertial-parameters.json', 'recovery-config.json'}
    for path, original_bytes in run.baseline.items():
        if str(path) not in replaced:
            assert (run.destination / path).read_bytes() == original_bytes


@pytest.mark.parametrize('kind', ['same', 'existing', 'inside_source', 'symlink_same', 'symlink_inside'])
def test_rejects_output_that_could_modify_original_experiment(saved_run, monkeypatch, tmp_path, kind):
    run = saved_run
    if kind == 'same':
        output = run.source
    elif kind == 'existing':
        output = tmp_path / 'existing'
        output.mkdir()
        (output / 'keep.txt').write_text('do not replace')
    elif kind == 'inside_source':
        output = run.source / 'new-result'
    else:
        alias = tmp_path / 'source-alias'
        alias.symlink_to(run.source, target_is_directory=True)
        output = alias if kind == 'symlink_same' else alias / 'new-result'

    def forbidden_solver(*args, **kwargs):
        pytest.fail('Unsafe output must be rejected before solving or copying')

    monkeypatch.setattr(cli, 'recover_inertial_parameters', forbidden_solver)
    with pytest.raises(ValueError, match='输出|目录'):
        cli.run_saved_recovery('rrr', run.source, output, run.config)
    assert _snapshot(run.source) == run.baseline
    assert run.events == []
    if kind == 'existing':
        assert (output / 'keep.txt').read_text() == 'do not replace'


@pytest.mark.parametrize('bad_input', ['base_expressions', 'rank', 'nominal'])
def test_rejects_incompatible_saved_fit_before_solver_or_scoring(saved_run, bad_input):
    run = saved_run
    report = deepcopy(run.report)
    if bad_input == 'base_expressions':
        report['fit']['base_parameter_expressions'].reverse()
    elif bad_input == 'rank':
        report['fit']['identifiable'] = False
    else:
        report['inertial_recovery']['nominal_parameters'][0] += 100
    (run.source / 'report.json').write_text(json.dumps(report))
    before = _snapshot(run.source)
    with pytest.raises(ValueError):
        cli.run_saved_recovery('rrr', run.source, run.destination, run.config)
    assert run.state['solver_calls'] == 0
    assert not any(isinstance(event, tuple) and event[0] == 'read_npz' for event in run.events)
    assert not run.destination.exists()
    assert _snapshot(run.source) == before


def test_failed_cli_recovery_saves_diagnostics_before_nonzero_exit(saved_run, monkeypatch, tmp_path):
    run = saved_run
    run.candidate.success = False
    config_path = tmp_path / 'recovery.json'
    config_path.write_text(json.dumps(run.config.to_dict()))
    monkeypatch.setattr(sys, 'argv', [
        'recover-inertia', '--robot', 'rrr', '--input-dir', str(run.source),
        '--output-dir', str(run.destination), '--recovery-config', str(config_path),
    ])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 3
    saved = json.loads((run.destination / 'report.json').read_text())
    assert saved['status'] == 'inertial_recovery_failed'
    assert saved['fit'] == run.report['fit']
    assert saved['inertial_recovery']['new_candidate'] is True
    assert _snapshot(run.source) == run.baseline
