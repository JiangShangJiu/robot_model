"""惯性参数恢复的坐标转换、梯度、物理验收及数值求解；不依赖 MuJoCo。"""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from robot_model.identification.recovery import (
    RecoveryConfig,
    _Layout,
    _audit,
    _bounds,
    _geometry_violations,
    _nominal_prior,
    _prior_scales,
    _physical_penalty,
    pack_inertial_parameters,
    recover_inertial_parameters,
)
from robot_model.utils.symbols import RobotSymbols


def _robot(dof=1, *, order="khalil", extras=False, observed=None):
    symbols = RobotSymbols(
        "recovery_test", dof, dyn_parms_order=order,
        frictionmodel={"viscous", "Coulomb", "offset"} if extras else None,
        driveinertiamodel="simplified" if extras else None,
    )
    size = len(symbols.dynparms())
    selected = list(range(size)) if observed is None else list(observed)
    dependent = [i for i in range(size) if i not in selected]
    return SimpleNamespace(
        symbols=symbols,
        dynamics=SimpleNamespace(
            Pb=np.eye(size)[:, selected],
            Pd=np.eye(size)[:, dependent],
            Kd=np.zeros((len(selected), len(dependent))),
        ),
    )


def _central_jacobian(function, x, step=1e-6):
    x = np.asarray(x, float)
    perturbations = np.eye(x.size) * step
    return np.column_stack([
        (np.asarray(function(x + delta)) - np.asarray(function(x - delta))) / (2 * step)
        for delta in perturbations
    ])


def _physical_fixture():
    masses = np.array([1.5, 2.3])
    centers = np.array([[0.01, -0.02, 0.03], [-0.04, 0.05, 0.02]])
    inertias = np.array([
        [[0.014, 0.001, 0], [0.001, 0.019, 0.0005], [0, 0.0005, 0.025]],
        [[0.036, -0.002, 0.001], [-0.002, 0.042, -0.003], [0.001, -0.003, 0.05]],
    ])
    return masses, centers, inertias


@pytest.mark.parametrize("order", ["khalil", "mass first"])
def test_pack_preserves_frame_convention_and_symbol_order_with_extras(order):
    robot = _robot(2, order=order, extras=True)
    masses, centers, inertias = _physical_fixture()
    extras = {"Ia_1": 0.004, "fv_1": 0.2, "fc_1": 0.08, "fo_1": -0.04,
              "Ia_2": 0.006, "fv_2": 0.1, "fc_2": 0.05, "fo_2": -0.02}
    packed = pack_inertial_parameters(robot, masses, centers, inertias, extras=extras)
    actual = dict(zip(map(str, robot.symbols.dynparms()), packed))

    for link, (mass, center, inertia) in enumerate(zip(masses, centers, inertias), start=1):
        # Use the cross-product form of the parallel-axis theorem independently.
        cx, cy, cz = center
        cross = np.array([[0, -cz, cy], [cz, 0, -cx], [-cy, cx, 0]])
        origin = inertia + mass * cross.T @ cross
        assert actual[f"m_{link}"] == mass
        for axis, value in zip("xyz", mass * center):
            assert actual[f"l_{link}{axis}"] == pytest.approx(value)
        for component, row, column in (
            ("xx", 0, 0), ("xy", 0, 1), ("xz", 0, 2),
            ("yy", 1, 1), ("yz", 1, 2), ("zz", 2, 2),
        ):
            assert actual[f"L_{link}{component}"] == pytest.approx(origin[row, column])
    for name, value in extras.items():
        assert actual[name] == value


@pytest.mark.parametrize("order", ["khalil", "mass first"])
def test_full_parameter_jacobian_matches_finite_difference(order):
    robot = _robot(2, order=order, extras=True)
    masses, centers, inertias = _physical_fixture()
    full = pack_inertial_parameters(
        robot, masses, centers, inertias, extras={"fo_1": -0.1, "Ia_2": 0.003},
    )
    layout = _Layout(robot)
    coordinates = layout.from_full(full)
    reconstructed, analytic = layout.full(coordinates, jacobian=True)
    numerical = _central_jacobian(layout.full, coordinates)
    np.testing.assert_allclose(reconstructed, full, rtol=0, atol=2e-16)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-7, atol=2e-10)


@pytest.mark.parametrize("mass_bound, expected_mass_gradient", [
    ({"total_mass_lower": 3.5}, -1),
    ({"total_mass_upper": 2.5}, 1),
])
def test_physical_penalty_uses_linear_mass_and_inertia_hinges(mass_bound, expected_mass_gradient):
    robot = _robot(2)
    layout = _Layout(robot)
    full = pack_inertial_parameters(
        robot, [1.0, 2.0], np.zeros((2, 3)),
        [np.diag([0.01, 0.02, 0.08]), np.diag([0.02, 0.025, 0.03])],
    )
    coordinates = layout.from_full(full)
    config = RecoveryConfig(**mass_bound)
    penalty, gradient = _physical_penalty(coordinates, layout, config, gradient=True)
    assert penalty == pytest.approx(0.5 + 0.025)
    expected = np.zeros(layout.size)
    expected[:2] = expected_mass_gradient
    expected[8:14] = [-0.5, 0, 0, -0.5, 0, 0.5]
    np.testing.assert_allclose(gradient, expected, atol=1e-14)


def test_physical_penalty_gradient_includes_offdiagonal_eigenvector_terms():
    robot = _robot()
    layout = _Layout(robot)
    full = pack_inertial_parameters(
        robot, 1.0, [0.03, 0.02, -0.01],
        [[0.01, 0.006, 0.002], [0.006, 0.02, -0.004], [0.002, -0.004, 0.08]],
    )
    coordinates = layout.from_full(full)
    config = RecoveryConfig(total_mass_lower=1.3)
    penalty, analytic = _physical_penalty(coordinates, layout, config, gradient=True)
    numerical = _central_jacobian(lambda x: _physical_penalty(x, layout, config), coordinates).ravel()
    assert penalty > 0.3
    assert np.linalg.norm(analytic[[5, 6, 8]]) > 0.01
    np.testing.assert_allclose(analytic, numerical, rtol=2e-7, atol=2e-9)


def test_physical_penalty_vanishes_strictly_inside_feasible_region():
    robot = _robot(2)
    layout = _Layout(robot)
    coordinates = layout.from_full(pack_inertial_parameters(robot, *_physical_fixture()))
    value, gradient = _physical_penalty(
        coordinates, layout, RecoveryConfig(total_mass_lower=3.0, total_mass_upper=5.0), gradient=True,
    )
    assert value == 0.0
    np.testing.assert_array_equal(gradient, np.zeros(layout.size))


def test_recovery_config_json_roundtrip_preserves_defaults():
    defaults = RecoveryConfig()
    serialized = json.loads(json.dumps(defaults.to_dict(), allow_nan=False))
    restored = RecoveryConfig(**serialized)
    assert restored.to_dict() == defaults.to_dict()
    assert restored.penalty_weights == (0.0, 10.0, 20.0, 30.0, 40.0)


@pytest.mark.parametrize("overrides", [
    {"penalty_weights": []}, {"penalty_weights": [10, 0]},
    {"penalty_weights": [0, np.nan]}, {"runs": 0}, {"annealing_maxiter": 0},
    {"total_mass_lower": 2, "total_mass_upper": 1},
])
def test_invalid_optimizer_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        RecoveryConfig(**overrides)


@pytest.mark.parametrize("masses, centers, inertias, extras", [
    (0, [0, 0, 0], np.eye(3), None),
    (1, [np.nan, 0, 0], np.eye(3), None),
    (1, [0, 0, 0], [[1, 1, 0], [0, 1, 0], [0, 0, 1]], None),
    (1, [0, 0, 0], np.eye(3), {"unknown": 1}),
])
def test_pack_rejects_invalid_physical_inputs(masses, centers, inertias, extras):
    with pytest.raises(ValueError):
        pack_inertial_parameters(_robot(), masses, centers, inertias, extras=extras)


class TestNumericalRecovery:
    @pytest.fixture(autouse=True)
    def _need_scipy(self):
        pytest.importorskip("scipy")

    @pytest.fixture
    def problem(self):
        robot = _robot()
        masses, centers, inertias = _physical_fixture()
        target = pack_inertial_parameters(robot, masses[:1], centers[:1], inertias[:1])
        config = RecoveryConfig(
            mass_lower=1.0, mass_upper=2.0,
            com_lower=-0.08, com_upper=0.08,
            inertia_diagonal_lower=0.005, inertia_diagonal_upper=0.05,
            inertia_offdiagonal_bound=0.003,
            total_mass_lower=1.1, total_mass_upper=1.9,
            runs=1, annealing_maxiter=2, local_maxiter=500, ftol=1e-14,
            seed=32, max_base_error=1e-5,
        )
        return robot, target, config

    @pytest.mark.parametrize("use_nominal", [False, True])
    def test_identity_base_map_recovers_physical_parameters(self, problem, use_nominal):
        robot, target, config = problem
        nominal = None
        if use_nominal:
            nominal = pack_inertial_parameters(
                robot, 1.3, [0.02, 0.01, -0.01], np.diag([0.02, 0.025, 0.03]),
            )
        records = []
        result = recover_inertial_parameters(
            robot, target, nominal_parameters=nominal, config=config, progress=records.append,
        )
        assert result.success, result.report
        assert result.report["physical_feasible"]
        np.testing.assert_allclose(result.full_parameters, target, rtol=0, atol=1e-5)
        np.testing.assert_allclose(result.base_parameters, target, rtol=0, atol=1e-5)
        np.testing.assert_allclose(result.masses, [1.5], atol=1e-5)
        np.testing.assert_allclose(result.centers_of_mass, [[0.01, -0.02, 0.03]], atol=1e-5)
        np.testing.assert_allclose(result.inertias_com, _physical_fixture()[2][:1], atol=1e-5)
        assert [r["penalty_weight"] for r in records] == list(config.penalty_weights)
        assert [r["stage"] for r in records] == list(range(len(config.penalty_weights)))
        assert all(r["initialization"] == ("nominal" if use_nominal else "random") for r in records)
        output = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        assert output["links"][0]["mass"] == pytest.approx(1.5, abs=1e-5)
        assert output["inertia_frame"] == "at center of mass, expressed in link coordinate axes"

    def test_infeasible_identified_inertia_is_not_reported_as_success(self, problem):
        robot, _, config = problem
        target = pack_inertial_parameters(robot, 1.5, [0, 0, 0], np.diag([0.01, 0.01, 0.1]))
        nominal = pack_inertial_parameters(robot, 1.5, [0, 0, 0], np.diag([0.02, 0.025, 0.03]))
        config = replace(config, inertia_diagonal_upper=0.15, max_base_error=1e-4)
        result = recover_inertial_parameters(robot, target, nominal_parameters=nominal, config=config)
        assert not result.success
        assert not (result.report["physical_feasible"] and result.report["matches_base"])

    def test_unidentifiable_masses_can_differ_with_same_observed_inertia(self, problem):
        _, _, config = problem
        # A single revolute-axis observation only identifies Lzz here.
        robot = _robot(observed=[5])
        first = pack_inertial_parameters(robot, 1.2, [0, 0, 0], np.diag([0.02, 0.025, 0.03]))
        second = pack_inertial_parameters(robot, 1.8, [0, 0, 0], np.diag([0.02, 0.025, 0.03]))
        results = [recover_inertial_parameters(robot, [0.03], nominal_parameters=p, config=config)
                   for p in (first, second)]
        assert all(result.success for result in results)
        for result in results:
            np.testing.assert_allclose(result.base_parameters, [0.03], atol=1e-8)
        assert abs(results[0].masses[0] - results[1].masses[0]) > 0.4

    @pytest.mark.parametrize("overrides", [
        {"total_mass_lower": 2.1, "total_mass_upper": None},
        {"total_mass_lower": None, "total_mass_upper": 0.9},
        {"mass_lower": np.nan}, {"mass_upper": [2.0, 3.0]},
        {"com_lower": [0.2, 0.2, 0.2]},
    ])
    def test_invalid_bounds_are_rejected_before_optimization(self, problem, overrides):
        robot, target, config = problem
        with pytest.raises(ValueError):
            recover_inertial_parameters(robot, target, config=replace(config, **overrides))

    @pytest.mark.parametrize("target", [np.zeros(9), np.zeros((10, 1)), np.full(10, np.nan)])
    def test_invalid_base_vector_is_rejected(self, problem, target):
        robot, _, config = problem
        with pytest.raises(ValueError):
            recover_inertial_parameters(robot, target, config=config)

    @pytest.mark.parametrize("nominal", [np.zeros(9), np.full(10, np.nan), np.zeros(10)])
    def test_invalid_nominal_vector_is_rejected(self, problem, nominal):
        robot, target, config = problem
        with pytest.raises(ValueError):
            recover_inertial_parameters(robot, target, nominal_parameters=nominal, config=config)

    @pytest.mark.parametrize("mapping", ["missing", "nan", "wrong_shape"])
    def test_invalid_base_mapping_is_rejected(self, problem, mapping):
        robot, target, config = problem
        if mapping == "missing":
            robot.dynamics.Kd = None
        elif mapping == "nan":
            robot.dynamics.Pb[0, 0] = np.nan
        else:
            robot.dynamics.Pb = np.eye(9)
        with pytest.raises(ValueError):
            recover_inertial_parameters(robot, target, config=config)


@pytest.mark.parametrize("overrides", [
    {"nominal_prior_weight": -1}, {"nominal_prior_weight": np.nan},
    {"nominal_prior_weight": np.inf}, {"geometry_center": [0, 0, 0]},
    {"geometry_radius": 0.2}, {"geometry_center": [0, 0, 0], "geometry_radius": 0},
    {"geometry_center": [0, np.nan, 0], "geometry_radius": 0.2},
])
def test_invalid_prior_and_geometry_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        RecoveryConfig(**overrides)


def test_spherical_envelope_derivatives_include_mass_and_com_coupling():
    layout = _Layout(_robot(2, extras=True))
    x = layout.from_full(pack_inertial_parameters(
        _robot(2, extras=True), *_physical_fixture(), extras={"fo_1": -0.08},
    ))
    config = RecoveryConfig(geometry_center=[[0.02, 0.01, -0.04], [0, 0, 0]],
                            geometry_radius=[0.11, 0.5], total_mass_upper=3.5)
    signed, analytic = _geometry_violations(x, layout, config, gradient=True)
    numeric = _central_jacobian(lambda v: _geometry_violations(v, layout, config), x)
    assert signed[0] > 0 and signed[1] < 0
    np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=2e-8)
    penalty, gradient = _physical_penalty(x, layout, config, gradient=True)
    numerical_penalty = _central_jacobian(lambda v: _physical_penalty(v, layout, config), x).ravel()
    assert penalty == pytest.approx(0.3 + signed[0])
    np.testing.assert_allclose(gradient, numerical_penalty, rtol=2e-7, atol=2e-8)


def test_spherical_envelope_rejects_tiny_mass_with_large_inertia():
    robot = _robot()
    layout = _Layout(robot)
    x = layout.from_full(pack_inertial_parameters(robot, 0.001, [0, 0, 0], 0.01*np.eye(3)))
    unconstrained = RecoveryConfig()
    lower, upper = _bounds(layout, unconstrained)
    assert _audit(x, layout, unconstrained, lower, upper)["physical_feasible"]
    constrained = replace(unconstrained, geometry_center=[0, 0, 0], geometry_radius=0.3)
    report = _audit(x, layout, constrained, lower, upper)
    assert not report["physical_feasible"]
    assert not report["geometry_feasible"]
    assert report["geometry_violations"][0] > 100
    assert report["geometry_second_moment_margins"][0] < 0
    assert report["mass_bound_contacts"] == [
        {"link": 1, "bound": "lower", "value": 0.001, "bound_value": 0.001},
    ]


def test_spherical_envelope_uses_displaced_center_and_boundary_is_feasible():
    robot = _robot()
    layout = _Layout(robot)
    # trace(I)/2 = 0.03, m*distance^2 = 0.02, m*r^2 = 0.05.
    x = layout.from_full(pack_inertial_parameters(robot, 2, [0.15, 0, 0], 0.02*np.eye(3)))
    config = RecoveryConfig(geometry_center=[0.05, 0, 0], geometry_radius=np.sqrt(0.025))
    lower, upper = _bounds(layout, config)
    report = _audit(x, layout, config, lower, upper)
    assert report["geometry_feasible"]
    assert report["physical_feasible"]
    assert report["geometry_second_moment_margins"][0] == pytest.approx(0, abs=1e-16)


def test_prior_gradient_uses_frobenius_inertia_and_excludes_extras():
    robot = _robot(2, extras=True)
    layout = _Layout(robot)
    initial = layout.from_full(pack_inertial_parameters(robot, *_physical_fixture()))
    config = RecoveryConfig(geometry_center=[0, 0, 0], geometry_radius=[0.2, 0.3])
    lower, upper = _bounds(layout, config)
    scales = _prior_scales(initial, layout, config, lower, upper)
    x = initial + np.linspace(-0.001, 0.002, layout.size)
    prior, analytic = _nominal_prior(x, initial, layout, scales, gradient=True)
    numeric = _central_jacobian(lambda v: _nominal_prior(v, initial, layout, scales), x).ravel()
    np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=2e-9)
    np.testing.assert_array_equal(analytic[20:], 0)
    np.testing.assert_array_equal(scales["mass"], initial[:2])
    np.testing.assert_array_equal(scales["center_of_mass"], [0.2, 0.3])
    np.testing.assert_allclose(scales["inertia_com"], np.linalg.norm(_physical_fixture()[2], axis=(1, 2)))
    x[20:] += 100
    assert _nominal_prior(x, initial, layout, scales) == prior
    assert _nominal_prior(initial, initial, layout, scales) == 0


def test_prior_requires_nominal_parameters_and_valid_geometry_shape():
    pytest.importorskip("scipy")
    robot = _robot()
    target = pack_inertial_parameters(robot, 1, [0, 0, 0], 0.01*np.eye(3))
    with pytest.raises(ValueError, match="nominal_parameters"):
        recover_inertial_parameters(robot, target, config=RecoveryConfig(nominal_prior_weight=0.01))
    with pytest.raises(ValueError, match="geometry_radius"):
        recover_inertial_parameters(robot, target, config=RecoveryConfig(
            geometry_center=[0, 0, 0], geometry_radius=[0.1, 0.2],
        ))


def test_prior_resolves_unidentifiable_mass_allocation_without_fixing_observed_sum():
    pytest.importorskip("scipy")
    robot = _robot(2, observed=[9])
    layout = _Layout(robot)
    # Observe only m1 + m2, so the individual masses remain unidentifiable.
    robot.dynamics.Kd[0, list(np.flatnonzero(robot.dynamics.Pd[layout.mass[1]]))[0]] = 1
    nominal = pack_inertial_parameters(
        robot, [1.2, 1.8], [[0, 0, 0], [0, 0, 0]], [0.02*np.eye(3), 0.03*np.eye(3)],
    )
    config = RecoveryConfig(
        mass_lower=0.5, mass_upper=4, com_lower=-0.2, com_upper=0.2,
        inertia_diagonal_upper=0.1, inertia_offdiagonal_bound=0.05,
        runs=1, annealing_maxiter=1, local_maxiter=500, ftol=1e-14,
        penalty_weights=(0, 10), max_base_error=0.001,
    )
    plain = recover_inertial_parameters(robot, [4], nominal_parameters=nominal, config=config)
    regularized = recover_inertial_parameters(
        robot, [4], nominal_parameters=nominal, config=replace(config, nominal_prior_weight=0.001),
    )
    assert plain.success, plain.report
    assert regularized.success, regularized.report
    # For a fixed total, relative mass regularization allocates the increment
    # proportionally to m0^2, an independently calculable nullspace solution.
    expected = np.array([1.2, 1.8]) + np.array([1.2, 1.8])**2/(1.2**2 + 1.8**2)
    np.testing.assert_allclose(regularized.masses, expected, atol=0.001)
    np.testing.assert_allclose(regularized.base_parameters, [4], atol=0.001)
    assert np.linalg.norm(plain.masses-regularized.masses) > 0.1
    assert regularized.report["nominal_prior_penalty"] < plain.report["nominal_prior_penalty"]
    assert regularized.report["nominal_prior_loss"] > 0
    assert regularized.report["nominal_prior_enabled"]
    json.dumps(regularized.to_dict(), allow_nan=False)


def test_successful_candidate_is_preferred_over_lower_loss_physical_failure(monkeypatch):
    scipy_optimize = pytest.importorskip("scipy.optimize")
    robot = _robot(observed=[5])
    layout = _Layout(robot)
    config = RecoveryConfig(runs=2, penalty_weights=(0,), annealing_maxiter=1,
                            max_base_error=0.1)
    lower, upper = _bounds(layout, config)
    # First point has the exact observed Lzz but violates inertia triangle;
    # the second has a tiny residual and is physically valid.
    points = [layout.from_full(pack_inertial_parameters(robot, 1, [0, 0, 0], inertia))
              for inertia in [np.diag([0.01, 0.01, 0.1]), 0.09*np.eye(3)]]
    starts = iter((point-lower)/(upper-lower) for point in points)
    monkeypatch.setattr(scipy_optimize, "dual_annealing", lambda *a, **k: SimpleNamespace(x=next(starts)))
    monkeypatch.setattr(scipy_optimize, "minimize", lambda function, x, **k:
                        SimpleNamespace(x=x, success=True, message="test candidate", nit=0))
    result = recover_inertial_parameters(robot, [0.1], config=config)
    assert result.success, result.report
    assert result.report["selected_run"] == 1
    assert result.report["attempts"][0]["loss"] < result.report["attempts"][1]["loss"]


def test_audit_identifies_com_and_inertia_box_contacts():
    robot = _robot()
    layout = _Layout(robot)
    config = RecoveryConfig(com_lower=-0.2, com_upper=0.2, inertia_diagonal_upper=0.03)
    x = layout.from_full(pack_inertial_parameters(robot, 1, [0, 0.2, -0.2], 0.03*np.eye(3)))
    lower, upper = _bounds(layout, config)
    report = _audit(x, layout, config, lower, upper)
    assert report['physical_feasible']
    assert {(c['link'], c['component'], c['bound']) for c in report['com_bound_contacts']} == {
        (1, 'y', 'upper'), (1, 'z', 'lower'),
    }
    assert {(c['component'], c['bound']) for c in report['inertia_bound_contacts']} == {
        ('xx', 'upper'), ('yy', 'upper'), ('zz', 'upper'),
    }
