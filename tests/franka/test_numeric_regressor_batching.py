"""Batched rigid-body columns preserve the scalar RNE and identification results."""

from functools import lru_cache

import numpy as np
import pytest

from robot_model import Robot
from robot_model.dynamics.base_parms import find_dyn_parm_deps
from robot_model.robots.franka import PANDA_MDH_PARMS
from .benchmark_numeric_regressor import column_regressor


@lru_cache(maxsize=None)
def _mixed_robot(convention, order):
    # Revolute / prismatic / revolute, with nonzero offsets and tilted axes.
    rows = [("pi/2", "0.2", "0.3", "q"),
            ("-pi/2", "0.1", "q", "pi/4"),
            ("pi/2", "0.3", "0.2", "q")]
    factory = Robot.from_dh if convention == "standard" else Robot.from_mdh
    robot = factory(
        "mixed_batch", rows, dyn_parms_order=order,
        frictionmodel={"viscous", "Coulomb", "offset"},
        driveinertiamodel="simplified", gravityacc=[0.8, -1.1, -9.81],
    )
    return robot.to_poe() if convention == "poe" else robot


@pytest.mark.parametrize("order", ["khalil", "siciliano"])
@pytest.mark.parametrize("convention,method", [
    ("standard", "park"), ("standard", "khalil"),
    ("modified", "park"), ("modified", "khalil"), ("poe", "park"),
])
def test_batched_columns_match_scalar_rne(convention, method, order):
    nd = _mixed_robot(convention, order).dynamics.numeric_dynamics(method=method)
    rng = np.random.default_rng(2026)
    states = [np.zeros((3, nd.dof))] + [rng.uniform(-2, 2, (3, nd.dof)) for _ in range(4)]
    for q, dq, ddq in states:
        H = nd.regressor(q, dq, ddq)
        np.testing.assert_allclose(H, column_regressor(nd, q, dq, ddq), rtol=0, atol=1e-12)
        for col, (kind, link, _) in enumerate(nd._parm_slots):
            if kind in ("m", "l", "Le"):
                # A body's parameters cannot affect a downstream joint's torque.
                np.testing.assert_array_equal(H[link + 1:, col], 0.0)
            else:
                expected = {"Ia": ddq[link], "fv": dq[link],
                            "fc": np.sign(dq[link]), "fo": 1.0}[kind]
                assert H[link, col] == expected
                np.testing.assert_array_equal(np.delete(H[:, col], link), 0.0)


@pytest.mark.parametrize("method", ["park", "khalil"])
def test_franka_batching_preserves_base_mapping_and_identification(method):
    robot = Robot.from_mdh(
        "panda_batch", PANDA_MDH_PARMS,
        frictionmodel={"viscous", "Coulomb"}, driveinertiamodel="simplified",
    )
    nd = robot.dynamics.numeric_dynamics(method=method)
    old = lambda q, dq, ddq: column_regressor(nd, q, dq, ddq)
    mappings = [find_dyn_parm_deps(7, 91, func, samples=80, seed=0)
                for func in (old, nd.regressor)]
    for original, batched in zip(*mappings):
        np.testing.assert_allclose(original, batched, rtol=0, atol=1e-10)
    assert mappings[0][0].shape == mappings[1][0].shape == (91, 62)
    rng = np.random.default_rng(42)
    states = rng.uniform(-np.pi, np.pi, (80, 3, 7))
    W_old = np.vstack([old(*state) @ mappings[0][0] for state in states])
    W_new = np.vstack([nd.regressor(*state) @ mappings[1][0] for state in states])
    np.testing.assert_allclose(W_old, W_new, rtol=0, atol=1e-10)
    observations = W_old @ rng.normal(size=62) + rng.normal(scale=0.01, size=len(W_old))
    old_fit = np.linalg.lstsq(W_old, observations, rcond=None)
    new_fit = np.linalg.lstsq(W_new, observations, rcond=None)
    assert old_fit[2] == new_fit[2] == 62
    np.testing.assert_allclose(old_fit[0], new_fit[0], rtol=0, atol=1e-10)
