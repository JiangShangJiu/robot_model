"""Regression checks for rank-deficient and underdetermined base mappings."""

import numpy as np
import pytest

from robot_model.dynamics.base_parms import find_dyn_parm_deps


@pytest.mark.parametrize(
    'matrix',
    [
        [[0., 1.], [0., 0.]],
        [[1., 1., 0.], [0., 0., 1.], [0., 0., 0.]],
        [[0., 0., 1., 2.], [0., 0., 0., 1.]],
        [[0., 0., 0.], [0., 0., 0.]],
    ],
)
def test_base_mapping_preserves_rank_and_torque(matrix):
    H = np.asarray(matrix)
    Pb, Pd, Kd = find_dyn_parm_deps(
        *H.shape, lambda q, dq, ddq: H, samples=1
    )
    assert Pb.shape[1] == np.linalg.matrix_rank(H)
    assert Pb.shape[1] + Pd.shape[1] == H.shape[1]
    B = Pb.T + Kd @ Pd.T
    np.testing.assert_allclose(H @ Pb @ B, H, atol=1e-10)
    pi = np.arange(1., H.shape[1] + 1.)
    np.testing.assert_allclose(H @ pi, (H @ Pb) @ (B @ pi), atol=1e-10)


def test_valid_existing_base_order_is_preserved():
    H = np.array([[1., 0., 0., 1., 2.],
                  [0., 2., 0., -1., 1.],
                  [0., 0., 3., 2., -1.]])
    Pb, Pd, Kd = find_dyn_parm_deps(
        *H.shape, lambda q, dq, ddq: H, samples=10
    )
    np.testing.assert_array_equal(Pb, np.eye(5)[:, :3])
    np.testing.assert_array_equal(Pd, np.eye(5)[:, 3:])
    np.testing.assert_allclose(H @ Pb @ (Pb.T + Kd @ Pd.T), H, atol=2e-10)


@pytest.mark.parametrize('keyword,value', [
    ('samples', 0), ('samples', -1), ('samples', 1.5),
    ('dof', 0), ('parm_num', 0), ('round_digits', -1), ('seed', -1),
])
def test_invalid_sampling_configuration_is_rejected(keyword, value):
    config = dict(dof=2, parm_num=2, samples=1)
    config[keyword] = value
    with pytest.raises(ValueError, match=keyword):
        find_dyn_parm_deps(regressor_func=lambda q, dq, ddq: np.eye(2), **config)


@pytest.mark.parametrize('matrix', [
    np.ones(4), np.ones((1, 4)), np.array([[1., np.nan], [0., 1.]]),
    np.array([[1., 0.], [0., np.inf]]),
])
def test_bad_regressor_is_rejected(matrix):
    with pytest.raises(ValueError, match='regressor_func'):
        find_dyn_parm_deps(2, 2, lambda q, dq, ddq: matrix, samples=1)
