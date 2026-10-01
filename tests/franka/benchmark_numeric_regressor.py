"""Compare the pre-batching column recursion with the batched numeric regressor.

Run from the repository root, optionally with saved, preprocessed MotionData::

    PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m tests.franka.benchmark_numeric_regressor \
        --data outputs/showcase-fresh/franka/train-processed.npz

Timing is informational; the 1e-10 consistency checks are mandatory.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np

from robot_model import Robot
from robot_model.dynamics.base_parms import find_dyn_parm_deps
from robot_model.robots.franka import PANDA_MDH_PARMS


def column_regressor(nd, q, dq, ddq):
    """Original algorithm: one shared forward pass, then one recursion per column."""
    if nd.method == "park":
        state = nd._park_forward(q, dq, ddq)
        backward = nd._park_backward
        dq_a, ddq_a = state[4:6]
    else:
        state = nd._khalil_forward(q, dq, ddq)
        backward = nd._khalil_backward
        dq_a, ddq_a = state[6:8]
    H = np.zeros((nd.dof, nd.n_dynparms))
    for col, (kind, joint, _) in enumerate(nd._parm_slots):
        if kind in ("Ia", "fv", "fc", "fo"):
            H[:, col] = nd._actuator_column(kind, joint, dq_a, ddq_a, nd.dof)
        else:
            H[:, col] = backward(*state, *nd._unit_params(col))
    return H


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Preprocessed MotionData .npz (q, dq, ddq, tau)")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--base-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=("park", "khalil"), default="khalil")
    args = parser.parse_args()
    robot = Robot.from_mdh(
        "panda_batch_benchmark", PANDA_MDH_PARMS,
        frictionmodel={"viscous", "Coulomb"}, driveinertiamodel="simplified",
    )
    nd = robot.dynamics.numeric_dynamics(method=args.method)
    old = lambda q, dq, ddq: column_regressor(nd, q, dq, ddq)
    rng = np.random.default_rng(args.seed)
    if args.data:
        with np.load(args.data) as data:
            q, dq, ddq, tau = (data[key] for key in ("q", "dq", "ddq", "tau"))
    else:
        q, dq, ddq = rng.uniform(-np.pi, np.pi, (3, args.samples, nd.dof))
        pi = rng.normal(size=nd.n_dynparms)
        tau = np.array([old(*state) @ pi for state in zip(q, dq, ddq)])
        tau += rng.normal(scale=0.01, size=tau.shape)
    matrices, mappings, timings = [], [], []
    for func in (old, nd.regressor):
        start = perf_counter()
        mapping = find_dyn_parm_deps(
            nd.dof, nd.n_dynparms, func, samples=args.base_samples, seed=args.seed,
        )
        base_seconds = perf_counter() - start
        func(q[0], dq[0], ddq[0])  # warm up
        start = perf_counter()
        H = np.vstack([func(*state) for state in zip(q, dq, ddq)])
        regressor_seconds = perf_counter() - start
        mappings.append(mapping)
        matrices.append(H)
        timings.append({"base_seconds": base_seconds,
                        "regressor_seconds": regressor_seconds,
                        "milliseconds_per_sample": regressor_seconds / len(q) * 1000})
    before, after = mappings
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_allclose(before[2], after[2], rtol=0, atol=1e-10)
    np.testing.assert_allclose(matrices[0], matrices[1], rtol=0, atol=1e-10)
    fits = [np.linalg.lstsq(H @ mapping[0], tau.ravel(), rcond=None)
            for H, mapping in zip(matrices, mappings)]
    np.testing.assert_allclose(fits[0][0], fits[1][0], rtol=0, atol=1e-10)
    assert fits[0][2] == fits[1][2] == before[0].shape[1] == 62
    result = {
        "method": args.method, "seed": args.seed, "samples": len(q),
        "base_samples": args.base_samples, "rank": int(fits[1][2]),
        "before": timings[0], "after": timings[1],
        "speedup": timings[0]["regressor_seconds"] / timings[1]["regressor_seconds"],
        "max_regressor_difference": float(np.max(np.abs(matrices[0] - matrices[1]))),
        "max_base_mapping_difference": float(np.max(np.abs(before[2] - after[2]))),
        "max_identified_parameter_difference": float(np.max(np.abs(fits[0][0] - fits[1][0]))),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
