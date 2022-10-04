import time

import numpy as np
import opti
import pytest

from mbo.algorithm.doe import DOptimalDesign


def test_propose_unconstrained():
    # Test case: no constraints, linear model
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [0, 1]),
            opti.Continuous("x2", [0, 1]),
        ],
        outputs=[opti.Continuous("y")],
    )

    obj = DOptimalDesign(problem=problem, model_type="linear")

    with pytest.warns(UserWarning):
        data = obj.propose(n_proposals=4).to_numpy()
    correct_data = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )

    assert np.shape(data) == (4, 2)
    for row in data:
        assert np.any([np.allclose(row, _row) for _row in correct_data])
    for row in correct_data:
        assert np.any([np.allclose(row, _row) for _row in data])


def test_propose_mixture_constraint():
    # Test case: one mixture constraints, fully-quadratic model
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [0, 1]),
            opti.Continuous("x2", [0, 1]),
        ],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearEquality(names=["x1", "x2"], rhs=1),
        ],
    )

    obj = DOptimalDesign(problem=problem, model_type="fully-quadratic")

    data = obj.propose().to_numpy()
    correct_data = np.array(
        [
            [0, 1],
            [1, 0],
            [0.5, 0.5],
        ]
    )

    assert np.shape(data) == (9, 2)
    for row in data:
        assert np.any([np.allclose(row, _row, atol=2e-3) for _row in correct_data])
    for row in correct_data:
        assert np.any([np.allclose(row, _row, atol=2e-3) for _row in data])


def test_propose_linear_constraint():
    # Test case: one linear inequality constraint, linear model
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [0, 1]),
            opti.Continuous("x2", [0, 1]),
            opti.Continuous("x3", [0, 1]),
        ],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.LinearInequality(names=["x2", "x3"], rhs=0.7),
        ],
    )

    obj = DOptimalDesign(
        problem=problem,
        model_type="linear",
    )

    data = obj.propose(n_proposals=10).to_numpy()
    correct_data = np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0.7, 0],
            [0, 0, 0.7],
            [1, 0.7, 0],
            [1, 0, 0.7],
        ]
    )

    assert np.shape(data) == (10, 3)
    for row in data:
        assert np.any([np.allclose(row, _row, atol=2e-3) for _row in correct_data])
    for row in correct_data:
        assert np.any([np.allclose(row, _row, atol=2e-3) for _row in data])


def test_propose_NChooseK_constraint():
    # Test case: one linear inequality constraint, linear model
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [0, 1]),
            opti.Continuous("x2", [0, 1]),
            opti.Continuous("x3", [0, 1]),
        ],
        outputs=[opti.Continuous("y")],
        constraints=[
            opti.NChooseK(names=["x2", "x3"], max_active=1),
        ],
    )

    obj = DOptimalDesign(
        problem=problem,
        model_type="linear",
    )

    data = obj.propose(n_proposals=10).to_numpy()
    correct_data = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
        ]
    )

    assert np.shape(data) == (10, 3)
    for row in data:
        assert np.any([np.allclose(row, _row, atol=1e-2) for _row in correct_data])
    for row in correct_data:
        assert np.any([np.allclose(row, _row, atol=1e-2) for _row in data])


def test_to_config():
    # Test case: no constraints, linear model
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x1", [0, 1]),
            opti.Continuous("x2", [0, 1]),
        ],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(names=["x1", "x2"], rhs=1)],
    )

    obj = DOptimalDesign(
        problem=problem,
        model_type="linear",
        ipopt_options={"maxiter": 42, "disp": 1e42},
    )

    config = obj.to_config()
    correct_keys = ["method", "problem", "parameters"]

    assert np.all([key in correct_keys for key in config.keys()])
    assert config["problem"] == problem.to_config()
    assert config["method"] == "DOptimalDesign"

    params = config["parameters"]
    assert params["model_type"] == "linear"
    assert params["tol"] == 0
    assert params["delta"] == 1e-7
    assert params["ipopt_options"] == {
        "maxiter": 42,
        "disp": 1e42,
        "max_cpu_time": 2700,
    }
    assert params["fixed_experiments"] is None
    assert params["jacobian_building_block"] is None


def test_time_limit():
    # Test case: large problem, short run time limit
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0, 1]) for i in range(20)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.LinearEquality(names=[f"x{i+1}" for i in range(20)], rhs=1)],
    )

    obj = DOptimalDesign(
        problem=problem, model_type="linear", ipopt_options={"max_cpu_time": 4.0}
    )

    t = time.time()
    obj.propose()
    assert time.time() - t <= 5
