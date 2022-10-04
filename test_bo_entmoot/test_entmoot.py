import pytest

try:
    # running these tests requires entmoot, gurobipy and access to the Gurobi cloud
    import entmoot  # noqa: F401
    import gurobipy  # noqa: F401
except ImportError:
    pytest.skip("Gurobi not available", allow_module_level=True)

import numpy as np
import opti
import pandas as pd

from mbo.algorithm import Entmoot


def test_checks():
    problem = opti.problems.Zakharov_Categorical(n_inputs=3)
    with pytest.raises(ValueError):
        Entmoot(problem)


def test_singleobjective():
    problem = opti.problems.Zakharov_NChooseKConstraint(n_inputs=5, n_max_active=3)

    data = pd.DataFrame(
        [
            [2, 0, -2, 0, 0],
            [0, 3, -2, 0, 0],
            [0, 0, -2, 0, 0],
            [2, 0, -2, 6, 0],
            [0, -3, 2, 0, -4],
        ],
        columns=problem.inputs.names,
    )
    data["y"] = problem.f(data)
    problem.set_data(data)
    optimizer = Entmoot(problem)

    # predict
    X_pred = pd.DataFrame(
        [[0, 0.2, 0.2, 0.2, 0], [0.1, 0.1, 0.1, 0, 0]], columns=problem.inputs.names
    )
    Y_pred = optimizer.predict(X_pred)
    assert len(Y_pred) == 2
    assert "mean_y" in Y_pred.columns
    assert "std_y" in Y_pred.columns
    assert np.all(X_pred.index == Y_pred.index)

    # propose
    X_next = optimizer.propose(n_proposals=3)
    assert len(X_next) == 3


def test_biobjective():
    # opti.problems.ReizmanSuzuki -> bi-objective, cat + cont variables
    problem = opti.problems.ReizmanSuzuki()
    optimizer = Entmoot(problem)

    # predict
    X_pred = problem.data[problem.inputs.names]
    Y_pred = optimizer.predict(X_pred)
    assert len(Y_pred) == len(X_pred)

    # proponse
    X_next = optimizer.propose(n_proposals=2)
    assert len(X_next) == 2
    assert problem.inputs.contains(X_next).all()

    # approximate pareto front
    front = optimizer.predict_pareto_front(n_points=3)
    assert len(front) == 3
