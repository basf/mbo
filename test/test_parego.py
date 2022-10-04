import numpy as np
import opti
import pytest

from mbo.algorithm import ParEGO
from mbo.error import UnsuitableAlgorithmError


def test_check_problem():
    # expect error for problem with no initial data
    problem = opti.problems.Detergent()
    with pytest.raises(UnsuitableAlgorithmError):
        ParEGO(problem)

    # expect error for problem with non-continuous inputs
    problem = opti.problems.DiscreteFuelInjector()
    with pytest.raises(UnsuitableAlgorithmError):
        ParEGO(problem)

    # expect error for problem with non-linear constraints
    problem = opti.problems.Qapi1()
    problem.create_initial_data(10)
    with pytest.raises(UnsuitableAlgorithmError):
        ParEGO(problem)

    # expect error for single-objective problem
    problem = opti.problems.Sphere(n_inputs=3)
    problem.create_initial_data(5)
    with pytest.raises(UnsuitableAlgorithmError):
        ParEGO(problem)


def test_config():
    problem = opti.problems.ZDT1(n_inputs=2)
    problem.create_initial_data(6)
    optimizer = ParEGO(problem, restarts=1)
    config = optimizer.to_config()
    assert config["method"] == "ParEGO"


def test_detergent():
    problem = opti.problems.Detergent()
    n_initial = 20
    problem.create_initial_data(n_initial)
    problem.data.loc[0, "y1"] = np.nan
    problem.data.loc[9, "y2"] = np.nan

    optimizer = ParEGO(problem, restarts=1)

    # experiements with missing observations are not pruned
    assert len(optimizer.problem.data) == n_initial

    # predict
    X = problem.inputs.sample(10)
    X.index = np.arange(len(X)) + 10
    Y = optimizer.predict(X)
    assert len(Y) == 10
    assert np.all(X.index == Y.index)

    # propose
    X = optimizer.propose(n_proposals=2)
    assert len(X) == 2
    assert problem.inputs.contains(X).all()

    # run
    optimizer.run(n_proposals=1, n_steps=2)
    assert len(optimizer.problem.data) == n_initial + 2

    # predict Pareto front
    front = optimizer.predict_pareto_front(n_points=42)
    assert len(front) <= 42


def test_problem_with_output_constraint():
    problem = opti.problems.Detergent_OutputConstraint()
    problem.create_initial_data(10)

    optimizer = ParEGO(problem, restarts=1)

    # predict
    X = problem.inputs.sample(10)
    Ypred = optimizer.predict(X)
    assert len(Ypred) == 10

    # propose
    X = optimizer.propose(n_proposals=2)
    assert len(X) == 2
    assert problem.inputs.contains(X).all()
