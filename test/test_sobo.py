import opti
import pytest

from mbo.algorithm import SOBO
from mbo.error import UnsuitableAlgorithmError


def test_check_problem():
    # expect error for problem with no initial data
    problem = opti.problems.Detergent()
    with pytest.raises(UnsuitableAlgorithmError):
        SOBO(problem)

    # expect error for multi-objective problem
    problem = opti.problems.ZDT1(n_inputs=3)
    problem.create_initial_data(5)
    with pytest.raises(UnsuitableAlgorithmError):
        SOBO(problem)

    # expect error for problem with categorical or discrete input
    problem = opti.problems.DiscreteFuelInjector()
    problem.create_initial_data(5)
    with pytest.raises(UnsuitableAlgorithmError):
        SOBO(problem)

    # expect error for problem with nonlinear constraints
    problem = opti.problems.Daechert1()
    problem.create_initial_data(5)
    with pytest.raises(UnsuitableAlgorithmError):
        SOBO(problem)


def test_sphere():
    problem = opti.problems.Sphere(n_inputs=3)
    problem.create_initial_data(10)
    optimizer = SOBO(problem)

    # parameters
    params = optimizer.get_model_parameters()
    assert len(params) == 1

    # predict
    X = problem.sample_inputs(10)
    Ypred = optimizer.predict(X)
    assert len(Ypred) == 10

    # propose
    X = optimizer.propose(n_proposals=2)
    assert len(X) == 2
    assert problem.inputs.contains(X).all()

    # predict optimum, n_levels should be ignored
    XY = optimizer.predict_pareto_front(n_levels=10)
    assert len(XY) == 1

    # to_config
    config = optimizer.to_config()
    assert config["method"] == "SOBO"
