import opti
import pytest

from mbo.algorithm.rf import RandomForest
from mbo.error import UnsuitableAlgorithmError


def test_check_problem():
    # expect error for problem with no initial data
    problem = opti.problems.Detergent()
    with pytest.raises(UnsuitableAlgorithmError):
        RandomForest(problem)

    # expect error for problem with output constraints
    problem = opti.problems.Detergent_OutputConstraint()
    with pytest.raises(UnsuitableAlgorithmError):
        RandomForest(problem)


def test_univariate():
    problem = opti.problems.Line1D()
    optimizer = RandomForest(problem)

    # predict
    X = problem.sample_inputs(10)
    Y = optimizer.predict(X)
    assert len(Y) == 10

    # propose, works even for single objective
    X = optimizer.propose(2)
    assert len(X) == 2


def test_reizman_suzuki1():
    problem = opti.problems.ReizmanSuzuki()
    optimizer = RandomForest(problem)

    # predict
    X = problem.sample_inputs(10)
    Y = optimizer.predict(X)
    assert len(Y) == 10

    # propose
    X = optimizer.propose(2)
    assert len(X) == 2

    # get insights
    params = optimizer.get_model_parameters()
    assert len(params) == problem.n_outputs
