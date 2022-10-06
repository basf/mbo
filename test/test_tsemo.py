import numpy as np
import opti
import pytest
from opti.constraint import LinearEquality
from opti.parameter import Continuous

from mbo.algorithm import TSEMO
from mbo.error import UnsuitableAlgorithmError


def test_check_problem():
    # expect error for problem with no initial data
    problem = opti.problems.ZDT1()
    with pytest.raises(UnsuitableAlgorithmError):
        TSEMO(problem)

    # expect error for problem with non-continuous inputs
    problem = opti.problems.DiscreteFuelInjector()
    problem.create_initial_data(5)
    with pytest.raises(UnsuitableAlgorithmError):
        TSEMO(problem)

    # expect error for problem with equality constraints
    problem = opti.Problem(
        inputs=[Continuous(f"x{i+1}", [0, 1]) for i in range(2)],
        outputs=[Continuous(f"y{i+1}", [0, 1]) for i in range(2)],
        constraints=[LinearEquality(["x1", "x2"], lhs=[1, 1], rhs=1)],
        f=lambda x: x,
    )
    problem.create_initial_data(10)
    with pytest.raises(UnsuitableAlgorithmError):
        TSEMO(problem)

    # expect error for single-objective problem
    problem = opti.problems.Sphere(n_inputs=3)
    problem.create_initial_data(10)
    with pytest.raises(UnsuitableAlgorithmError):
        TSEMO(problem)

    # expect error when not at least one full observation available
    problem = opti.problems.ZDT1(n_inputs=3)
    problem.create_initial_data(10)
    problem.data["y1"] = np.nan
    with pytest.raises(UnsuitableAlgorithmError):
        TSEMO(problem)


def test_unconstrained_problem():
    problem = opti.problems.ZDT1(n_inputs=2)
    problem.create_initial_data(10)
    optimizer = TSEMO(problem)

    config = optimizer.to_config()
    assert config["method"] == "TSEMO"

    # predict
    X = problem.inputs.sample(10)
    Ypred = optimizer.predict(X)
    assert len(Ypred) == 10

    # propose
    X = optimizer.propose(4)
    assert len(X) == 4
    assert problem.inputs.contains(X).all()

    # run
    optimizer.run(n_proposals=1, n_steps=2)
    assert len(optimizer.problem.data) == 12

    # # check convergence to true pareto front
    # Y = optimizer.data[problem.outputs.names].values
    # exp_front = opti.metric.pareto_front(Y)
    # true_front = problem.get_optima()[problem.outputs.names].values
    # assert opti.metric.generational_distance(exp_front, true_front) < 0.5

    # predict Pareto front
    front = optimizer.predict_pareto_front(n_points=42)
    assert len(front) <= 42


def test_missing_output():
    problem = opti.problems.ZDT1(n_inputs=3)
    problem.create_initial_data(10)
    problem.data.loc[0, problem.outputs.names[0]] = np.nan
    problem.data.loc[9, problem.outputs.names[1]] = np.nan
    optimizer = TSEMO(problem)

    # proposals work
    X = optimizer.propose(n_proposals=2)
    assert len(X) == 2
    assert problem.inputs.contains(X).all()
