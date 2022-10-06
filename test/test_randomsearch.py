import opti
import pytest

from mbo.algorithm import RandomSearch
from mbo.error import UnsuitableAlgorithmError


def test_check_problem():
    # expect error for problems with non-linear equality constraints
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i+1}", [0, 1]) for i in range(3)],
        outputs=[opti.Continuous("y")],
        constraints=[opti.NonlinearEquality("x1**2 + x2**2 + x3")],
    )
    with pytest.raises(UnsuitableAlgorithmError):
        RandomSearch(problem)


def test_unconstrained_problem():
    problem = opti.problems.ZDT1(n_inputs=5)
    optimizer = RandomSearch(problem)

    X = optimizer.propose(n_proposals=10)
    assert len(X) == 10
    assert problem.inputs.contains(X).all()

    optimizer.run(n_proposals=3, n_steps=10)
    assert len(optimizer.problem.data) == 30

    config = optimizer.to_config()
    assert config["method"] == "RandomSearch"


def test_constrained_problem():
    problem = opti.problems.Hyperellipsoid()
    optimizer = RandomSearch(problem)

    X = optimizer.propose(n_proposals=10)
    assert len(X) == 10
    assert problem.inputs.contains(X).all()


def test_single_objective_problem():
    problem = opti.problems.Sphere(n_inputs=3)
    optimizer = RandomSearch(problem)

    X = optimizer.propose(n_proposals=10)
    assert len(X) == 10
    assert problem.inputs.contains(X).all()


if __name__ == "__main__":
    test_check_problem()
