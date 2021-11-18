from opti.problems import ZDT1

from mbo.algorithm import Algorithm


def test_init():
    # set up with initial data from problem
    problem = ZDT1(n_inputs=3)
    problem.create_initial_data(6)
    optimizer = Algorithm(problem)
    assert len(optimizer.problem.get_data()) == 6
