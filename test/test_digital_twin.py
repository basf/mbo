import numpy as np
import pandas as pd
from opti.problems import ZDT1, Cake

from mbo.algorithm import Algorithm, RandomSearch
from mbo.digital_twin import NotLogger, _check_args, run


class _ZeroProposer(Algorithm):
    def __init__(self, problem):
        super().__init__(problem)
        self.name = "_ZeroProposer"
        self.problem = problem

    def propose(self, n_proposals: int = 1) -> pd.DataFrame:
        names = self.problem.inputs.names
        return pd.DataFrame(data=np.zeros((n_proposals, len(names))), columns=names)


def test_check_inputs():
    problem = Cake()
    _check_args(problem, _ZeroProposer(problem))


def test_run():
    problem = ZDT1()
    algorithm = RandomSearch(problem)
    metrics = run(algorithm, max_n_measurements=15, logger=NotLogger())
    assert isinstance(metrics, pd.DataFrame)
    assert len(metrics) == 14


if __name__ == "__main__":
    test_run()
    test_check_inputs()
