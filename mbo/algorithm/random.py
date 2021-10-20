import opti

from mbo.algorithm import Algorithm


class RandomSearch(Algorithm):
    def __init__(self, problem: opti.Problem):
        """Random selection of new data.

        Args:
            problem: Optimization problem
            data: Initial Data to use instead of the problem.data
        """
        super().__init__(problem)

    def propose(self, n_proposals=1):
        df = self.problem.sample_inputs(n_proposals)
        df.index += 0 if self.problem.data is None else len(self.problem.data)
        return df

    def to_config(self) -> dict:
        return {"method": "RandomSearch", "problem": self._problem.to_config()}
