from typing import Optional

import opti
import pandas as pd

from mbo.algorithm.algorithm import Algorithm
from mbo.error import UnsuitableAlgorithmError


class RandomSearch(Algorithm):
    """Random sampling of input spaces.

    This algorithm uses a suitable method to sample new points from the input space:
     - For unconstrained spaces a Sobol sequence is used.
     - For linearly constrained spaces, a hit-and-run Monte Carlo sampler is used.
     - For non-linear constraints a default random sampling method is used.

    Description:
    - Model: None.
    - Proposals: Random proposals.
    - Optimization: None.
    - Pareto approximation: None.
    """

    def __init__(self, problem: opti.Problem):
        super().__init__(problem)

    def _initialize_problem(self) -> None:
        # check if a suitable sampler for the input space is available
        try:
            self.propose()
        except Exception as err:
            raise UnsuitableAlgorithmError(
                f"RandomSearch: Can't sample from input space. {str(err)}"
            )

    def propose(self, n_proposals: Optional[int] = None) -> pd.DataFrame:
        if n_proposals is None:
            n_proposals = 1

        df = opti.sampling.constrained_sampling(
            n_proposals,
            parameters=self.problem.inputs,
            constraints=self._problem.constraints,
        )
        df.index += 0 if self.problem.data is None else len(self.problem.data)
        return df

    def to_config(self) -> dict:
        """Serialize the algorithm settings to a dictionary."""
        return {"method": "RandomSearch", "problem": self._problem.to_config()}
