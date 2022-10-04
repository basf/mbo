from typing import Optional

import numpy as np
import opti
import pandas as pd
import torch
from botorch.utils.gp_sampling import get_gp_samples

from mbo.algorithm import Algorithm
from mbo.error import UnsuitableAlgorithmError
from mbo.metric import hypervolume_improvement
from mbo.optimize import run_nsga3
from mbo.torch_tools import fit_gp, get_gp_parameters, predict, torch_kwargs


class TSEMO(Algorithm):
    """Thompson-Sampling for Efficient Multiobjective Optimization (TSEMO).

    In TSEMO new experiments are proposed with the following steps:
        1. Construct a GP model.
        2. Sample a function from the GP posterior via spectral sampling.
        3. Generate a set of efficient points for the sampled function using a multi-objective evolutionary algorithm.
        4. Select the point with the highest hypervolume improvement.
        5. Repeat steps 2-4 for batch proposals.

    References:
    - Bradford 2018, Efficient multiobjective optimization employing Gaussian processes, spectral sampling and a genetic algorithm.
    """

    def __init__(
        self,
        problem: opti.Problem,
        rff_samples: int = 500,
        pop_size: int = 100,
        generations: int = 100,
    ):
        """TSEMO algorithm.

        Args:
            problem: Problem definition.
            rff_samples: Number random fourier features. Increases the approximation accuracy of the GP function samples.
            pop_size: Population size for the optimization (NSGA-3). Increases the Pareto approximation accuracy.
            generations: Number of generations used for the optimization (NSGA-3). Increases the optimization accuracy.
        """
        super().__init__(problem)

        self.rff_samples = rff_samples
        self.pop_size = pop_size
        self.generations = generations

    def _initialize_problem(self) -> None:
        # Require 2-9 objectives (hypervolume calculation)
        if not (2 <= len(self.problem.objectives) <= 9):
            raise UnsuitableAlgorithmError("TSEMO requires 2-9 objectives.")

        # Require no output constraint (this could be implemented though)
        if self.problem.output_constraints:
            raise UnsuitableAlgorithmError("TSEMO cannot handle output constraints.")

        # Require all continuous inputs (GP modeling)
        for p in self.problem.inputs:
            if not (isinstance(p, opti.Continuous)):
                raise UnsuitableAlgorithmError(
                    "TSEMO can only optimize over continuous inputs."
                )

        # Require no equality constraints (pymoo cannot handle them)
        if self.problem.constraints is not None:
            for c in self.problem.constraints:
                if c.is_equality:
                    raise UnsuitableAlgorithmError(
                        "TSEMO cannot handle equality constraints."
                    )

        # Require initial data
        if self.problem.data is None:
            raise UnsuitableAlgorithmError("TSEMO requires initial data.")

        # Require one full observation for the hypervolume improvement
        Y = self.problem.data[self.problem.objectives.names]
        if len(Y.dropna(how="any")) < 2:
            raise UnsuitableAlgorithmError("TSEMO requires >1 full observation.")

        self._fit_model()

    def _fit_model(self):
        """Fit a GP model to the data."""
        X, Y = self.problem.get_XY()  # this drops incomplete observations
        self.model = fit_gp(X, Y, standardize=False)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the model posterior mean and std."""
        X = data[self.problem.inputs.names].values
        Ymean, Ystd = predict(self.model, X)
        return pd.DataFrame(
            np.concatenate([Ymean, Ystd], axis=1),
            columns=[f"mean_{n}" for n in self.problem.outputs.names]
            + [f"std_{n}" for n in self.problem.outputs.names],
            index=data.index,
        )

    def propose(self, n_proposals: Optional[int] = None) -> pd.DataFrame:
        """Propose new of experiments."""
        if n_proposals is None:
            n_proposals = 1

        prob = self.problem

        gp_sample = get_gp_samples(
            self.model,
            num_outputs=prob.n_outputs,
            n_samples=n_proposals,
            num_rff_features=self.rff_samples,
        )

        proposals = np.zeros((n_proposals, prob.n_inputs))
        for i in range(n_proposals):

            def f(x):
                return gp_sample(torch.tensor(x, **torch_kwargs))[i].numpy()

            x = run_nsga3(prob, f, pop_size=self.pop_size, generations=self.generations)
            Y = pd.DataFrame(f(x), columns=prob.outputs.names)
            Z = prob.objectives(Y).values

            # select the point with the highest hypervolume improvement
            Zd = prob.objectives(prob.data).dropna(how="any").values

            nadir = np.max(np.row_stack([Z, Zd]), axis=0) + 0.05
            hvi = hypervolume_improvement(Z, Zd, nadir)
            proposals[i] = x[np.argmax(hvi)]

        return pd.DataFrame(proposals, columns=prob.inputs.names)

    def predict_pareto_front(self, n_points: Optional[int] = None) -> pd.DataFrame:
        """Approximate the Pareto front using the posterior mean.

        Args:
            n_points (optional): Number of Pareto points. Default = 10 * n_objectives.
        """
        if n_points is None:
            n_points = 10 * self.problem.n_objectives

        def f(x):
            return predict(self.model, x)[0]

        X = run_nsga3(problem=self.problem, f=f, pop_size=n_points)
        Ymean, _ = predict(self.model, X)

        return pd.DataFrame(
            np.concatenate([X, Ymean], axis=1),
            columns=self.problem.inputs.names + self.problem.outputs.names,
        )

    def get_model_parameters(self) -> pd.DataFrame:
        """Get a dataframe of the model parameters."""
        params = get_gp_parameters(self.model)
        return params.rename(
            index={i: name for i, name in enumerate(self.problem.outputs.names)},
            columns={
                f"ls_{i}": f"ls_{name}"
                for i, name in enumerate(self.problem.inputs.names)
            },
        )

    def to_config(self) -> dict:
        """Serialize the algorithm settings to a dictionary."""
        return {
            "method": "TSEMO",
            "problem": self.problem.to_config(),
            "parameters": {
                "rff_samples": self.rff_samples,
                "generations": self.generations,
                "pop_size": self.pop_size,
            },
        }
