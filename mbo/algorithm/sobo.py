from typing import Optional

import numpy as np
import opti
import pandas as pd
import torch
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.sampling.samplers import SobolQMCNormalSampler

from mbo.algorithm import Algorithm
from mbo.error import UnsuitableAlgorithmError
from mbo.torch_tools import (
    fit_gp,
    get_gp_parameters,
    make_constraints,
    make_objective,
    predict,
    torch_kwargs,
)


class SOBO(Algorithm):
    """Single-objective Bayesian optimization with Gaussian prossess.

    This algorithm supports single-objective optimization over all-continuous and linearly constrained input spaces.

    Description:
    - Model: Gaussian processes with a Matern 5/2 kernel.
    - Proposals: BayesOpt with expected improvement.
    - Optimization: L-BFGS-B.
    - Pareto approximation: Single-objective optimization.
    """

    def __init__(
        self,
        problem: opti.Problem,
        acquisition="EI",
        mc_samples: int = 256,
        restarts: int = 10,
    ):
        """SOBO algorithm.

        Args:
            problem: Problem definition.
            acquisition: Acquistion function. Currently, only "EI" is supported.
            mc_samples: Number of MC samples for estimating the acquisition function.
            restarts: Number of optimization restarts.
        """
        super().__init__(problem)
        self.acquisition = acquisition
        self.restarts = restarts
        self.mc_samples = mc_samples
        self.sampler = SobolQMCNormalSampler(num_samples=mc_samples)
        self._fit_model()

    def _initialize_problem(self) -> None:
        # Require a single objective
        if self.problem.n_objectives > 1:
            raise UnsuitableAlgorithmError("SOBO requires a single objective.")

        # Require all continuous inputs
        for p in self.problem.inputs:
            if not (isinstance(p, opti.Continuous)):
                raise UnsuitableAlgorithmError(
                    "SOBO can only optimize over continuous inputs."
                )

        # Require initial data
        if self.problem.data is None:
            raise UnsuitableAlgorithmError("SOBO requires initial data.")

        # Require no nonlinear constraint (BoTorch cannot handle them).
        # We'll be working with normalized inputs, thus, transform bounds and constraints as well.
        try:
            constraints = make_constraints(self.problem, normalize=True)
            self.bounds = constraints["bounds"]
            self.equalities = constraints["equalities"]
            self.inequalities = constraints["inequalities"]
        except ValueError:
            raise ValueError("SOBO can only handle linear constraints.")

    def _normalize_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.problem.inputs.transform(data, continuous="normalize")

    def _unnormalize_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed = []
        for p in self.problem.inputs:
            s = data[p.name]
            transformed.append(p.from_unit_range(s))
        return pd.concat(transformed, axis=1)

    def _fit_model(self) -> None:
        """Fit a GP model to the data."""
        data = self.problem.get_data().dropna(subset=self.problem.outputs.names)
        Xn = self._normalize_inputs(data).values
        Y = data[self.problem.outputs.names].values
        self.model = fit_gp(Xn, Y)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the model posterior mean and std."""
        Xn = self._normalize_inputs(data).values
        Ymean, Ystd = predict(self.model, Xn)
        return pd.DataFrame(
            np.concatenate([Ymean, Ystd], axis=1),
            columns=[f"mean_{n}" for n in self.problem.outputs.names]
            + [f"std_{n}" for n in self.problem.outputs.names],
            index=data.index,
        )

    def propose(self, n_proposals: Optional[int] = None) -> pd.DataFrame:
        """Generate new experimental proposals."""
        if n_proposals is None:
            n_proposals = 1

        Y = self.problem.get_Y()
        Yt = torch.tensor(Y, **torch_kwargs)

        # greedy-sequentially optimize the acquistion for each proposal, setting previous candidates as pending
        objective = GenericMCObjective(
            make_objective(
                self.problem.objectives[0], self.problem.outputs, maximize=True
            )
        )
        candidates, _ = optimize_acqf_list(
            acq_function_list=[
                qExpectedImprovement(
                    model=self.model,
                    objective=objective,
                    best_f=objective(Yt).max(),
                    sampler=self.sampler,
                )
                for _ in range(n_proposals)
            ],
            bounds=self.bounds,
            equality_constraints=self.equalities,
            inequality_constraints=self.inequalities,
            num_restarts=self.restarts,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": 200},
        )
        X = pd.DataFrame(
            candidates.detach().numpy(),
            columns=self.problem.inputs.names,
            index=np.arange(n_proposals) + len(self.problem.get_data()),
        )
        return self._unnormalize_inputs(X)

    def predict_pareto_front(self, *args, **kwargs) -> pd.DataFrame:
        """Return the optimum of the posterior mean."""
        # There is currently no way to PosteriorMean with a generic objectives,
        # hence we use qUpperConfidenceBound with beta=0 (mean) and q=1.
        objective = GenericMCObjective(
            make_objective(self.problem.objectives[0], self.problem.outputs)
        )
        acq_func = qUpperConfidenceBound(
            model=self.model,
            beta=0,  # posterior mean
            objective=objective,
            sampler=self.sampler,
        )
        xopt, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            equality_constraints=self.equalities,
            inequality_constraints=self.inequalities,
            num_restarts=self.restarts,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": 1000},
        )
        x = xopt.detach().numpy()
        Ymean, _ = predict(self.model, x)
        Y = pd.DataFrame(Ymean, columns=self.problem.outputs.names)
        X = pd.DataFrame(x, columns=self.problem.inputs.names)
        X = self._unnormalize_inputs(X)
        return pd.concat([X, Y], axis=1)

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
            "method": "SOBO",
            "problem": self.problem.to_config(),
            "parameters": {
                "acquisition": self.acquisition,
                "restarts": self.restarts,
                "mc_samples": self.mc_samples,
            },
        }
