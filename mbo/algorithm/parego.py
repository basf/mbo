from typing import Optional

import numpy as np
import opti
import pandas as pd
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
from botorch.optim.optimize import optimize_acqf_list
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.transforms import normalize

from mbo.algorithm import Algorithm
from mbo.error import UnsuitableAlgorithmError
from mbo.optimize import run_nsga3
from mbo.torch_tools import (
    fit_gp,
    fit_gp_list,
    get_gp_parameters,
    make_constraints,
    make_objective,
    predict,
    torch_kwargs,
)


class ParEGO(Algorithm):
    """Multi-objective Bayesian optimization with Gaussian prossess.

    This algorithm supports multi-objective optimization over all-continuous and linearly constrained input spaces.
    Output constraints are supported.

    Description:
    - Model: Gaussian processes with a Matern 5/2 kernel.
    - Proposals: BayesOpt with random Chebyshef scalarization and expected improvement.
    - Optimization: L-BFGS-B.
    - Pareto approximation: Chebyshef scalarization with weigths on a regular grid.

    References:
    - Knowles 2006, ParEGO: A hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems
    - Dalton 2020, Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization
    """

    def __init__(
        self,
        problem: opti.Problem,
        mc_samples: int = 1024,
        restarts: int = 3,
    ):
        """ParEGO algorithm.

        Args:
            problem: Problem definition.
            mc_samples: Number of MC samples for estimating the acquisition function.
            restarts: Number of optimization restarts.
        """
        super().__init__(problem)
        self.restarts = restarts
        self.mc_samples = mc_samples
        self.sampler = SobolQMCNormalSampler(num_samples=mc_samples)

    def _initialize_problem(self) -> None:
        # Require multiple objectives (scalarization needs multiple objectives)
        if self.problem.n_objectives < 2:
            raise UnsuitableAlgorithmError("ParEGO requires multiple objectives.")
        self.objective_funcs = [
            make_objective(obj, self.problem.outputs, maximize=True)
            for obj in self.problem.objectives
        ]

        # Check for output constraints
        if self.problem.output_constraints is not None:
            self.constraint_funcs = [
                make_objective(obj, self.problem.outputs)
                for obj in self.problem.output_constraints
            ]

        # Require all continuous inputs (would require special GP kernel otherwise)
        for p in self.problem.inputs:
            if not (isinstance(p, opti.Continuous)):
                raise UnsuitableAlgorithmError(
                    "ParEGO can only optimize over continuous inputs."
                )

        # Require initial data
        if self.problem.data is None:
            raise UnsuitableAlgorithmError("ParEGO requires initial data.")

        # Require at least one full observation of all objectives for the EI scalarization
        Y = self.problem.get_Y().astype(float)
        if not np.isfinite(Y).all(axis=1).any():
            raise UnsuitableAlgorithmError(
                "ParEGO requires at least one full observation."
            )

        # Require no nonlinear constraint (BoTorch cannot handle them).
        # We'll be working with normalized inputs, thus transform bounds and constraints as well
        try:
            constraints = make_constraints(self.problem, normalize=True)
            self.bounds = constraints["bounds"]
            self.equalities = constraints["equalities"]
            self.inequalities = constraints["inequalities"]
        except ValueError as ve:
            raise UnsuitableAlgorithmError(
                f"ParEGO can only handle linear constraints. {str(ve)}"
            )

    def _get_parego_objective(self, weights):
        # get the observed objective bounds for normalization
        Z = self.problem.objectives(self.problem.data).values
        Z_bounds = torch.tensor(
            np.array([np.nanmin(Z, axis=0), np.nanmax(Z, axis=0)]), **torch_kwargs
        )
        weights = torch.tensor(weights, **torch_kwargs)

        def augmented_chebyshef(Y: torch.Tensor) -> torch.Tensor:
            Z = torch.stack([f(Y) for f in self.objective_funcs], dim=-1)
            wZ = weights * normalize(Z, Z_bounds)
            return wZ.min(axis=-1).values + 0.01 * wZ.sum(axis=-1)

        if self.problem.output_constraints is None:
            objective = GenericMCObjective(augmented_chebyshef)
        else:
            objective = ConstrainedMCObjective(
                objective=augmented_chebyshef,
                constraints=self.constraint_funcs,
            )

        # best objective value observed so far
        Y = self.problem.get_Y()
        Y = torch.tensor(Y[np.isfinite(Y).all(axis=1)], **torch_kwargs)
        best = objective(Y).max()
        return objective, best

    def _normalize_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.problem.inputs.transform(data, continuous="normalize")

    def _unnormalize_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed = []
        for p in self.problem.inputs:
            s = data[p.name]
            transformed.append(p.from_unit_range(s))
        return pd.concat(transformed, axis=1)

    def _fit_model(self) -> None:
        data = self.problem.get_data()
        Xn = self._normalize_inputs(data).values
        Y = data[self.problem.outputs.names].values
        if np.all(np.isfinite(Y)):
            self.model = fit_gp(Xn, Y)
        else:  # missing outputs --> use a list of GPs
            self.model = fit_gp_list(Xn, Y)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        Xn = self._normalize_inputs(data).values
        Ymean, Ystd = predict(self.model, Xn)
        return pd.DataFrame(
            np.concatenate([Ymean, Ystd], axis=1),
            columns=[f"mean_{n}" for n in self.problem.outputs.names]
            + [f"std_{n}" for n in self.problem.outputs.names],
            index=data.index,
        )

    def propose(self, n_proposals: Optional[int] = None) -> pd.DataFrame:
        if n_proposals is None:
            n_proposals = 1

        weights = opti.sampling.simplex.sample(self.problem.n_objectives, n_proposals)

        # create a list of acquisition functions with random Chebyshef scalarizations
        acq_func_list = []
        for w in weights:
            objective, best = self._get_parego_objective(w)
            acq_func = qExpectedImprovement(
                model=self.model,
                objective=objective,
                best_f=best,
                sampler=self.sampler,
            )
            acq_func_list.append(acq_func)

        # greedy-sequentially optimize each acquistion, setting previous candidates as pending
        candidates, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=self.bounds,
            equality_constraints=self.equalities,
            inequality_constraints=self.inequalities,
            num_restarts=self.restarts,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": 200},
        )

        proposals = pd.DataFrame(
            candidates.detach().numpy(),
            columns=self.problem.inputs.names,
            index=np.arange(n_proposals) + len(self.problem.get_data()),
        )
        return self._unnormalize_inputs(proposals)

    def predict_pareto_front(self, n_points: Optional[int] = None) -> pd.DataFrame:
        """Approximate the Pareto front using the posterior mean.

        Args:
            n_points (optional): Number of Pareto points. Default = 10 * n_objectives.
        """
        if n_points is None:
            n_points = 10 * self.problem.n_objectives

        def f(x):
            x = self._normalize_inputs(
                pd.DataFrame(x, columns=self.problem.inputs.names)
            )
            return predict(self.model, x.values)[0]

        X = run_nsga3(problem=self.problem, f=f, pop_size=n_points)
        X = pd.DataFrame(X, columns=self.problem.inputs.names)
        Y, _ = predict(self.model, self._normalize_inputs(X).values)
        Y = pd.DataFrame(Y, columns=self.problem.outputs.names)
        return pd.concat([X, Y], axis=1)

    def get_model_parameters(self) -> pd.DataFrame:
        """Get a dataframe of the model parameters."""
        params = get_gp_parameters(self.model)
        # set parameter names
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
            "method": "ParEGO",
            "problem": self.problem.to_config(),
            "parameters": {
                "restarts": self.restarts,
                "mc_samples": self.mc_samples,
            },
        }
