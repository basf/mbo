from typing import Optional

import numpy as np
import opti
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor

from mbo.algorithm import Algorithm
from mbo.error import UnsuitableAlgorithmError


class RandomForest(Algorithm):
    """Surrogate-assisted optimization with RandomForests.

    The algorithm can handle constrained mixed variable problems with single and multiple objectives.
    It is intended as a cheap alternative to ENTMOOT.

    Description:
    - Model: RandomForest with a distance-based uncertainty measure.
    - Proposals: BayesOpt with random Chebyshef scalarization and optimistic confidence bound.
    - Optimization: Brute-force.
    - Pareto approximation: Not implemented.
    """

    def __init__(self, problem: opti.Problem, n_samples: int = 10000, **kwargs):
        """RandomForest algorithm.

        Args:
            problem: Problem definition.
            n_samples: Number of samples for the brute-force optimization.
        """
        super().__init__(problem)
        self.n_samples = n_samples
        self._initialize_problem()

    def _initialize_problem(self) -> None:
        # Check for initial data
        if self.problem.data is None:
            raise UnsuitableAlgorithmError("RandomForest requires initial data.")

        # Estimate range of outputs from observed values.
        Y = self.problem.get_Y()
        self.y_range = Y.max(axis=0) - Y.min(axis=0)

        # Check for output constraints
        if self.problem.output_constraints is not None:
            raise UnsuitableAlgorithmError(
                "Output constraints not implemented for RandomForest."
            )

    def _fit_model(self) -> None:
        X = self.problem.data[self.problem.inputs.names]
        Xt = self.problem.inputs.transform(X, categorical="dummy-encode").values
        Y = self.problem.get_Y().squeeze()
        self.model = RandomForestRegressor().fit(Xt, Y)

    def _min_distance(self, X1: pd.DataFrame, X2: pd.DataFrame):
        """Matrix of L1-norm-distances between each point in X1 and X2"""
        kwargs = dict(
            continuous="normalize", discrete="normalize", categorical="onehot-encode"
        )
        inputs = self.problem.inputs
        Xt1 = inputs.transform(X1, **kwargs)
        Xt2 = inputs.transform(X2, **kwargs)

        # set all onehot-encode values to 0.5 so that the L1-distance becomes 1
        cat_cols = [c for c in Xt1 if "§" in c]
        Xt1[cat_cols] /= 2
        Xt2[cat_cols] /= 2

        D = torch.cdist(torch.tensor(Xt1.values), torch.tensor(Xt2.values)).numpy()
        return D.min(axis=1)

    def _uncertainty(self, X: pd.DataFrame, X_data: Optional[pd.DataFrame] = None):
        """Uncertainty estimate ~ distance to the closest data point."""
        if X_data is None:
            X_data = self.problem.data
        min_dist = self._min_distance(X, X_data)
        min_dist = min_dist / self.problem.n_inputs
        return self.y_range * min_dist[:, np.newaxis]

    def predict(self, X: pd.DataFrame):
        Xt = self.problem.inputs.transform(X, categorical="dummy-encode")
        Ym = self.model.predict(Xt.values)
        Ys = self._uncertainty(X)
        return pd.DataFrame(
            np.c_[Ym, Ys],
            columns=[f"mean_{n}" for n in self.problem.outputs.names]
            + [f"std_{n}" for n in self.problem.outputs.names],
            index=X.index,
        )

    def propose(
        self,
        n_proposals: Optional[int] = None,
        kappa: float = 0.5,
        weights: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if n_proposals is None:
            n_proposals = 1

        if weights is None:
            n_objectives = len(self.problem.objectives)
            weights = opti.sampling.simplex.sample(n_objectives, n_proposals)
        else:
            weights = np.atleast_2d(weights)

        proposals = []

        for w in weights:
            X = self.problem.sample_inputs(self.n_samples)
            Xt = self.problem.inputs.transform(X, categorical="dummy-encode")
            Y_mean = pd.DataFrame(
                self.model.predict(Xt.values), columns=self.problem.outputs.names
            )
            Z_mean = self.problem.objectives(Y_mean)

            # take the pending proposals into account for the uncertainty
            X_data = self.problem.data[self.problem.inputs.names]
            X_data = pd.concat([X_data] + proposals, axis=0)
            Y_std = self._uncertainty(X, X_data)
            Z_std = Y_std  # may not be true for close-to-target objectives

            # optimistic confidence bound
            cb = Z_mean.to_numpy() - kappa * Z_std
            # normalize so that the Pareto front is in the unit range
            s = opti.metric.is_pareto_efficient(cb)
            cb_min = cb[s].min(axis=0)
            cb_max = cb[s].max(axis=0)
            cb = (cb - cb_min) / np.clip(cb_max - cb_min, 1e-7, None)

            # weighted max norm
            A = np.max(w * cb, axis=1)
            best = np.argmin(A)
            proposals.append(X.iloc[[best]])

        return pd.concat(proposals)

    def get_model_parameters(self) -> pd.DataFrame:
        # get the columns labels for the one-hot encoded inputs in case there are categoricals
        cols = self.problem.inputs.transform(
            self.problem.data, categorical="dummy-encode"
        ).columns

        # return the feature importances
        m = self.problem.n_outputs
        params = pd.DataFrame(
            index=self.problem.outputs.names,
            data=np.tile(self.model.feature_importances_, reps=(m, 1)),
            columns=cols,
        )
        params.index.name = "output"
        return params

    def to_config(self) -> dict:
        return {
            "method": "RandomForest",
            "problem": self.problem.to_config(),
            "parameters": {"n_samples": self.n_samples},
        }
