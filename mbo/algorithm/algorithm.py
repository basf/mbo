from typing import Optional

import opti
import pandas as pd
from tqdm import tqdm


class Algorithm:
    """Base class for (Bayesian) optimization algorithms."""

    def __init__(self, problem: opti.Problem):
        self._problem = problem
        self._model = None
        self._initialize_problem()

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, problem):
        self._problem = problem
        self._initialize_problem()

    @property
    def model(self):
        if not self._model:
            self._fit_model()
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def _fit_model(self) -> None:
        """Fit the surrogate model to the available data. This should call _tune_model() if neccessary."""
        pass

    def _tune_model(self) -> None:
        """Tune the surrogate model."""
        pass

    def _initialize_problem(self) -> None:
        """Additional problem related initializations or checks."""
        pass

    def copy(self, data: Optional[pd.DataFrame] = None) -> "Algorithm":
        """Creates a copy of the optimizer where the data is possibly replaced."""
        new_opt = self.from_config(self.to_config())
        if data is not None:
            new_opt._problem.set_data(data)
            new_opt._fit_model()
        return new_opt

    def add_data_and_fit(self, data: pd.DataFrame) -> None:
        """Add new data points and retune & refit the model. Also known as Tell."""
        self._problem.add_data(data)
        self._tune_model()
        self._fit_model()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the posterior mean and standard deviation at given inputs X."""
        raise NotImplementedError

    def predict_pareto_front(self, n_points: Optional[int] = None) -> pd.DataFrame:
        """Find the optimizer x of the posterior mean E[f(x)].

        For single-objective problems the single optimizer is determined.
        For multi-objective problems a finite representation of the Pareto front is computed.

        Returns:
            pd.DataFrame: Dataframe containing [X, Y] where X is the optimizer(s) and Y = self.predict(X).
        """
        raise NotImplementedError

    def propose(self, n_proposals: Optional[int] = None) -> pd.DataFrame:
        """Propose a set of experiments according to the algorithm. Also known as Ask."""
        raise NotImplementedError

    def run(
        self, n_proposals: int = 1, n_steps: int = 10, show_progress: bool = False
    ) -> None:
        """Run the algorithm to optimize the problem."""
        if self._problem.f is None:
            raise ValueError(
                "problem.f() not defined. For external function evaluations use propose() instead."
            )

        for _ in tqdm(range(n_steps), disable=not show_progress):
            X = self.propose(n_proposals)
            Y = self._problem.f(X)
            self.add_data_and_fit(pd.concat([X, Y], axis=1))

    def get_model_parameters(self) -> pd.DataFrame:
        """Get the (hyper)parameters of the surrogate model."""
        raise NotImplementedError

    def to_config(self) -> dict:
        """Serialize the algorithm settings to a configuration dict.

        This should include model hyperparameters so that an algorithm loaded from a config does not need to be tuned again.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict):
        """Create an algorithm instance from a configuration dict."""
        problem = opti.Problem.from_config(config["problem"])
        parameters = config.get("parameters", {})
        return cls(problem, **parameters)
