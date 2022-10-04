import logging
import logging.config
import warnings
from typing import Callable, Dict, Optional, Union

import numpy as np
import opti
import pandas as pd
from doe import find_local_max_ipopt
from doe.sampling import OptiSampling, Sampling
from doe.utils import ProblemContext, get_formula_from_string
from formulaic import Formula

from mbo.algorithm.algorithm import Algorithm

# filter warning for linearization of NChooseK constraints because the default value is True here.
warnings.filterwarnings("ignore", message="linearized versions of NChooseK constraints")

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "default": {"format": "%(asctime)s [%(process)s] %(levelname)s: %(message)s"}
    },
    "handlers": {
        "console": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "level": "INFO",
        }
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "gunicorn": {"propagate": True},
        "gunicorn.access": {"propagate": True},
        "gunicorn.error": {"propagate": True},
        "uvicorn": {"propagate": True},
        "uvicorn.access": {"propagate": True},
        "uvicorn.error": {"propagate": True},
    },
}

logger = logging.getLogger(__name__)

logger.debug("BO algorithm DoE logger active.")


class DOptimalDesign(Algorithm):
    """Generation of a D-optimal design.

    This algorithm uses IPOPT/cyipopt to find a D-optimal design for a given problem and model.

    The following steps are taken:
        1. Number of experiments is determined as n_experiments = n_model_terms - n_zero_eigenvalues + 3
        where n_zero_eigenvalues is the number of necessarily vanishing eigenvalues of the information matrix.
        2. Find a value for IPOPT option 'maxiter' to prevent very long computation times.
        3. Pass the objective (logdet of regularized fisher matrix) and the constraints to IPOPT and start optimization.
        4. Return the results as pandas.DataFrame.
    """

    def __init__(
        self,
        problem: opti.Problem,
        model_type: Union[str, Formula] = "linear",
        tol: float = 0,
        delta: float = 1e-7,
        ipopt_options: Optional[Dict] = None,
        jacobian_building_block: Optional[Callable] = None,
        sampling: Union[np.ndarray, Sampling] = OptiSampling,
        fixed_experiments: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            problem (opti.Problem): problem containing decision variables and constraints.
            model_type (str, Formula): keyword or formulaic Formula describing the model.
            tol (float): Tolerance for equality/NChooseK constraint violation. Default value is 0.
            delta (float): Regularization parameter. Default value is 1e-7.
            ipopt_options (Dict): options for IPOPT. For more information see [this link](https://coin-or.github.io/Ipopt/OPTIONS.html).
            jacobian_building_block (Callable): Only needed for models of higher order than 3. derivatives of each model term with respect to each input variable.
            sampling (Sampling, np.ndarray): Sampling class or a np.ndarray object containing the initial guess.
            fixed_experiments (np.ndarray): numpy array containing experiments that will definitely part of the design.
                Values are set before the optimization.
        """
        # test if model_type is parseable
        problem_context = ProblemContext(problem)
        get_formula_from_string(model_type=model_type, problem_context=problem_context)

        self.model_type = model_type

        _ipopt_options = {"max_cpu_time": 2700.0}  # limit max run time to 45 minutes
        if ipopt_options is not None:
            _ipopt_options.update(ipopt_options)

        self.kwargs = {
            "tol": tol,
            "delta": delta,
            "ipopt_options": _ipopt_options,
            "jacobian_building_block": jacobian_building_block,
            "sampling": sampling,
            "fixed_experiments": fixed_experiments,
        }
        super().__init__(problem)

    def propose(self, n_proposals: Optional[int] = None) -> pd.DataFrame:
        """Generate a D-optimal experimental design.

        Args:
            n_proposals (int, None): Number of experiments. If None, the number of experiments is determined automatically.

        Returns:
            A pandas.DataFrame with the experimental proposals.
        """
        df = find_local_max_ipopt(
            problem=self.problem,
            model_type=self.model_type,
            n_experiments=n_proposals,
            **self.kwargs,
        )
        df.index = [i for i in range(len(df))]
        if self.problem.data is not None:
            df.index += len(self.problem.data)
        return df

    def to_config(self) -> dict:
        """Serialize the algorithm settings to a dictionary."""
        # sampling defaults to OptiSampling, which is not serializable
        kwargs_parameters = self.kwargs
        remove_parameters = ["sampling"]
        for c, p in enumerate(remove_parameters):
            if c == 0:
                logger.info("Serialization not possible for following objects:")
            kwarg_value = kwargs_parameters.pop(p, None)
            logger.info(f"{p}: {kwarg_value}")
        return {
            "method": "DOptimalDesign",
            "problem": self._problem.to_config(),
            "parameters": {"model_type": self.model_type, **kwargs_parameters},
        }
