from typing import Callable

import numpy as np
import opti
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.factory import get_reference_directions, get_termination
from pymoo.optimize import minimize


def run_nsga3(
    problem: opti.Problem, f: Callable, pop_size=100, generations=100
) -> np.ndarray:
    """Run NSGA-3 for given problem and evaluation function f."""

    class PymooProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=problem.n_inputs,
                n_obj=problem.n_objectives,
                n_constr=0 if problem.constraints is None else len(problem.constraints),
                xl=problem.inputs.bounds.loc["min"].values,
                xu=problem.inputs.bounds.loc["max"].values,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            X = pd.DataFrame(x, columns=problem.inputs.names)
            Y = pd.DataFrame(f(x), columns=problem.outputs.names)
            out["F"] = problem.objectives(Y).values
            if problem.constraints:
                out["G"] = problem.constraints.satisfied(X)
            return out

    ref_dirs = get_reference_directions("energy", problem.n_objectives, pop_size)

    result = minimize(
        problem=PymooProblem(),
        algorithm=NSGA3(ref_dirs),
        termination=get_termination("n_gen", generations),
    )

    return result.X
