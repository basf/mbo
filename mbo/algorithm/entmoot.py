import os
from typing import Optional

import gurobipy
import numpy as np
import opti
import pandas as pd
from entmoot.optimizer import Optimizer as EntmootOptimizer
from entmoot.optimizer.gurobi_utils import get_core_gurobi_model
from entmoot.space.space import Categorical, Integer, Real, Space

from mbo.algorithm import Algorithm


class Entmoot(Algorithm):
    """
    ENTMOOT (ENsemble Tree MOdel Optimization Tool) is a framework to handle tree-based models
    together with mixed-integer programming for use in Bayesian optimization.

    References:
        - Thebelt 2020, ENTMOOT: A Framework for Optimization over Ensemble Tree Models
        - Thebelt 2021, Multi-objective constrained optimization for energy applications via tree ensembles
    """

    def __init__(self, problem: opti.Problem, **kwargs):
        super().__init__(problem)

        # set up skopt parameter space
        dimensions = []
        for parameter in self.problem.inputs:
            if isinstance(parameter, opti.Continuous):
                dimensions.append(Real(*parameter.bounds, name=parameter.name))
            elif isinstance(parameter, opti.Categorical):
                dimensions.append(Categorical(parameter.domain, name=parameter.name))
            elif isinstance(parameter, opti.Discrete):
                # skopt only supports integer variables [1, 2, 3, 4], not discrete ones [1, 2, 4]
                # We handle this by rounding the proposals
                dimensions.append(Integer(*parameter.bounds, name=parameter.name))
        self.space = Space(dimensions)

        # set up Entmoot optimizer
        self.entmoot_optimizer = EntmootOptimizer(
            dimensions=dimensions,
            base_estimator="ENTING",
            n_initial_points=0,
            num_obj=problem.n_objectives,
            base_estimator_kwargs=kwargs.get("base_est_params", {}),
        )

        self._gurobi_env = None

        # non-lazy model fitting because skopt layout deviates too much from bo
        self._fit_model()

    def _initialize_problem(self) -> None:
        if self.problem.data is None:
            raise ValueError("Entmoot requires initial data.")

    def _fit_model(self) -> None:
        # clear Entmoot's data before calling tell() to not add duplicate data
        self.entmoot_optimizer.Xi = []
        self.entmoot_optimizer.yi = []

        data = self.problem.get_data()
        x = data[self.problem.inputs.names].to_numpy().tolist()
        y = data[self.problem.outputs.names].to_numpy().squeeze().tolist()
        self.entmoot_optimizer.tell(x=x, y=y)

    def _get_gurobi_env(self) -> gurobipy.Env.CloudEnv:
        """Lazy creation of Gurobi env"""
        if self._gurobi_env is None:
            # set up gurobi env
            self._gurobi_env = gurobipy.Env.CloudEnv(
                logfilename="gurobi.log",
                accessID=os.environ["GRB_CLOUDACCESSID"],
                secretKey=os.environ["GRB_CLOUDKEY"],
                pool=os.environ["GRB_CLOUDPOOL"],
            )
        return self._gurobi_env

    def _get_gurobi_model(self):
        """Get the gurobi core model including constraints."""
        model = get_core_gurobi_model(self.space, env=self._get_gurobi_env())

        # migrate constraints from opti to gurobi
        if self.problem.constraints:
            for c in self.problem.constraints:
                if isinstance(c, opti.constraint.LinearInequality):
                    coef = {x: a for (x, a) in zip(c.names, c.lhs)}
                    model.addConstr(
                        (
                            sum(
                                coef[v.varname] * v
                                for v in model.getVars()
                                if v.varname in coef
                            )
                            <= c.rhs
                        ),
                        name="linear-inequality",
                    )
                elif isinstance(c, opti.constraint.LinearEquality):
                    coef = {x: a for (x, a) in zip(c.names, c.lhs)}
                    model.addConstr(
                        (
                            sum(
                                coef[v.varname] * v
                                for v in model.getVars()
                                if v.varname in coef
                            )
                            == c.rhs
                        ),
                        name="linear-equality",
                    )
                elif isinstance(c, opti.constraint.NChooseK):
                    # Big-M implementation of n-choose-k constraint
                    y = model.addVars(c.names, vtype=gurobipy.GRB.BINARY)
                    model.addConstrs(
                        (
                            y[v.varname] * v.lb <= v
                            for v in model.getVars()
                            if v.varname in c.names
                        ),
                        name="n-choose-k-constraint LB",
                    )
                    model.addConstrs(
                        (
                            y[v.varname] * v.ub >= v
                            for v in model.getVars()
                            if v.varname in c.names
                        ),
                        name="n-choose-k-constraint UB",
                    )
                    model.addConstr(
                        y.sum() == c.max_active, name="max active components"
                    )
                else:
                    raise ValueError(f"Constraint of type {type(c)} not supported.")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xn = X[self.problem.inputs.names].to_numpy().tolist()
        mean, std = self.entmoot_optimizer.predict_with_est(Xn)
        mean = pd.DataFrame(
            np.array(mean).T,
            columns=[f"mean_{n}" for n in self.problem.outputs.names],
        )
        # Entmoot only returns a single std list. Need to duplicate for all outputs
        std = pd.DataFrame(
            np.array([std] * self.problem.n_outputs).T,
            columns=[f"std_{n}" for n in self.problem.outputs.names],
        )
        Y_pred = pd.concat([mean, std], axis=1)
        Y_pred.index = X.index
        return Y_pred

    def predict_pareto_front(self, n_points: Optional[int] = None) -> pd.DataFrame:
        if n_points is None:
            n_points = 10 * self.problem.n_objectives

        pf_res = self.entmoot_optimizer.predict_pareto(
            sampling_strategy="random",
            num_samples=n_points,
            add_model_core=self._get_gurobi_model(),
            gurobi_env=self._get_gurobi_env(),
        )

        pf_list = [list(x) + y for x, y in pf_res]  # convert to a list of lists
        return pd.DataFrame(
            pf_list, columns=self.problem.inputs.names + self.problem.outputs.names
        )

    def propose(self, n_proposals: Optional[int] = None) -> pd.DataFrame:
        if n_proposals is None:
            n_proposals = 1

        X_next = self.entmoot_optimizer.ask(
            n_points=n_proposals,
            add_model_core=self._get_gurobi_model(),
            gurobi_env=self._get_gurobi_env(),
        )
        X_next = np.atleast_2d(X_next)
        return self.problem.inputs.to_df(X_next, to_numeric=True)

    def get_model_parameters(self) -> pd.DataFrame:
        """Get the (hyper)parameters of the surrogate model."""
        prob = self.problem
        opt = self.entmoot_optimizer
        est = opt.base_estimator_

        # Make sure the EntingRegressor is fitted. This doesn't happen on self._fit_model()!
        est.fit(opt.space.transform(opt.Xi), opt.yi)

        # Estimate the feature importances from the number of splits in which a variable is used.
        df = pd.DataFrame(columns=[f"importance_{n}" for n in prob.inputs.names])
        for output, regressor in zip(prob.outputs.names, est.regressor_):
            df.loc[output] = regressor.feature_importances_

        df = (df.T / df.max(axis=1)).T  # normalize
        return df

    def to_config(self) -> dict:
        """Serialize the algorithm settings to a dictionary."""
        return {
            "method": "ENTMOOT",
            "problem": self._problem.to_config(),
            "parameters": {},
        }
