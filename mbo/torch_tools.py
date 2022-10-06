from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from opti.constraint import LinearEquality, LinearInequality
from opti.objective import CloseToTarget, Maximize, Minimize, Objective
from opti.parameter import Parameters
from opti.problem import Problem

torch_kwargs = {"dtype": torch.double}


def _set_gp_priors(model: Union[SingleTaskGP, ModelListGP]):
    """Set priors for the GP parameters.

    By default BoTorch places a highly informative prior on the kernel lengthscales,
    which easily leads to overfitting. Here we set a broader prior distribution for the
    lengthscale. The priors for the noise and signal variance are set more tightly.
    """
    model.likelihood.noise_covar.register_prior(
        "noise_prior",
        GammaPrior(concentration=2.0, rate=4.0),
        "noise",
    )
    model.covar_module.register_prior(
        "outputscale_prior",
        GammaPrior(concentration=2.0, rate=4.0),
        "outputscale",
    )
    model.covar_module.base_kernel.register_prior(
        "lengthscale_prior",
        GammaPrior(concentration=2.0, rate=0.2),
        "lengthscale",
    )


def fit_gp(X: np.ndarray, Y: np.ndarray, standardize: bool = True) -> SingleTaskGP:
    """Fit a multi-output GP model"""
    Xt = torch.tensor(X, **torch_kwargs)
    Yt = torch.tensor(Y, **torch_kwargs)

    model = SingleTaskGP(
        Xt, Yt, outcome_transform=Standardize(m=Y.shape[-1]) if standardize else None
    )
    _set_gp_priors(model)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def fit_gp_list(X: np.ndarray, Y: np.ndarray, standardize: bool = True) -> ModelListGP:
    """Fit a list of single-output GP models, handles missing data."""
    Xt = torch.tensor(X, **torch_kwargs)
    Yt = torch.tensor(Y, **torch_kwargs)

    models = []
    for m in range(Y.shape[-1]):
        s = np.isfinite(Y[:, m])
        model = SingleTaskGP(
            Xt[s],
            Yt[s, [m]].reshape(-1, 1),
            outcome_transform=Standardize(m=1) if standardize else None,
        )
        _set_gp_priors(model)
        models.append(model)

    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def get_gp_parameters(model: Union[SingleTaskGP, ModelListGP]) -> pd.DataFrame:
    """Get a dataframe with the GP hyperparameters."""
    params = []
    M = model.num_outputs
    if isinstance(model, SingleTaskGP):
        for m in range(M):
            p = {
                "noise": model.likelihood.noise[m].item() ** 0.5,
                "scale": model.covar_module.outputscale.view(-1)[m].item() ** 0.5,
                "nu": model.covar_module.base_kernel.nu,
            }
            lengthscales = model.covar_module.base_kernel.lengthscale.view(M, -1)[m]
            for i, ls in enumerate(lengthscales):
                p[f"ls_{i}"] = ls.item()
                p["mean"] = model.mean_module.constant[m].item()
            params.append(p)
    elif isinstance(model, ModelListGP):
        for m in range(M):
            mm = model.models[m]
            p = {
                "noise": mm.likelihood.noise.item() ** 0.5,
                "scale": mm.covar_module.outputscale.item() ** 0.5,
                "nu": mm.covar_module.base_kernel.nu,
            }
            lengthscales = mm.covar_module.base_kernel.lengthscale[0]
            for i, ls in enumerate(lengthscales):
                p[f"ls_{i}"] = ls.item()
                p["mean"] = mm.mean_module.constant.item()
            params.append(p)
    else:
        raise TypeError(f"Unknown model type {type(model)}")
    params = pd.DataFrame(params)
    params.index.name = "output"
    return params


def predict(
    model: Union[SingleTaskGP, ModelListGP], X: np.ndarray, chunk_size=1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the posterior mean and std. This is done in batches to keep memory low."""
    Ymean = []
    Ystd = []
    splits = np.arange(0, len(X), chunk_size)[1:]
    splits = np.split(X, splits)
    for Xs in splits:
        posterior = model.posterior(torch.tensor(Xs, **torch_kwargs))
        Ymean.append(posterior.mean.detach().numpy())
        Ystd.append(posterior.variance.detach().numpy() ** 0.5)
    Ymean = np.concatenate(Ymean)
    Ystd = np.concatenate(Ystd)
    return Ymean, Ystd


def make_constraints(problem: Problem, normalize: bool = False) -> dict:
    """Convert bounds and constraints to the form required by BoTorch.

    Args:
        problem: Problem definition
        normalize: transform bounds and constraints to the range [0, 1]

    Returns:
        dict with bounds, equalities and inequalities
        where the constraints are lists of tuples (indices, weights, right-hand-side)
    """
    equalities = []
    inequalities = []

    if problem.constraints is not None:
        for c in problem.constraints:
            if not isinstance(c, (LinearEquality, LinearInequality)):
                raise ValueError("Botorch only supports linear constraints")

            indices = [problem.inputs.names.index(n) for n in c.names]
            lhs = c.lhs
            rhs = c.rhs

            if normalize:
                xlo, xhi = problem.inputs.bounds[c.names].values
                rhs = c.rhs - lhs @ xlo
                lhs = lhs * (xhi - xlo)

            # botorch assumes lhs * x >= rhs, whereas opti assumes lhs * x <= lhs
            constraint = [
                (
                    torch.tensor(indices, dtype=torch.long),
                    -torch.tensor(lhs, **torch_kwargs),
                    -rhs,
                )
            ]
            if isinstance(c, LinearEquality):
                equalities += constraint
            if isinstance(c, LinearInequality):
                inequalities += constraint

    if normalize:
        D = len(problem.inputs)
        bounds = torch.stack([torch.zeros(D), torch.ones(D)])
    else:
        bounds = torch.tensor(problem.inputs.bounds.values, **torch_kwargs)

    # set empty constraint lists to None so that Botorch's expensive polytope sampling is only used when neccessary
    if len(equalities) == 0:
        equalities = None
    if len(inequalities) == 0:
        inequalities = None

    return {"bounds": bounds, "equalities": equalities, "inequalities": inequalities}


def make_objective(
    objective: Objective, outputs: Parameters, maximize: bool = True
) -> Callable:
    """Make a callable for converting output values to objective values.

    Args:
        objective: Objective for which to generate a callable
        outputs: List of all outputs in order to identify the position of output for the given objective.
        maximize: Optimization direction. Use `maximize=True` if the optimizer expects to maxize objectives.

    Returns:
        Callable: Returns the objective values for objective m, given a tensor of output values of shape (..., M).
    """
    sign = -1 if maximize else 1
    i = outputs.names.index(objective.name)
    if isinstance(objective, Minimize):
        return lambda Y: sign * (Y[..., i] - objective.target)
    elif isinstance(objective, Maximize):
        return lambda Y: -sign * (Y[..., i] - objective.target)
    elif isinstance(objective, CloseToTarget):
        return lambda Y: sign * (Y[..., i] - objective.target).abs()
    else:
        raise ValueError(f"Unknown objective function {objective}")


def to_mng_format(data: pd.DataFrame, config):
    return {
        "data": data.to_dict(),
        "config": config,
    }


def from_mng_format(param_dict, Algorithm):
    cfg = param_dict["config"]
    problem = Problem.from_config(cfg["problem"])

    alg = Algorithm(
        problem=problem,
        data=pd.DataFrame(param_dict["data"]),
        **{k: v for k, v in cfg["parameters"].items()},
    )

    return alg
