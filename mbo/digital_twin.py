import copy
import shutil
import warnings
from importlib import import_module
from typing import Any, Callable, List, Optional

import numpy as np
import opti
import pandas as pd
from opti import Problem
from tqdm import tqdm

import mbo as bo
from mbo.algorithm import Algorithm

try:
    # to have parallel runs
    import multiprocess as mp
except ImportError:
    mp = None


class NotLogger:
    def set_experiment(self, _):
        pass

    def start_run(self, run_name):
        return self

    def log_params(self, _):
        pass

    def log_metric(self, _, __, step):
        pass

    def end_run(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def delete_run(self):
        pass


def _check_args(problem: Problem, algorithm: Algorithm):
    if not isinstance(algorithm, Algorithm):
        warnings.warn("Algorithm type seems wrong")
    if not isinstance(problem, Problem):
        warnings.warn("Problem type seems wrong")


def _unwrap_or_default_metrics(metrics, problem) -> List[Callable[[opti.Problem], Any]]:
    if metrics is None:
        if len(problem.objectives) > 1:
            # Pareto
            nadir = problem.objectives.bounds(problem.outputs).loc["max"].values.copy()
            if np.sum(np.isinf(nadir)) > 0:
                nadir = None

            def hypervolume(prb):
                data = prb.data
                Z = prb.objectives(data)
                return bo.metric.hypervolume(Z.values, ref_point=nadir)

            return [hypervolume]
        else:
            # Single objective
            def objective(prb):
                return prb.objectives(prb.data).min().item()

            return [objective]
    return metrics


def _unwrap_or_default_logger(logger) -> Any:
    if logger is None:
        try:
            return import_module("mlflow")
        except ImportError:
            warnings.warn("Not logging. Could not import mlflow.")
            return NotLogger()
    return logger


def _one_iteration(query_idx, method, n_proposals, metrics, logger):
    X = method.propose(n_proposals)
    Y = method.problem.f(X)
    XY = pd.concat([X, Y], axis=1)
    method.add_data_and_fit(XY)
    new_metrics_values = {m.__name__: m(method.problem) for m in metrics}
    for n, v in new_metrics_values.items():
        logger.log_metric(n, v, step=query_idx)
    return XY, new_metrics_values


def _single_run(args):
    (
        initial_method,
        method_name,
        logger,
        max_n_measurements,
        n_initial,
        n_proposals,
        metrics,
        single_process,
        run_idx,
    ) = args
    method = copy.deepcopy(initial_method)
    problem_name = method.problem.name
    if n_initial > 0:
        method.problem.create_initial_data(n_initial)
    with logger.start_run(run_name=f"1 run {method_name} vs. {problem_name}") as run:
        logger.log_params(
            {
                "problem": problem_name,
                "method": method_name,
                "max_n_measurements": max_n_measurements,
                "n_proposals": n_proposals,
                "n_initial": n_initial,
            }
        )
        metrics_values_of_run = []

        if single_process:
            n_cols, _ = shutil.get_terminal_size((80, 20))
            with tqdm(bar_format="{desc}", position=0, ncols=n_cols) as proposal_line:
                with tqdm(
                    range(0, max_n_measurements - n_initial, n_proposals), position=1
                ) as pbar:
                    for query_idx in pbar:
                        XY, metrics_values = _one_iteration(
                            query_idx, method, n_proposals, metrics, logger
                        )
                        metrics_values_of_run.append(metrics_values)
                        desc = "Proposals:" + ", ".join(
                            f"{v: 1.3e}" for v in XY.values[0]
                        )
                        proposal_line.set_description_str(desc[:n_cols])
        else:
            pbar = tqdm(
                range(0, max_n_measurements - n_initial, n_proposals), position=run_idx
            )
            for query_idx in pbar:
                XY, metrics_values = _one_iteration(
                    query_idx, method, n_proposals, metrics, logger
                )
                metrics_values_of_run.append(metrics_values)
                desc = f"run {run_idx:02d}, " + ", ".join(
                    f"{v: 1.3e}" for v in XY.values[0]
                )
                pbar.set_description(desc[:40])

    return (
        pd.DataFrame(metrics_values_of_run),
        run.info.run_id if hasattr(run, "info") else None,
    )


def _run(args, n_runs, n_cores):
    if n_cores == 1 or n_runs == 1 or mp is None:
        run_ids = []
        metrics_of_runs: List[pd.DataFrame] = []
        for _ in range(n_runs):
            # the args-suffix is True for single process and -1 for in this case unused run_idx
            metrics_of_run, run_id = _single_run(args + (True, -1))
            metrics_of_runs.append(metrics_of_run)
            run_ids.append(run_id)
        return metrics_of_runs, run_ids
    else:
        n_proc = min(n_runs, n_cores)
        ctx = mp.get_context("spawn")
        p = ctx.Pool(n_proc)
        args = args[:2] + (NotLogger(),) + args[3:] + (False,)
        # add index for process to show progress
        args = [args + (i,) for i in range(n_runs)]
        res = p.map(_single_run, args)
        return tuple(zip(*res))


def run(
    method: Algorithm,
    param_log_str: str = "",
    max_n_measurements: int = 20,
    n_proposals: int = 1,
    n_initial: int = 1,
    metrics: Optional[List[Callable[[opti.Problem], Any]]] = None,
    experiment_name: Optional[str] = "Digital Twin Benchmark",
    logger: Optional[Any] = None,
    n_runs: int = 1,
    delete_intermediate_runs: bool = True,
    n_cores=3,
):
    """Run an algorithm on a problem and log results to mlflow if available

    Args:
        method: optimization algorithm, contains the problem
        param_log_str: free string for, e.g., parameters of the algorithm
        max_n_measurements: maximum number of measurements/"chemical experiment simulations"
                            including n_initial
        n_proposals: same as batch size
        n_initial: number of initial proposals, if > 0 all data from problem will be replaced by newly generated data
        metrics: list of callables that map problems where data is attached to iteratively to metric values. Defaults to None.
        experiment_name: name of the experiment where the results are logged to.
        logger: module name or instance to be used for logging. must fulfill the mlflow interface
        n_runs: number of runs to averaged.
        delete_intermediate_runs: if True, only the mean over all runs will be stored and intermediate runs will be deleted,
        n_cores: number of cores for multiprocessing over runs

    Returns:
        means of computed metric values per step
    """
    _check_args(method.problem, method)
    problem_name = method.problem.name
    method_name = type(method).__name__
    if n_initial == 0 and len(method.problem.get_data()) == 0:
        raise ValueError(
            "No initial data available. Pass either n_initial > 0 or make sure your "
            "problem contains data."
        )

    initial_method = copy.deepcopy(method)

    logger = _unwrap_or_default_logger(logger)
    assert logger is not None
    logger.set_experiment(experiment_name)

    metrics = _unwrap_or_default_metrics(metrics, method.problem)

    args = (
        initial_method,
        method_name,
        logger,
        max_n_measurements,
        n_initial,
        n_proposals,
        metrics,
    )
    metrics_of_runs, run_ids = _run(args, n_runs, n_cores=n_cores)
    metrics_mean = sum(metrics_of_runs) / len(metrics_of_runs)
    with logger.start_run(
        run_name=f"{method_name} vs. {problem_name} mean over {n_runs}"
    ):
        logger.log_params(
            {
                "problem": problem_name,
                "param_log_str": param_log_str,
                "method": method_name,
                "max_n_measurements": max_n_measurements,
                "n_proposals": n_proposals,
                "n_initial": n_initial,
                "n_runs": n_runs,
            }
        )
        for idx, row in metrics_mean.iterrows():
            for name in row.index:
                logger.log_metric(name, row[name], step=idx)
    if delete_intermediate_runs and run_ids[0] is not None:
        for run_id in run_ids:
            logger.delete_run(run_id)
    return metrics_mean
