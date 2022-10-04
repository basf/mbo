from typing import Dict, Optional, Sequence, Union

import hvwfg
import numpy as np
import pandas as pd
from opti.metric import pareto_front
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from mbo.algorithm import Algorithm


def hypervolume(
    A: np.ndarray, ref_point: Optional[Union[np.ndarray, Sequence]] = None
) -> float:
    """Hypervolume indicator.

    Calculates the volume that is dominated by a set of points.

    Args:
        A (2D-array): Set of points.
        ref_point (1D-array, optional): Upper bound to the hypervolume.
            If not specfied the nadir of A is used.

    Returns:
        float: Dominated hypervolume.
    """
    A = pareto_front(A)
    if ref_point is None:
        ref_point = A.max(axis=0)
    else:
        ref_point = np.asarray(ref_point, dtype=np.float64)
    return hvwfg.wfg(A, ref_point)


def hypervolume_improvement(
    A: np.ndarray,
    R: np.ndarray,
    ref_point: Optional[Union[np.ndarray, Sequence]] = None,
) -> np.ndarray:
    """Hypervolume improvement.

    Calculate for each point in A the increase in hypervolume when added to a set R.

    Args:
        A (2D-array): Set of candidate points.
        R (2D-array): Set of reference points.
        ref_point (1D-array, optional): Upper bound to the hypervolume.
            If not specfied the nadir of R is used.

    Returns:
        array: Exclusive hypervolume contribution for each point in A.
    """
    # use nadir of reference front if no ref_point given
    if ref_point is None:
        ref_point = R.max(axis=0)

    hv = hypervolume(R, ref_point)
    hvi = np.zeros(len(A))
    for i, a in enumerate(A):
        if (a > R).all(axis=1).any():
            continue  # if a is dominated by R, its contribution is 0 and we can skip
        hvi[i] = hypervolume(np.row_stack([a, R]), ref_point) - hv
    return hvi


def cross_validate(optimizer: Algorithm, n_splits: int = 50) -> Dict[str, pd.DataFrame]:
    """Perform grouped k-fold cross-validation, refitting the model on each fold.

    Returns: dict
        predictions: dataframe of observed values and predicted mean / std
        parameters: long-format dataframe of model parameters per split
        metrics: dataframe with goodness-of-fit measures
    """
    # Ensure the model is tuned so it does not happen inside the cross-validation loop.
    optimizer._tune_model()

    data = optimizer.problem.data

    if data is None:
        raise ValueError("No data for cross-validation")
    if len(data) < 5:
        raise ValueError("Not enough data for cross-validation")

    # Group by input values to ensure correct cross-validation in the presence of repeated experiments.
    groups = np.zeros(len(data), dtype=int)
    G = data.groupby(optimizer.problem.inputs.names)
    for i, g in enumerate(G.indices.values()):
        groups[g] = i

    n_splits = min(len(G), n_splits)

    predictions = []
    parameters = []

    splits = GroupKFold(n_splits=n_splits).split(data, groups=groups)
    for i, (train, test) in enumerate(splits):
        train_data = data.iloc[train]
        test_data = data.iloc[test]
        opt = optimizer.copy(train_data)

        df = opt.get_model_parameters()
        df["cv_split"] = i
        parameters.append(df)

        df = pd.concat([test_data, opt.predict(test_data)], axis=1)
        df["cv_split"] = i
        predictions.append(df)

    predictions = pd.concat(predictions).loc[data.index]
    parameters = pd.concat(parameters)
    metrics = goodness_of_fit(predictions)
    return {
        "predictions": predictions,
        "parameters": parameters,
        "metrics": metrics,
    }


def goodness_of_fit(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate the goodness of fit for given model predictions."""
    result = pd.DataFrame()
    output_names = [n[5:] for n in predictions.columns if n.startswith("mean_")]
    for name in output_names:
        s = predictions[name].notna()
        Y = predictions.loc[s, name].values
        Yp = predictions.loc[s, f"mean_{name}"].values
        result.loc[name, "RMSE"] = mean_squared_error(Y, Yp) ** 0.5
        result.loc[name, "MAE"] = mean_absolute_error(Y, Yp)
        result.loc[name, "R2"] = r2_score(Y, Yp)
    return result


def adjusted_r2_score(y: np.array, yp: np.array, n_features: int, **kwargs) -> float:
    """Adjusted R2 score.

    Args:
        y: True outputs.
        yp: Predicted outputs.
        n_features: Number of considered features.
        kwargs: Additional arguments to pass to r2_score.
    """
    r2 = r2_score(y, yp, **kwargs)
    n_samples = len(y)
    if n_samples <= n_features + 1:
        return -np.inf
    coef = (n_samples - 1) / (n_samples - n_features - 1)
    return 1 - (1 - r2) * coef
