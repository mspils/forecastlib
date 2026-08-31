"""Evaluation metrics for time series forecasting models.

Provides functions for calculating various error metrics (MAE, MSE, RMSE, etc.)
and prediction confidence scores.
"""

from functools import wraps
from typing import Any, Callable  # noqa: UP035

import numpy as np
import torch
from torch import Tensor


def numpy_to_torch(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap functions that only take torch tensors to also take np.arrays.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that accepts numpy arrays or tensors.

    """

    @wraps(func)
    def wrapper(x: np.ndarray | Tensor, y: np.ndarray | Tensor, alpha: float | None = None) -> Any:
        """Convert inputs to tensors and call wrapped function.

        Args:
            x: Predictions (numpy array or tensor).
            y: Ground truth values (numpy array or tensor).
            alpha: Optional alpha parameter.

        Returns:
            Result from wrapped function.

        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        if alpha is None:
            return func(x, y)
        return func(x, y, alpha)

    return wrapper


@numpy_to_torch
def conformal_series(y_true: Tensor, y_pred: Tensor, alpha: float = 0.95) -> Tensor:
    """Calculate conformal prediction scores for multiple sets of predictions.

    Parameters
    ----------
    y_true : torch.Tensor
        True target values
    y_pred : torch.Tensor
        Predicted values
    alpha : float, optional (default=0.05)
        1 - Desired confidence level (between 0 and 1)

    Returns
    -------
    conformity_scores : torch.Tensor
        The calculated conformity scores for each of the 48 sets

    """
    confidence = 1 - alpha
    residuals = torch.abs(y_true - y_pred)
    return torch.quantile(residuals, confidence, dim=0)


@numpy_to_torch
def bias_series(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate bias for all prediction steps.

    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values.

    Returns:
        Tensor: metric for all forecast horizons.

    """
    return torch.mean(torch.sub(y_pred, y_true), axis=0)


@numpy_to_torch
def nse_series(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate NSE for all prediction steps.

    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values.

    Returns:
        Tensor: metric for all forecast horizons.

    """
    nom = torch.sum(torch.square(torch.sub(y_true, y_pred)), axis=0)
    denom = torch.sum(torch.square(torch.sub(y_true, torch.mean(y_true, axis=0))), axis=0)

    return 1 - (nom / denom)


@numpy_to_torch
def r_squared_series(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate R^2 for all prediction steps.

    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values.

    Returns:
        Tensor: metric for all forecast horizons.

    """
    residual = torch.sum(torch.square(torch.sub(y_true, y_pred)), axis=0)
    total = torch.sum(torch.square(torch.sub(y_true, torch.mean(y_true, axis=0))), axis=0)
    return torch.subtract(1, torch.divide(residual, total))


@numpy_to_torch
def mse_series(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate the name giving metric for all prediction steps.

    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values.

    Returns:
        Tensor: namegiving values for all prediction steps

    """
    return torch.mean(torch.square(torch.subtract(y_true, y_pred)), axis=0)


@numpy_to_torch
def rmse_series(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate the name giving metric for all prediction steps.

    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values.

    Returns:
        Tensor: namegiving values for all prediction steps

    """
    return torch.sqrt(torch.mean(torch.square(torch.subtract(y_true, y_pred)), axis=0))


@numpy_to_torch
def kge_series(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate the name giving metric for all prediction steps.

    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values.

    Returns:
        Tensor: namegiving values for all prediction steps

    """
    # calculate error in timing and dynamics r
    # (Pearson's correlation coefficient)
    sim_mean = torch.mean(y_pred, axis=0)
    obs_mean = torch.mean(y_true, axis=0)

    r_nom = torch.sum((y_pred - sim_mean) * (y_true - obs_mean), axis=0)
    r_denom = torch.sqrt(torch.sum((y_pred - sim_mean) ** 2, axis=0) * torch.sum((y_true - obs_mean) ** 2, axis=0))
    r = r_nom / r_denom

    # calculate error in spread of flow alpha
    alpha = torch.std(y_pred, axis=0) / torch.std(y_true, axis=0)
    # calculate error in volume beta (bias of mean discharge)
    beta = torch.sum(y_pred, axis=0) / torch.sum(y_true, axis=0)
    # calculate the Kling-Gupta Efficiency KGE
    return 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


@numpy_to_torch
def mae_series(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate the name giving metric for all prediction steps.

    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values.

    Returns:
        Tensor: namegiving values for all prediction steps

    """
    return torch.mean(torch.abs(torch.subtract(y_true, y_pred)), axis=0)


# TODO higher order/currying?
@numpy_to_torch
def p10_series(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate the name giving metric for all prediction steps.

    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values.

    Returns:
        Tensor: namegiving values for all prediction steps

    """
    diff = torch.abs(torch.subtract(y_true, y_pred))
    nom = torch.sum(torch.greater(diff, 10), axis=0)
    return nom / (y_true.shape[0] / 100)


@numpy_to_torch
def p20_series(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate the name giving metric for all prediction steps.

    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values.

    Returns:
        Tensor: namegiving values for all prediction steps

    """
    diff = torch.abs(torch.subtract(y_true, y_pred))
    nom = torch.sum(torch.greater(diff, 20), axis=0)
    return nom / (y_true.shape[0] / 100)


@numpy_to_torch
def wape_series(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate the name giving metric (Weighted Average Percentage Error Metric) for all prediction steps.

    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values.

    Returns:
        Tensor: namegiving values for all prediction steps

    """
    return (y_true - y_pred).abs().sum(axis=0) / y_true.abs().sum(axis=0)


@numpy_to_torch
def mape_series(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Calculate the name giving metric (MAPE) for all prediction steps.

    e.g. 48 values if you predict the next 48 hours for each sample
    Args:
        y_true (Tensor): true values
        y_pred (Tensor): predicted values.

    Returns:
        Tensor: namegiving values for all prediction steps

    """
    return ((y_true - y_pred) / y_true).abs().mean(axis=0)


def get_metric_dict(metric_name_list: list[str]) -> dict:
    """Take a list of metric names and return a dictionary of metric names and functions.

    Args:
        metric_name_list (_type_): _description_

    Returns:
        dict: _description_

    """
    metric_mapping = {
        "bias": bias_series,
        "nse": nse_series,
        "kge": kge_series,
        "mse": mse_series,
        "mae": mae_series,
        "mape": mape_series,
        "p10": p10_series,
        "p20": p20_series,
        "rmse": rmse_series,
        "r2": r_squared_series,
        "wape": wape_series,
        "conf50": lambda x, y: conformal_series(x, y, 0.5),
        "conf10": lambda x, y: conformal_series(x, y, 0.1),
        "conf05": lambda x, y: conformal_series(x, y, 0.05),
    }

    metrics = {}
    for el in metric_name_list:
        if el in metric_mapping:
            metrics[el] = metric_mapping[el]
        else:
            msg = f"{el} is not a valid metric name."
            raise ValueError(msg)
    return metrics
