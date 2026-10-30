"""Evaluation metrics for workload forecasting models.

This module provides functions to compute various evaluation metrics for
workload forecasting models, including mean absolute error (MAE), root mean
square error (RMSE), mean absolute scaled error (MASE), and under-provisioning
statistics. It also includes functionality to score holdout datasets against
model predictions.
"""

import numpy as np
import pandas as pd

from workload_forecasting.config.logger import get_logger
from workload_forecasting.modeling.forecasting import TARGETS

# Initialize the logger
logger = get_logger(__name__)


def _provision_stats(actual: np.ndarray, capacity: np.ndarray) -> dict[str, float]:
    """Compute the under-provisioning rate, mean shortfall, and mean waste.

    This module is used to evaluate the performance of a capacity provisioning
    policy by comparing the actual observed values with the provisioned capacity
    values. It calculates three key metrics:
    1. Under-provisioning rate: The proportion of time the actual demand exceeds the provisioned capacity.
    2. Mean shortfall: The average amount by which the actual demand exceeds the provisioned capacity.
    3. Mean waste: The average amount by which the provisioned capacity exceeds the actual demand.

    Parameters
    ----------
    actual : np.ndarray
        The actual observed values.
    capacity : np.ndarray
        The provisioned capacity values.

    Returns
    -------
    dict[str, float]
        A dictionary containing the under-provisioning rate, mean shortfall, and mean waste.
    """
    return {
        "under_rate": float((actual > capacity).mean()),
        "mean_shortfall": float(np.maximum(actual - capacity, 0).mean()),
        "mean_waste": float(np.maximum(capacity - actual, 0).mean()),
    }


def score_holdout(train: pd.DataFrame, test: pd.DataFrame, res) -> dict[str, dict]:
    """Compute holdout scores for each target.

    Parameters
    ----------
    train : pd.DataFrame
        The training dataset containing historical data.
    test : pd.DataFrame
        The test dataset containing holdout data for evaluation.
    res : object
        An object containing the model predictions, selected models, and prediction intervals.

    Returns
    -------
    dict[str, dict]
        A dictionary where each key is a target variable and the value is another dictionary
        containing various evaluation metrics for that target.
    """
    logger.info("Computing holdout scores for targets: %s", TARGETS)
    out = {}
    for t in TARGETS:
        logger.debug("Evaluating target: %s", t)
        y, p = test[t].to_numpy(), res.predictions[t].to_numpy()
        naive = np.full_like(y, train[t].iloc[-1], dtype=float)
        lo, hi = res.lower[t].to_numpy(), res.upper[t].to_numpy()

        mae, mae_naive = np.abs(y - p).mean(), np.abs(y - naive).mean()
        scale = np.abs(np.diff(train[t])).mean()

        mase = float(mae / scale) if scale > 0 else None
        skill = float(1 - mae / mae_naive) if mae_naive > 0 else None

        out[t] = {
            "model": res.selected[t],
            "mae": float(mae),
            "rmse": float(np.sqrt(((y - p) ** 2).mean())),
            "mase": mase,
            "naive_mae": float(mae_naive),
            "skill_vs_naive": skill,
            "interval_coverage": float(((y >= lo) & (y <= hi)).mean()),
            # headroom policy: provision at the upper band instead of the point forecast
            "policy_point": _provision_stats(y, p),
            "policy_upper_band": _provision_stats(y, hi),
        }

        logger.debug(
            "Target %s results - Model: %s, MAE: %.4f, MASE: %s, Skill: %s",
            t,
            out[t]["model"],
            out[t]["mae"],
            f"{mase:.4f}" if mase is not None else "N/A",
            f"{skill:.4f}" if skill is not None else "N/A",
        )

    logger.info("Holdout scoring completed for all targets.")
    return out
