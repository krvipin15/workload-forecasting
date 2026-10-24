"""Aggregation Module for Forecasting Project.

This module provides functionality to aggregate raw CPU
readings from virtual machines (VMs) into a structured
time-series format suitable for forecasting tasks.
"""

from typing import cast

import pandas as pd

from workload_forecasting.config.logger import get_logger
from workload_forecasting.config.settings import get_settings

# Initialize the logger
logger = get_logger(__name__)


def aggregate_cpu_data(cpu_readings_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw VM CPU readings into global time-series targets.

    This function calculates two primary metrics per timestamp:
    1. active_vm_count: The number of distinct VMs with observations.
    2. aggregate_cpu_utilization: The mean average CPU utilization across all VMs.

    The resulting series is reindexed to a strict 5-minute grid to detect
    missing observations or gaps in the data.

    Parameters
    ----------
    cpu_readings_df : pd.DataFrame
        The raw CPU readings dataframe containing at least 'timestamp',
        'vm_id', and 'avg_cpu' columns.

    Returns
    -------
    pd.DataFrame
        A dataframe indexed by timestamp containing the aggregate targets
        'active_vm_count' and 'aggregate_cpu_utilization'.

    Raises
    ------
    ValueError
        If gaps are detected in the timestamp grid, as the project
        specification forbids inventing values for missing observations.
    KeyError
        If required columns ('timestamp', 'vm_id', 'avg_cpu') are missing
        from the input dataframe.
    """
    settings = get_settings()

    logger.info(
        "Starting aggregation of CPU data to derive targets", total_rows=len(cpu_readings_df)
    )

    try:
        # Drop rows with NaN values in the "avg_cpu" column to ensure clean means
        valid = cpu_readings_df.dropna(subset=["avg_cpu"])
        n_dropped = len(cpu_readings_df) - len(valid)
        if n_dropped > 0:
            logger.info("Dropped rows with NaN avg_cpu", n=n_dropped)
        else:
            logger.debug("No NaN values found in avg_cpu")

        # Aggregate the data by timestamp
        logger.debug("Performing groupby aggregation on timestamp")
        y = valid.groupby("timestamp").agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        logger.info("Aggregation complete", aggregate_rows=len(y))

        # Create a complete timestamp grid with a step of 5 minutes (300s)
        start = cast(int, y.index.min())
        stop = cast(int, y.index.max()) + 1
        step_seconds = settings.PARAMS.preprocessing.observation_interval_seconds

        logger.debug("Reindexing to 5-minute grid", start=start, stop=stop)
        grid = pd.RangeIndex(start, stop, step_seconds, name="timestamp")
        y = y.reindex(grid)

        # Check for missing timestamps in the grid
        n_gaps = int(y.isna().any(axis=1).sum())

        if n_gaps > 0:
            logger.error("Temporal gaps detected in the aggregated series", gaps=n_gaps)
            raise ValueError(
                f"Detected {n_gaps} missing timestamps in the grid. "
                "The dataset contract requires explicit handling; automatic filling is prohibited."
            )

        logger.info("Timestamp grid is continuous; no gaps detected")

        # Reset index to turn 'timestamp' back into a column and ensure correct types
        result = y.reset_index().astype({"active_vm_count": "int64"})
        logger.info("Targets built successfully", final_shape=result.shape)

        return result

    except KeyError as e:
        logger.exception("Required column missing from input dataframe", missing_col=str(e))
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred during target building", error=str(e))
        raise
