"""Command-Line Interface for the Forecasting Pipeline."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from workload_forecasting.config.logger import get_logger
from workload_forecasting.config.settings import get_settings
from workload_forecasting.data.aggregation import aggregate_cpu_data
from workload_forecasting.data.loader import load_datasets
from workload_forecasting.data.validation import (
    ValidationReport,
    validate_aggregated_targets,
    validate_datasets,
)

# Initialize the logger
logger = get_logger(__name__)


def _cmd_predict(args: argparse.Namespace) -> None:
    """Run the complete pipeline and generate predictions.csv."""
    settings = get_settings()
    validation_report = ValidationReport()

    # 1. Load and Validate
    cpu_readings_df, vm_metadata_df = load_datasets(args.data)
    validated_cpu, validated_metadata = validate_datasets(
        cpu_readings_df,
        vm_metadata_df,
        validation_report,
    )

    _ = validated_metadata  # Currently unused, but kept for potential future use

    # 2. Aggregate and Validate
    targets_df = aggregate_cpu_data(validated_cpu)
    targets_df = validate_aggregated_targets(targets_df, validation_report)

    # 3. Split into Train and Holdout
    holdout_size = settings.PARAMS.preprocessing.holdout_window_size
    train_df: pd.DataFrame = targets_df.iloc[:-holdout_size].copy()
    holdout_df: pd.DataFrame = targets_df.iloc[-holdout_size:].copy()

    logger.info(
        "Train and holdout datasets created successfully.",
        train_size=len(train_df),
        holdout_size=len(holdout_df),
    )


def main() -> None:
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="workload_forecasting",
        description="Cloud Workload Forecasting Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_predict = subparsers.add_parser(
        "predict", help="Run full pipeline and generate predictions.csv."
    )

    # Change required=True to default=Path(...)
    p_predict.add_argument(
        "--data",
        type=Path,
        default=Path("challenge_data"),
        help="Path to the challenge_data directory (default: challenge_data).",
    )
    p_predict.add_argument(
        "--output",
        type=Path,
        default=Path("predictions/predictions.csv"),
        help="Path to save predictions.csv (default: predictions/predictions.csv).",
    )
    p_predict.set_defaults(func=_cmd_predict)

    args = parser.parse_args()

    try:
        args.func(args)
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
