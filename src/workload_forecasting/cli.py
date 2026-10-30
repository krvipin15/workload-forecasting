"""Command-Line Interface for the Forecasting Pipeline.

This module provides a command-line interface (CLI) to run
the workload forecasting pipeline. It allows users to specify
input data and output paths for predictions.
"""

import argparse
import sys
from pathlib import Path

from workload_forecasting.config.logger import get_logger
from workload_forecasting.pipelines.training import train

# Initialize the logger
logger = get_logger(__name__)


def main() -> None:
    """Parse command-line arguments and execute the pipeline."""
    parser = argparse.ArgumentParser(
        prog="workload_forecasting",
        description="Run the workload forecasting pipeline to generate predictions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_predict = subparsers.add_parser(
        "predict",
        help="Run the workload forecasting pipeline to generate predictions.",
    )
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
    p_predict.set_defaults(
        func=lambda args: train(data_dir=args.data, predictions_path=args.output)
    )
    args = parser.parse_args()

    try:
        args.func(args)
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
