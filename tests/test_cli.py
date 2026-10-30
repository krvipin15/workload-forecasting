"""Test cases for the CLI entry point of the workload_forecasting package."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from workload_forecasting.cli import main


def test_main_predict_success():
    """Test that the 'predict' command calls the train function with correct args."""
    # Mock sys.argv to simulate: python -m workload_forecasting predict --data my_data --output my_out.csv
    test_args = ["workload_forecasting", "predict", "--data", "my_data", "--output", "my_out.csv"]

    with (
        patch.object(sys, "argv", test_args),
        patch("workload_forecasting.cli.train") as mock_train,
    ):
        main()

        # Verify train was called with Path objects as specified in argparse
        mock_train.assert_called_once_with(
            data_dir=Path("my_data"), predictions_path=Path("my_out.csv")
        )


def test_main_predict_defaults():
    """Test that the 'predict' command uses default paths when args are omitted."""
    test_args = ["workload_forecasting", "predict"]

    with (
        patch.object(sys, "argv", test_args),
        patch("workload_forecasting.cli.train") as mock_train,
    ):
        main()

        mock_train.assert_called_once_with(
            data_dir=Path("challenge_data"), predictions_path=Path("predictions/predictions.csv")
        )


def test_main_invalid_command():
    """Test that providing an invalid command exits the program."""
    test_args = ["workload_forecasting", "invalid_cmd"]

    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
        main()


def test_main_pipeline_failure():
    """Test that an exception in the pipeline results in sys.exit(1)."""
    test_args = ["workload_forecasting", "predict"]

    with (
        patch.object(sys, "argv", test_args),
        patch("workload_forecasting.cli.train", side_effect=RuntimeError("Training failed")),
        patch("workload_forecasting.cli.logger") as mock_logger,
    ):
        with pytest.raises(SystemExit) as excinfo:
            main()

        # Verify it exited with status 1
        assert excinfo.value.code == 1
        # Verify the error was logged
        mock_logger.exception.assert_called_once_with("Pipeline failed")
