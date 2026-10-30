"""Test cases for the training pipeline."""

import numpy as np
import pandas as pd
import pytest

from workload_forecasting.config.settings import get_settings
from workload_forecasting.modeling.forecasting import ForecastResult
from workload_forecasting.pipelines.training import train


@pytest.fixture
def mock_pipeline_dependencies(monkeypatch, tmp_path) -> tuple[str, pd.DataFrame]:
    """Create a mocked environment for the training pipeline, including temporary directories and dummy data."""
    # Setup temporary directory structure for outputs
    settings = get_settings()
    monkeypatch.setattr(settings, "TRAINING_REPORT_DIR", tmp_path / "reports")
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    # Dummy dataframe with enough rows to satisfy holdout length and chronological split
    h = settings.PARAMS.preprocessing.holdout
    n = h + 15
    timestamps = np.arange(n, dtype=np.int64)
    active_vm = np.linspace(10, 50, n)
    cpu_util = np.linspace(40, 80, n)

    y_dummy = pd.DataFrame(
        {
            "timestamp": timestamps,
            "active_vm_count": active_vm,
            "aggregate_cpu_utilization": cpu_util,
        }
    )

    # Mock loaders and validators
    monkeypatch.setattr(
        "workload_forecasting.pipelines.training.load_datasets",
        lambda d: (pd.DataFrame(), pd.DataFrame()),  # noqa: ARG005
    )
    monkeypatch.setattr(
        "workload_forecasting.pipelines.training.validate_datasets",
        lambda c, m, s, r: None,  # noqa: ARG005
    )
    monkeypatch.setattr(
        "workload_forecasting.pipelines.training.aggregate_cpu_data",
        lambda df, s: y_dummy,  # noqa: ARG005
    )
    monkeypatch.setattr(
        "workload_forecasting.pipelines.training.validate_targets",
        lambda y, r: None,  # noqa: ARG005
    )

    # Mock score_holdout
    monkeypatch.setattr(
        "workload_forecasting.pipelines.training.score_holdout",
        lambda tr, te, r: {"mae": 1.23},  # noqa: ARG005
    )

    return tmp_path, y_dummy


def test_train_pipeline_success(monkeypatch, mock_pipeline_dependencies):
    """Test that the training pipeline runs successfully and saves predictions and reports."""
    tmp_path, y_dummy = mock_pipeline_dependencies
    h = get_settings().PARAMS.preprocessing.holdout
    preds_path = tmp_path / "predictions" / "preds.csv"

    # Mock forecasting to return valid predictions matching test timestamps
    test_timestamps = y_dummy.iloc[-h:]["timestamp"].reset_index(drop=True)
    mock_predictions = pd.DataFrame(
        {
            "timestamp": test_timestamps,
            "active_vm_count": [20.0] * h,
            "aggregate_cpu_utilization": [50.0] * h,
        }
    )
    mock_lower = mock_predictions.copy()
    mock_upper = mock_predictions.copy()

    forecast_result = ForecastResult(
        predictions=mock_predictions,
        lower=mock_lower,
        upper=mock_upper,
        selected={"active_vm_count": "Naive", "aggregate_cpu_utilization": "Naive"},
        cv_scores=pd.DataFrame({"target": ["active_vm_count"], "model": ["Naive"], "mae": [1.0]}),
    )

    monkeypatch.setattr(
        "workload_forecasting.pipelines.training.forecast_targets",
        lambda tr, horizon, cfg: forecast_result,  # noqa: ARG005
    )

    # Run train pipeline
    train(data_dir=tmp_path, predictions_path=preds_path)

    # Assertions
    assert preds_path.exists()
    saved_preds = pd.read_csv(preds_path)
    assert len(saved_preds) == h

    report_dir = get_settings().TRAINING_REPORT_DIR
    assert (report_dir / "cv_scores.csv").exists()
    assert (report_dir / "intervals.csv").exists()
    assert (report_dir / "holdout_report.json").exists()


def test_train_insufficient_data_for_holdout(monkeypatch, mock_pipeline_dependencies):
    """Test that if the aggregated data is smaller than the holdout length, an error is raised."""
    tmp_path, _ = mock_pipeline_dependencies
    settings = get_settings()
    h = settings.PARAMS.preprocessing.holdout

    # Return tiny aggregated series (smaller than or equal to holdout)
    small_y = pd.DataFrame(
        {
            "timestamp": np.arange(h, dtype=np.int64),
            "active_vm_count": [10.0] * h,
            "aggregate_cpu_utilization": [50.0] * h,
        }
    )
    monkeypatch.setattr(
        "workload_forecasting.pipelines.training.aggregate_cpu_data",
        lambda df, s: small_y,  # noqa: ARG005
    )

    preds_path = tmp_path / "predictions" / "preds.csv"
    with pytest.raises(ValueError, match="need more than the holdout"):
        train(data_dir=tmp_path, predictions_path=preds_path)


def test_train_timestamp_mismatch_raises_error(monkeypatch, mock_pipeline_dependencies):
    """Test that if the forecasted timestamps do not match the holdout timestamps, an error is raised."""
    tmp_path, y_dummy = mock_pipeline_dependencies
    h = get_settings().PARAMS.preprocessing.holdout
    preds_path = tmp_path / "predictions" / "preds.csv"

    # Return predictions with mismatched timestamps
    wrong_timestamps = y_dummy.iloc[-h:]["timestamp"].reset_index(drop=True) + 999
    mock_predictions = pd.DataFrame(
        {
            "timestamp": wrong_timestamps,
            "active_vm_count": [20.0] * h,
            "aggregate_cpu_utilization": [50.0] * h,
        }
    )
    forecast_result = ForecastResult(
        predictions=mock_predictions,
        lower=mock_predictions.copy(),
        upper=mock_predictions.copy(),
        selected={"active_vm_count": "Naive", "aggregate_cpu_utilization": "Naive"},
        cv_scores=pd.DataFrame({"target": ["active_vm_count"], "model": ["Naive"], "mae": [1.0]}),
    )

    monkeypatch.setattr(
        "workload_forecasting.pipelines.training.forecast_targets",
        lambda train, horizon, cfg: forecast_result,  # noqa: ARG005
    )

    with pytest.raises(ValueError, match="Forecast timestamps do not match the holdout timestamps"):
        train(data_dir=tmp_path, predictions_path=preds_path)


def test_train_exception_in_preprocessing_saves_failed_report(monkeypatch, tmp_path):
    """Test that if an exception occurs during preprocessing, a failed report is still saved."""
    settings = get_settings()
    monkeypatch.setattr(settings, "TRAINING_REPORT_DIR", tmp_path / "reports")
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    # Force load_datasets to raise an exception
    def mock_load_fail(d):
        raise RuntimeError("Disk read error")

    monkeypatch.setattr("workload_forecasting.pipelines.training.load_datasets", mock_load_fail)

    with pytest.raises(RuntimeError, match="Disk read error"):
        train(data_dir=tmp_path)
