"""Test cases for the forecasting module."""

import numpy as np
import pandas as pd
import pytest

from workload_forecasting.config.settings import ForecastingParams
from workload_forecasting.modeling.forecasting import (
    ForecastResult,
    _models,
    _with_ensemble,
    cross_validate,
    forecast_targets,
    score_cv,
    to_long,
)


@pytest.fixture
def default_params() -> ForecastingParams:
    """Return default forecasting parameters for testing."""
    return ForecastingParams(
        window_sizes=[2],
        season_lengths=[4],
        cv_windows=2,
        cv_step=1,
        min_train=10,
        peak_quantile=0.8,
        interval_level=95,
    )


@pytest.fixture
def sample_train_data() -> pd.DataFrame:
    """Return a sample training DataFrame for testing."""
    n = 30
    timestamps = np.arange(n, dtype=np.int64)
    active_vm = np.linspace(10, 40, n) + np.sin(np.arange(n)) * 2
    cpu_util = np.linspace(50, 90, n) + np.cos(np.arange(n)) * 5
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "active_vm_count": active_vm,
            "aggregate_cpu_utilization": cpu_util,
        }
    )


def test_models_initialization(default_params):
    """Test that the _models function returns a non-empty list of models with expected aliases."""
    models = _models(default_params)
    assert len(models) > 0
    aliases = [m.alias for m in models]
    assert "Naive" in aliases
    assert "HistoricAverage" in aliases
    assert "WindowAverage_2" in aliases
    assert "SeasonalNaive_4" in aliases
    assert "AutoETS" in aliases
    assert "AutoARIMA" in aliases


def test_with_ensemble():
    """Test that the _with_ensemble function correctly computes ensemble forecasts."""
    # Present
    df = pd.DataFrame({"Naive": [10.0, 20.0], "HistoricAverage": [20.0, 40.0], "Other": [1, 2]})
    res = _with_ensemble(df)
    assert "Ensemble_Naive_HistoricAverage" in res.columns
    np.testing.assert_allclose(res["Ensemble_Naive_HistoricAverage"], [15.0, 30.0])

    # Missing
    df_missing = pd.DataFrame({"Naive": [10.0]})
    res_missing = _with_ensemble(df_missing)
    assert "Ensemble_Naive_HistoricAverage" not in res_missing.columns


def test_to_long_valid(sample_train_data):
    """Test that the to_long function correctly transforms a valid wide DataFrame into long format."""
    long_df = to_long(sample_train_data)
    assert not long_df.empty
    assert set(long_df.columns) == {"unique_id", "ds", "y"}
    assert set(long_df["unique_id"].unique()) == {"active_vm_count", "aggregate_cpu_utilization"}
    assert len(long_df) == len(sample_train_data) * 2
    assert pd.api.types.is_integer_dtype(long_df["ds"])


def test_to_long_empty():
    """Test that the to_long function returns an empty DataFrame with the correct columns when given an empty input."""
    empty_df = pd.DataFrame(columns=["timestamp", "active_vm_count", "aggregate_cpu_utilization"])
    res = to_long(empty_df)
    assert res.empty
    assert list(res.columns) == ["unique_id", "ds", "y"]


def test_to_long_missing_columns(sample_train_data):
    """Test that the to_long function raises a ValueError when required columns are missing from the input DataFrame."""
    bad_df = sample_train_data.drop(columns=["active_vm_count"])
    with pytest.raises(ValueError, match="Training data is missing required columns"):
        to_long(bad_df)


def test_to_long_nan_timestamps(sample_train_data):
    """Test that the to_long function raises a ValueError when the input DataFrame contains NaN timestamps."""
    sample_train_data = sample_train_data.astype({"timestamp": "float64"})
    sample_train_data.loc[0, "timestamp"] = np.nan
    with pytest.raises(ValueError, match="Training timestamps must not contain missing values"):
        to_long(sample_train_data)


def test_to_long_duplicate_timestamps(sample_train_data):
    """Test that the to_long function raises a ValueError when the input DataFrame contains duplicate timestamps."""
    sample_train_data.loc[1, "timestamp"] = sample_train_data.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="Training timestamps must be unique"):
        to_long(sample_train_data)


def test_to_long_non_constant_interval(sample_train_data):
    """Test that the to_long function raises a ValueError when the input DataFrame has non-constant timestamp intervals."""
    bad_df = sample_train_data.copy()
    bad_df.loc[2, "timestamp"] = 100  # Unique but non-constant interval
    with pytest.raises(
        ValueError, match="Training timestamps must have a constant positive interval"
    ):
        to_long(bad_df)


def test_cross_validate_edge_cases(default_params, sample_train_data):
    """Test edge cases for the cross_validate function, including empty data, invalid horizon, and insufficient history."""
    long_df = to_long(sample_train_data)

    # Empty data
    assert cross_validate(pd.DataFrame(), h=2, cfg=default_params) is None

    # Invalid horizon
    with pytest.raises(ValueError, match="Forecast horizon must be at least 1"):
        cross_validate(long_df, h=0, cfg=default_params)

    # Insufficient history (huge min_train)
    strict_params = ForecastingParams(
        window_sizes=[2],
        season_lengths=[4],
        cv_windows=2,
        cv_step=1,
        min_train=1000,
        peak_quantile=0.8,
        interval_level=95,
    )
    assert cross_validate(long_df, h=2, cfg=strict_params) is None


def test_cross_validate_success(default_params, sample_train_data):
    """Test that the cross_validate function returns a valid DataFrame with expected columns when given valid input data."""
    long_df = to_long(sample_train_data)
    cv_res = cross_validate(long_df, h=2, cfg=default_params)
    assert cv_res is not None
    assert "unique_id" in cv_res.columns
    assert "ds" in cv_res.columns
    assert "y" in cv_res.columns
    assert "Naive" in cv_res.columns


def test_score_cv_empty():
    """Test that the score_cv function returns an empty DataFrame when given empty input data."""
    assert score_cv(pd.DataFrame(), pd.DataFrame(), ForecastingParams()).empty


def test_score_cv_success(default_params, sample_train_data):
    """Test that the score_cv function returns a valid DataFrame with expected columns when given valid input data."""
    long_df = to_long(sample_train_data)
    cv_res = cross_validate(long_df, h=2, cfg=default_params)
    scores = score_cv(cv_res, long_df, default_params)  # ty: ignore[invalid-argument-type]
    assert not scores.empty
    assert set(scores.columns) >= {
        "target",
        "model",
        "mae",
        "rmse",
        "mase",
        "peak_mae",
        "skill_vs_naive",
    }


def test_forecast_targets_validation(sample_train_data, default_params):
    """Test that the forecast_targets function raises appropriate errors for invalid input data."""
    with pytest.raises(ValueError, match="Training data must not be empty"):
        forecast_targets(pd.DataFrame(), h=2, cfg=default_params)

    with pytest.raises(ValueError, match="Forecast horizon must be at least 1"):
        forecast_targets(sample_train_data, h=0, cfg=default_params)

    # Non-increasing timestamps when len > 1 (e.g. negative difference)
    bad_train = sample_train_data.copy()
    bad_train.loc[1, "timestamp"] = -5
    with pytest.raises(ValueError, match="Training timestamps must increase"):
        forecast_targets(bad_train, h=2, cfg=default_params)

    # Non-constant interval when len > 1
    bad_interval = sample_train_data.copy()
    bad_interval.loc[2, "timestamp"] = 5
    with pytest.raises(ValueError, match="Training timestamps must have a constant interval"):
        forecast_targets(bad_interval, h=2, cfg=default_params)


def test_forecast_targets_success(sample_train_data, default_params):
    """Test that the forecast_targets function returns a valid ForecastResult with expected attributes when given valid input data."""
    result = forecast_targets(sample_train_data, h=3, cfg=default_params)
    assert isinstance(result, ForecastResult)
    assert isinstance(result.predictions, pd.DataFrame)
    assert isinstance(result.lower, pd.DataFrame)
    assert isinstance(result.upper, pd.DataFrame)
    assert isinstance(result.selected, dict)
    assert isinstance(result.cv_scores, pd.DataFrame)
    assert len(result.predictions) == 3
    assert "active_vm_count" in result.predictions.columns
    assert "aggregate_cpu_utilization" in result.predictions.columns


def test_forecast_targets_insufficient_history(default_params):
    """Test that the forecast_targets function returns a ForecastResult with Naive model selected when training data is insufficient."""
    # Single row training data (triggers step=1 fallback and cv=None)
    single_row = pd.DataFrame(
        {
            "timestamp": [0],
            "active_vm_count": [10.0],
            "aggregate_cpu_utilization": [50.0],
        }
    )
    result = forecast_targets(single_row, h=2, cfg=default_params)
    assert isinstance(result, ForecastResult)
    assert len(result.predictions) == 2
    assert result.cv_scores.empty
    assert result.selected["active_vm_count"] == "Naive"
