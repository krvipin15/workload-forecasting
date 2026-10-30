"""Test cases for the evaluation module."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from workload_forecasting.modeling.evaluation import _provision_stats, score_holdout


@dataclass
class MockResults:
    """Mocks the result object containing predictions and intervals."""

    predictions: dict
    lower: dict
    upper: dict
    selected: dict


def test_provision_stats():
    """Test the _provision_stats function with known inputs and expected outputs."""
    actual = np.array([10.0, 20.0, 30.0])
    capacity = np.array([15.0, 15.0, 40.0])

    stats = _provision_stats(actual, capacity)

    assert stats["under_rate"] == pytest.approx(1 / 3)
    assert stats["mean_shortfall"] == pytest.approx(5 / 3)
    assert stats["mean_waste"] == pytest.approx(5.0)


def test_score_holdout(monkeypatch):
    """Test the score_holdout function with a simple example."""
    # Mock TARGETS to be a simple list
    monkeypatch.setattr("workload_forecasting.modeling.evaluation.TARGETS", ["cpu_util"])

    # Setup training data (used for naive forecast and MASE scale)
    # diffs: [1, 1], mean diff = 1.0
    train = pd.DataFrame({"cpu_util": [10.0, 11.0, 12.0]})

    # Setup test data (actuals)
    test = pd.DataFrame({"cpu_util": [13.0, 15.0]})

    res = MockResults(
        predictions={"cpu_util": pd.Series([12.0, 14.0])},
        lower={"cpu_util": pd.Series([11.0, 13.0])},
        upper={"cpu_util": pd.Series([13.0, 15.0])},
        selected={"cpu_util": "prophet"},
    )

    scores = score_holdout(train, test, res)

    target_score = scores["cpu_util"]
    assert target_score["model"] == "prophet"
    assert target_score["mae"] == pytest.approx(1.0)
    assert target_score["rmse"] == pytest.approx(1.0)
    assert target_score["mase"] == pytest.approx(1.0)
    assert target_score["naive_mae"] == pytest.approx(2.0)
    assert target_score["skill_vs_naive"] == pytest.approx(0.5)
    assert target_score["interval_coverage"] == pytest.approx(1.0)

    # Verify provisioning policy calls
    assert "under_rate" in target_score["policy_point"]
    assert "under_rate" in target_score["policy_upper_band"]


def test_score_holdout_edge_cases(monkeypatch):
    """Test edge cases for score_holdout, including zero scale and perfect predictions."""
    monkeypatch.setattr("workload_forecasting.modeling.evaluation.TARGETS", ["cpu_util"])

    # Edge Case 1: No variation in training data (scale = 0)
    train = pd.DataFrame({"cpu_util": [10.0, 10.0]})
    test = pd.DataFrame({"cpu_util": [11.0]})

    # Edge Case 2: Perfect predictions (MAE = 0)
    res = MockResults(
        predictions={"cpu_util": pd.Series([11.0])},
        lower={"cpu_util": pd.Series([10.0])},
        upper={"cpu_util": pd.Series([12.0])},
        selected={"cpu_util": "perfect_model"},
    )

    scores = score_holdout(train, test, res)
    target_score = scores["cpu_util"]

    # If scale is 0, MASE should be None
    assert target_score["mase"] is None
    # If MAE is 0, skill should be 1.0 (1 - 0/naive_mae)
    assert target_score["skill_vs_naive"] == pytest.approx(1.0)


def test_score_holdout_zero_naive_mae(monkeypatch):
    """Test that skill_vs_naive is None when naive_mae is zero to avoid division by zero."""
    monkeypatch.setattr("workload_forecasting.modeling.evaluation.TARGETS", ["cpu_util"])

    # Train and Test are exactly the same value (Naive MAE = 0)
    train = pd.DataFrame({"cpu_util": [10.0, 10.0]})
    test = pd.DataFrame({"cpu_util": [10.0]})
    res = MockResults(
        predictions={"cpu_util": pd.Series([11.0])},  # Forecast is wrong
        lower={"cpu_util": pd.Series([9.0])},
        upper={"cpu_util": pd.Series([12.0])},
        selected={"cpu_util": "bad_model"},
    )

    scores = score_holdout(train, test, res)
    assert scores["cpu_util"]["skill_vs_naive"] is None
