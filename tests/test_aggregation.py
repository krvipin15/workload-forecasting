"""Test cases for the aggregate_cpu_data function."""

import pandas as pd
import pytest

from workload_forecasting.data.aggregation import aggregate_cpu_data


def test_aggregate_cpu_data_success():
    """Test that the function correctly aggregates CPU data."""
    data = pd.DataFrame(
        {
            "timestamp": [100, 200, 300, 100, 200, 300],
            "vm_id": [1, 1, 1, 2, 2, 2],
            "avg_cpu": [10.0, 20.0, 30.0, 20.0, 40.0, 60.0],
        }
    )

    result = aggregate_cpu_data(data, step_seconds=100)

    # Assertions
    assert len(result) == 3
    assert list(result["timestamp"]) == [100, 200, 300]
    # active_vm_count should be 2 for all
    assert (result["active_vm_count"] == 2).all()
    # aggregate_cpu_utilization: (10+20)/2 = 15, (20+40)/2 = 30, (30+60)/2 = 45
    assert list(result["aggregate_cpu_utilization"]) == [15.0, 30.0, 45.0]
    assert result["active_vm_count"].dtype == "int64"


def test_aggregate_cpu_data_cleans_nan_and_duplicates():
    """Test that the function correctly cleans NaN values and duplicate rows."""
    # Setup: include a NaN and a duplicate row
    data = pd.DataFrame(
        {
            "timestamp": [100, 100, 200, 200],
            "vm_id": [1, 1, 1, 1],
            "avg_cpu": [10.0, 10.0, None, 20.0],
        }
    )

    result = aggregate_cpu_data(data, step_seconds=100)

    assert len(result) == 2
    assert result["active_vm_count"].sum() == 2


def test_aggregate_cpu_data_temporal_gaps():
    """Test that the function raises an error when there are temporal gaps in the data."""
    # Setup: Missing the 200 timestamp
    data = pd.DataFrame({"timestamp": [100, 300], "vm_id": [1, 1], "avg_cpu": [10.0, 20.0]})

    # step_seconds=100 implies 100, 200, 300. 200 is missing.
    with pytest.raises(ValueError, match="Detected 1 missing timestamps"):
        aggregate_cpu_data(data, step_seconds=100)


def test_aggregate_cpu_data_missing_columns():
    """Test that the function raises an error when required columns are missing."""
    # Setup: Missing 'avg_cpu' column
    data = pd.DataFrame({"timestamp": [100, 200], "vm_id": [1, 1]})

    with pytest.raises(KeyError):
        aggregate_cpu_data(data, step_seconds=100)


def test_aggregate_cpu_data_single_point():
    """Test that the function can handle a single data point without errors."""
    # Setup: Only one data point
    data = pd.DataFrame({"timestamp": [100], "vm_id": [1], "avg_cpu": [10.0]})

    result = aggregate_cpu_data(data, step_seconds=100)
    assert len(result) == 1
    assert result.iloc[0]["active_vm_count"] == 1
