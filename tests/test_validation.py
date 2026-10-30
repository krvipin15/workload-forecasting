"""Test cases for the data validation module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pandera.pandas as pa
import pytest

from workload_forecasting.data.validation import (
    ValidationReport,
    validate_datasets,
    validate_targets,
)

# Mock Schemas for Testing
mock_cpu_schema = MagicMock()
mock_meta_schema = MagicMock()
mock_targets_schema = MagicMock()


@pytest.fixture(autouse=True)
def mock_schemas():
    """Set mock schemas to the validation module before each test and restore after."""
    with (
        patch("workload_forecasting.data.validation.cpu_readings_schema", mock_cpu_schema),
        patch("workload_forecasting.data.validation.vm_metadata_schema", mock_meta_schema),
        patch("workload_forecasting.data.validation.aggregate_targets_schema", mock_targets_schema),
    ):
        yield


@pytest.fixture
def report() -> ValidationReport:
    """Create a fresh ValidationReport for each test."""
    return ValidationReport()


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provide sample CPU readings and VM metadata DataFrames."""
    cpu_df = pd.DataFrame(
        {
            "vm_id": [1, 1, 2],
            "timestamp": [100, 200, 100],
            "value": [10.0, 12.0, 15.0],
        }
    )
    meta_df = pd.DataFrame({"vm_id": [1, 2], "os": ["Linux", "Windows"]})
    return cpu_df, meta_df


def test_report_add_error_general(report):
    """Test adding a general error to the ValidationReport."""
    exc = ValueError("General error")
    report.add_error("test_dataset", exc)
    assert len(report.errors) == 1
    assert report.errors[0]["error_type"] == "ValueError"
    assert report.errors[0]["message"] == "General error"


def test_report_add_error_schema(report):
    """Test adding a SchemaErrors error to the ValidationReport."""
    # Mock a pandera SchemaErrors object
    mock_exc = MagicMock(spec=pa.errors.SchemaErrors)
    mock_exc.failure_cases = pd.DataFrame({"column": ["cpu"], "check": ["min"]})

    report.add_error("cpu_readings", mock_exc)
    assert len(report.errors) == 1
    assert "n_failures" in report.errors[0]
    assert isinstance(report.errors[0]["examples"], list)


def test_report_save(report, tmp_path):
    """Test saving the ValidationReport to a JSON file."""
    with patch("workload_forecasting.data.validation.get_settings") as mock_settings:
        mock_settings.return_value.VALIDATION_REPORT_DIR = tmp_path
        report.status = "completed"
        report.save()

        report_file = tmp_path / "validation_report.json"
        assert report_file.exists()
        content = report_file.read_text()
        assert '"status": "completed"' in content


def test_validate_datasets_success(sample_data, report):
    """Test successful validation of CPU readings and VM metadata datasets."""
    cpu_df, meta_df = sample_data
    # Setup mocks to return the input dfs
    mock_cpu_schema.validate.return_value = cpu_df
    mock_meta_schema.validate.return_value = meta_df

    # step_seconds=100 matches the sample_data timestamps
    validate_datasets(cpu_df, meta_df, 100, report)

    assert report.cpu_readings_valid is True
    assert report.vm_metadata_valid is True
    assert report.metrics["n_vms"] == 2
    assert report.metrics["irregular_steps"] == 0


def test_validate_datasets_consistency_warnings(sample_data, report):
    """Test validation when CPU readings and VM metadata datasets are inconsistent."""
    cpu_df, meta_df = sample_data
    # Add a VM to CPU that isn't in Metadata
    cpu_df_ext = pd.concat(
        [
            cpu_df,
            pd.DataFrame({"vm_id": [3], "timestamp": [100], "value": [1.0]}),
        ]
    )

    mock_cpu_schema.validate.return_value = cpu_df_ext
    mock_meta_schema.validate.return_value = meta_df

    # Use an incorrect step_seconds to trigger irregular steps warning
    validate_datasets(cpu_df_ext, meta_df, 999, report)

    assert any("VM populations differ" in w for w in report.warnings)
    assert any("steps are not 999s" in w for w in report.warnings)
    assert report.metrics["vms_without_metadata"] == 1


def test_validate_datasets_no_observations(sample_data, report):
    """Test validation when CPU readings dataset has no observations."""
    _, meta_df = sample_data
    # Create an empty DF that passes schema but fails consistency
    empty_cpu = pd.DataFrame(columns=["vm_id", "timestamp"])

    mock_cpu_schema.validate.return_value = empty_cpu
    mock_meta_schema.validate.return_value = meta_df

    with pytest.raises(ValueError, match="CPU readings dataset contains no VM observations"):
        validate_datasets(empty_cpu, meta_df, 100, report)


def test_validate_datasets_schema_failure(sample_data, report):
    """Test validation when a SchemaErrors exception occurs during dataset validation."""
    cpu_df, meta_df = sample_data

    # Define a dummy exception that inherits from SchemaErrors
    class DummySchemaError(pa.errors.SchemaErrors):
        def __init__(self):
            pass

        def __str__(self):
            return "Schema errors found in cpu_readings"

    # Instantiate and add the required attribute for ValidationReport.add_error
    error_instance = DummySchemaError()
    error_instance.failure_cases = pd.DataFrame({"column": ["cpu"], "check": ["min"]})

    # Set as the side effect
    mock_cpu_schema.validate.side_effect = error_instance

    with pytest.raises(pa.errors.SchemaErrors):
        validate_datasets(cpu_df, meta_df, 100, report)

    assert report.status == "failed"
    assert len(report.errors) == 1
    assert report.errors[0]["dataset"] == "vm_cpu_readings"


def test_validate_datasets_generic_failure(sample_data, report):
    """Test validation when a generic exception occurs during dataset validation."""
    cpu_df, meta_df = sample_data
    mock_cpu_schema.validate.side_effect = RuntimeError("Crash")

    with pytest.raises(RuntimeError):
        validate_datasets(cpu_df, meta_df, 100, report)

    assert report.errors[0]["dataset"] == "general_dataset_validation"


def test_validate_targets_success(report):
    """Test successful validation of aggregated targets dataset."""
    df = pd.DataFrame({"target": [1, 2]})
    mock_targets_schema.validate.return_value = df

    validate_targets(df, report)
    assert report.aggregated_targets_valid is True


def test_validate_targets_failure(report):
    """Test validation when an exception occurs during aggregated targets validation."""
    df = pd.DataFrame({"target": [1, 2]})
    mock_targets_schema.validate.side_effect = Exception("Target Error")

    with pytest.raises(Exception, match="Target Error"):
        validate_targets(df, report)

    assert report.status == "failed"
    assert report.errors[0]["dataset"] == "aggregated_targets"
