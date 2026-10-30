"""Test cases for the data loader module."""

import gzip
from pathlib import Path

import pandas as pd
import pytest

from workload_forecasting.data.loader import load_datasets


def _create_mock_csv_gz(path: Path, data: pd.DataFrame) -> None:
    """Write the DataFrame to a CSV string and then compress it using gzip."""
    with gzip.open(path, "wb") as f:
        csv_data = data.to_csv(index=False).encode("utf-8")
        f.write(csv_data)


def test_load_datasets_success(tmp_path):
    """Test loading datasets from valid compressed CSV files."""
    # Setup: Create valid mock files
    cpu_df = pd.DataFrame({"vm_id": [1, 2], "cpu": [10.5, 20.1]})
    meta_df = pd.DataFrame({"vm_id": [1, 2], "os": ["Linux", "Windows"]})

    _create_mock_csv_gz(tmp_path / "test_cpu_readings.csv.gz", cpu_df)
    _create_mock_csv_gz(tmp_path / "test_metadata.csv.gz", meta_df)

    # Execute
    cpu_res, meta_res = load_datasets(tmp_path)

    # Assert
    pd.testing.assert_frame_equal(cpu_res, cpu_df)
    pd.testing.assert_frame_equal(meta_res, meta_df)


def test_load_datasets_missing_cpu_file(tmp_path):
    """Test loading datasets when the CPU readings file is missing."""
    # Setup: Only create metadata file
    meta_df = pd.DataFrame({"vm_id": [1], "os": ["Linux"]})
    _create_mock_csv_gz(tmp_path / "test_metadata.csv.gz", meta_df)

    with pytest.raises(FileNotFoundError, match="Could not find a CPU readings file"):
        load_datasets(tmp_path)


def test_load_datasets_missing_metadata_file(tmp_path):
    """Test loading datasets when the metadata file is missing."""
    # Setup: Only create CPU file
    cpu_df = pd.DataFrame({"vm_id": [1], "cpu": [10.5]})
    _create_mock_csv_gz(tmp_path / "test_cpu_readings.csv.gz", cpu_df)

    with pytest.raises(FileNotFoundError, match="Could not find a metadata file"):
        load_datasets(tmp_path)


def test_load_datasets_empty_file(tmp_path):
    """Test loading datasets when one of the files is empty, which should raise an EmptyDataError."""
    # Setup: CPU file is valid, metadata file is totally empty
    cpu_df = pd.DataFrame({"vm_id": [1], "cpu": [10.5]})
    _create_mock_csv_gz(tmp_path / "test_cpu_readings.csv.gz", cpu_df)

    # Create a completely empty .gz file to trigger pd.errors.EmptyDataError
    with gzip.open(tmp_path / "test_metadata.csv.gz", "wb"):
        pass

    with pytest.raises(pd.errors.EmptyDataError):
        load_datasets(tmp_path)


def test_load_datasets_df_empty_but_parsable(tmp_path, caplog):
    """Test loading datasets when files are empty but have headers, which should return empty DataFrames."""
    # Setup: Files exist and have headers, but no data (returns empty DF)
    cpu_df = pd.DataFrame(columns=["vm_id", "cpu"])
    meta_df = pd.DataFrame(columns=["vm_id", "os"])

    _create_mock_csv_gz(tmp_path / "test_cpu_readings.csv.gz", cpu_df)
    _create_mock_csv_gz(tmp_path / "test_metadata.csv.gz", meta_df)

    cpu_res, meta_res = load_datasets(tmp_path)

    assert cpu_res.empty
    assert meta_res.empty
    # Verify that the warning log was triggered
    assert "CPU readings file is empty" in caplog.text
    assert "VM metadata file is empty" in caplog.text


def test_load_datasets_unexpected_exception(tmp_path, monkeypatch):
    """Test loading datasets when an unexpected exception occurs during file reading."""
    # Setup: Valid files
    cpu_df = pd.DataFrame({"a": [1]})
    meta_df = pd.DataFrame({"a": [1]})
    _create_mock_csv_gz(tmp_path / "test_cpu_readings.csv.gz", cpu_df)
    _create_mock_csv_gz(tmp_path / "test_metadata.csv.gz", meta_df)

    # Mock pd.read_csv to raise a generic exception
    def mock_read_csv(*args, **kwargs):
        raise RuntimeError("Unexpected system failure")

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    with pytest.raises(RuntimeError, match="Unexpected system failure"):
        load_datasets(tmp_path)
