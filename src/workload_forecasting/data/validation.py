"""Data Validation Module for Workload Forecasting Project.

This module validates raw challenge files (`vm_cpu_readings.csv.gz` and
`vm_metadata.csv.gz`), performs cross-dataset consistency checks, logs
execution details with structlog, and exports structured JSON validation
reports to the configured report directory.
"""

from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import pandera.pandas as pa

from workload_forecasting.config.logger import get_logger
from workload_forecasting.config.settings import get_settings
from workload_forecasting.data.schemas import cpu_readings_schema, vm_metadata_schema

# Constants
STEP_SECONDS = 300
MAX_FAILURES_IN_REPORT = 50


def _failure_details(exc: pa.errors.SchemaErrors) -> dict[str, Any]:
    """Extract failure details from a pandera SchemaErrors exception.

    Parameters
    ----------
    exc : pa.errors.SchemaErrors
        The SchemaErrors exception raised during schema validation.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the number of failures and a list of example
        failure cases (up to MAX_FAILURES_IN_REPORT).
    """
    cases = exc.failure_cases
    return {
        "n_failures": len(cases),
        "examples": cases.head(MAX_FAILURES_IN_REPORT).astype(str).to_dict(orient="records"),
    }


def _consistency_check(
    cpu_readings_df: pd.DataFrame,
    vm_metadata_df: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    """Perform cross-dataset consistency checks between CPU readings and VM metadata.

    Parameters
    ----------
    cpu_readings_df : pd.DataFrame
        DataFrame containing CPU readings with columns `vm_id`, `timestamp`, and `cpu_usage`.
    vm_metadata_df : pd.DataFrame
        DataFrame containing VM metadata with columns `vm_id`, `vm_type`, and `region`.
    report : dict[str, Any]
        Dictionary to store validation results and warnings.
    """
    # Identify unique VM IDs in both datasets
    cpu_vm_ids = set(cpu_readings_df["vm_id"])
    metadata_vm_ids = set(vm_metadata_df["vm_id"])

    # Determine VMs that are only in CPU readings or only in metadata
    only_cpu = cpu_vm_ids - metadata_vm_ids
    only_meta = metadata_vm_ids - cpu_vm_ids

    # Add to report and raise an error if there are inconsistencies
    report["vms_without_metadata"] = len(only_cpu)
    report["metadata_without_readings"] = len(only_meta)

    # Compute and report the number of observations per VM in CPU readings
    counts = cpu_readings_df.groupby("vm_id").size()
    if counts.empty:
        raise ValueError("CPU readings dataset contains no VM observations")

    report["n_vms"] = int(cpu_readings_df["vm_id"].nunique())
    report["n_cpu_rows"] = len(cpu_readings_df)
    report["n_timestamps"] = int(cpu_readings_df["timestamp"].nunique())

    # Compute and report the number of irregular time steps in CPU readings
    ordered = cpu_readings_df.sort_values(["vm_id", "timestamp"])
    steps = ordered.groupby("vm_id")["timestamp"].diff().dropna()
    irregular_count = int((steps != STEP_SECONDS).sum().item())
    report["irregular_steps"] = irregular_count

    # Add warnings to the report if there are any irregularities
    warnings = []
    if irregular_count > 0:
        warnings.append(f"{irregular_count} steps are not {STEP_SECONDS}s")

    if only_cpu or only_meta:
        warnings.append(
            f"VM populations differ: {len(only_cpu)} CPU VMs lack metadata, "
            f"{len(only_meta)} metadata VMs lack readings."
        )

    report["warnings"] = warnings


def validate_dataset(data_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate VM CPU readings and metadata datasets from the dataset folder.

    Note: This function assumes that the input CSV files are compressed with
    gzip and have the `.csv.gz` extension.

    Parameters
    ----------
    data_dir : Path or str
        Directory path containing `vm_cpu_readings.csv.gz` and `vm_metadata.csv.gz`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the validated CPU readings DataFrame and VM metadata DataFrame.

    Raises
    ------
    FileNotFoundError
        If required input CSV files are missing in `data_dir`.
    pandera.errors.SchemaErrors
        If the data fails schema validation checks.
    Exception
        For any unexpected failures during loading, validation, or reporting.
    """
    logger = get_logger(__name__)

    # Define the dir and file paths
    cpu_readings_file = Path(data_dir) / "vm_cpu_readings.csv.gz"
    metadata_file = Path(data_dir) / "vm_metadata.csv.gz"

    report_data: dict[str, Any] = {
        "status": "failed",
        "cpu_readings_valid": False,
        "vm_metadata_valid": False,
        "errors": [],
        "warnings": [],
    }

    try:
        # Check if the required files exist
        for required_file in (cpu_readings_file, metadata_file):
            if not required_file.is_file():
                raise FileNotFoundError(f"Missing file or not a file: {required_file}")

        # Load the datasets
        logger.debug("Loading CPU readings dataset", file_path=str(cpu_readings_file))
        cpu_readings_df = pd.read_csv(cpu_readings_file, compression="gzip")
        logger.info("CPU readings dataset loaded", shape=cpu_readings_df.shape)
        logger.debug("Loading VM metadata dataset", file_path=str(metadata_file))
        vm_metadata_df = pd.read_csv(metadata_file, compression="gzip")
        logger.info("VM metadata dataset loaded", shape=vm_metadata_df.shape)

        report_data["input_files"] = {
            "cpu_readings": str(cpu_readings_file),
            "vm_metadata": str(metadata_file),
        }

        # Validate the datasets against their respective schemas
        logger.info("Executing schema validation on CPU readings")
        try:
            validated_cpu = cpu_readings_schema.validate(cpu_readings_df, lazy=True)
            report_data["cpu_readings_valid"] = True
            logger.info("CPU readings schema validation passed", total_rows=len(validated_cpu))
        except pa.errors.SchemaErrors as exc:
            report_data["errors"].append(
                {
                    "dataset": "vm_cpu_readings",
                    "error_type": type(exc).__name__,
                    **_failure_details(exc),
                }
            )
            raise

        logger.info("Executing schema validation on VM metadata")
        try:
            validated_metadata = vm_metadata_schema.validate(vm_metadata_df, lazy=True)
            report_data["vm_metadata_valid"] = True
            logger.info("VM metadata schema validation passed", total_rows=len(validated_metadata))
        except pa.errors.SchemaErrors as exc:
            report_data["errors"].append(
                {
                    "dataset": "vm_metadata",
                    "error_type": type(exc).__name__,
                    **_failure_details(exc),
                }
            )
            raise

        logger.info("Checking cross-dataset entity consistency")
        _consistency_check(
            validated_cpu,
            validated_metadata,
            report_data,
        )

        report_data["status"] = "success"
        logger.info(
            "Dataset validation completed successfully",
            n_vms=report_data["n_vms"],
            n_timestamps=report_data["n_timestamps"],
        )

        return validated_cpu, validated_metadata

    except FileNotFoundError as exc:
        report_data["errors"].append({"error_type": type(exc).__name__, "message": str(exc)})
        logger.exception("Required input file missing", error=str(exc))
        raise
    except Exception as exc:
        if not any(
            error.get("error_type") == type(exc).__name__ for error in report_data["errors"]
        ):
            report_data["errors"].append({"error_type": type(exc).__name__, "message": str(exc)})
        logger.exception("Dataset validation failed", error=str(exc))
        raise

    finally:
        report_file = get_settings().VALIDATION_REPORT_DIR / "validation_report.json"
        report_file.write_bytes(orjson.dumps(report_data, option=orjson.OPT_INDENT_2, default=str))
        logger.info("Validation report saved successfully", report_path=str(report_file))
