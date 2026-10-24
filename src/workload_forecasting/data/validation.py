"""Data Validation Module for Forecasting Project.

This module provides functions to validate the VM CPU readings
and metadata datasets, as well as the final aggregated time-series
targets. It uses the `pandera` library for schema validation and performs
cross-dataset consistency checks.
"""

from dataclasses import dataclass, field
from typing import Any

import orjson
import pandas as pd
import pandera.pandas as pa

from workload_forecasting.config.logger import get_logger
from workload_forecasting.config.settings import get_settings
from workload_forecasting.data.schemas import (
    aggregate_targets_schema,
    cpu_readings_schema,
    vm_metadata_schema,
)

# Initialize the logger
logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Container to track validation state across different pipeline stages."""

    status: str = "started"
    cpu_readings_valid: bool = False
    vm_metadata_valid: bool = False
    aggregated_targets_valid: bool = False
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add_error(self, dataset: str, exc: Exception):
        """Format and add errors to the report."""
        error_entry = {"dataset": dataset, "error_type": type(exc).__name__}

        if isinstance(exc, pa.errors.SchemaErrors):
            cases = exc.failure_cases
            error_entry.update(
                {
                    "n_failures": len(cases),
                    "examples": cases.head(100).astype(str).to_dict(orient="records"),
                }
            )
        else:
            error_entry["message"] = str(exc)

        self.errors.append(error_entry)

    def save(self) -> None:
        """Persist the report to a JSON file in the validation report directory."""
        settings = get_settings()
        report_file = settings.VALIDATION_REPORT_DIR / "validation_report.json"

        # Convert dataclass to dict for serialization
        data = {
            "status": self.status,
            "cpu_readings_valid": self.cpu_readings_valid,
            "vm_metadata_valid": self.vm_metadata_valid,
            "aggregated_targets_valid": self.aggregated_targets_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            **self.metrics,
        }

        report_file.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2, default=str))
        logger.info("Validation report saved successfully", report_path=str(report_file))


def _consistency_check(
    cpu_readings_df: pd.DataFrame,
    vm_metadata_df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    """Perform cross-dataset consistency checks.

    Parameters
    ----------
    cpu_readings_df : pd.DataFrame
        DataFrame containing CPU readings with 'vm_id' and 'timestamp'.
    vm_metadata_df : pd.DataFrame
        DataFrame containing VM metadata with 'vm_id'.
    report : ValidationReport
        The validation report to update with metrics and warnings.
    """
    settings = get_settings()
    step_seconds = settings.PARAMS.preprocessing.observation_interval_seconds

    # Check for VM IDs present in one dataset but not the other
    cpu_vm_ids = set(cpu_readings_df["vm_id"])
    metadata_vm_ids = set(vm_metadata_df["vm_id"])
    only_cpu = cpu_vm_ids - metadata_vm_ids
    only_meta = metadata_vm_ids - cpu_vm_ids
    report.metrics["vms_without_metadata"] = len(only_cpu)
    report.metrics["metadata_without_readings"] = len(only_meta)

    # Check for empty observations in CPU readings
    counts = cpu_readings_df.groupby("vm_id").size()
    if counts.empty:
        raise ValueError("CPU readings dataset contains no VM observations")

    # Compute and store metrics in the report
    report.metrics["n_vms"] = int(cpu_readings_df["vm_id"].nunique())
    report.metrics["n_cpu_rows"] = len(cpu_readings_df)
    report.metrics["n_timestamps"] = int(cpu_readings_df["timestamp"].nunique())

    # Check for irregular time steps in CPU readings
    ordered = cpu_readings_df.sort_values(["vm_id", "timestamp"])
    steps = ordered.groupby("vm_id")["timestamp"].diff().dropna()
    irregular_count = int((steps != step_seconds).sum().item())
    report.metrics["irregular_steps"] = irregular_count

    if irregular_count > 0:
        report.warnings.append(f"{irregular_count} steps are not {step_seconds}s")

    if only_cpu or only_meta:
        report.warnings.append(
            f"VM populations differ: {len(only_cpu)} CPU VMs lack metadata, "
            f"{len(only_meta)} metadata VMs lack readings."
        )


def validate_datasets(
    cpu_readings_df: pd.DataFrame,
    vm_metadata_df: pd.DataFrame,
    report: ValidationReport,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the CPU readings and VM metadata datasets.

    Parameters
    ----------
    cpu_readings_df : pd.DataFrame
        DataFrame containing CPU readings with 'vm_id' and 'timestamp'.
    vm_metadata_df : pd.DataFrame
        DataFrame containing VM metadata with 'vm_id'.
    report : ValidationReport
        The validation report to update with metrics and warnings.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Validated CPU readings and VM metadata DataFrames.

    Raises
    ------
    Exception
        If validation fails for either dataset, the exception is logged and re-raised.
    """
    try:
        logger.info("Executing schema validation on CPU readings")
        validated_cpu = cpu_readings_schema.validate(cpu_readings_df, lazy=True)
        report.cpu_readings_valid = True

        logger.info("Executing schema validation on VM metadata")
        validated_metadata = vm_metadata_schema.validate(vm_metadata_df, lazy=True)
        report.vm_metadata_valid = True

        logger.info("Checking cross-dataset entity consistency")
        _consistency_check(validated_cpu, validated_metadata, report)

        return validated_cpu, validated_metadata

    except Exception as exc:
        report.status = "failed"
        if isinstance(exc, pa.errors.SchemaErrors):
            # Determine which dataset failed by looking at the error context if possible
            dataset = "vm_cpu_readings" if "cpu_readings" in str(exc).lower() else "vm_metadata"
            report.add_error(dataset, exc)
        else:
            report.add_error("general_dataset_validation", exc)

        logger.exception("Dataset validation failed", error=str(exc))
        raise


def validate_aggregated_targets(targets_df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """Validate the aggregated time-series targets.

    Parameters
    ----------
    targets_df : pd.DataFrame
        DataFrame containing the aggregated time-series targets.
    report : ValidationReport
        The validation report to update with metrics and warnings.

    Returns
    -------
    pd.DataFrame
        Validated aggregated targets DataFrame.
    """
    logger.info("Executing schema validation on aggregated targets")
    try:
        validated_df = aggregate_targets_schema.validate(targets_df, lazy=True)
        report.aggregated_targets_valid = True
        return validated_df
    except Exception as exc:
        report.status = "failed"
        report.add_error("aggregated_targets", exc)
        logger.exception("Aggregated targets validation failed", error=str(exc))
        raise
