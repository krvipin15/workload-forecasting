"""Data Validation Module for Workload Forecasting Project.

This module validates raw challenge files (`vm_cpu_readings.csv.gz` and
`vm_metadata.csv.gz`), performs cross-dataset consistency checks, logs
execution details with structlog, and exports structured JSON validation
reports to the configured report directory.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import pandera.pandas as pa

from src.workload_forecasting.config.logger import get_logger
from src.workload_forecasting.config.settings import get_settings
from src.workload_forecasting.data.schemas import cpu_readings_schema, vm_metadata_schema


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
    }

    try:
        # Check if the required files exist
        if not cpu_readings_file.is_file():
            raise FileNotFoundError(f"Missing file or not a file: {cpu_readings_file}")
        if not metadata_file.is_file():
            raise FileNotFoundError(f"Missing file or not a file: {metadata_file}")

        # Load the datasets
        logger.debug("Loading CPU readings dataset", file_path=str(cpu_readings_file))
        cpu_readings_df = pd.read_csv(cpu_readings_file, compression="gzip")
        logger.info("CPU readings dataset loaded", shape=cpu_readings_df.shape)
        logger.debug("Loading VM metadata dataset", file_path=str(metadata_file))
        vm_metadata_df = pd.read_csv(metadata_file, compression="gzip")
        logger.info("VM metadata dataset loaded", shape=vm_metadata_df.shape)

        # Validate the datasets against their respective schemas
        logger.info("Executing schema validation on CPU readings")
        try:
            validated_cpu = cpu_readings_schema.validate(cpu_readings_df, lazy=True)
            report_data["cpu_readings_valid"] = True
            logger.info("CPU readings schema validation passed", total_rows=len(validated_cpu))
        except pa.errors.SchemaErrors as err:
            failure_dict: dict = err.failure_cases.to_dict()
            logger.error("CPU readings schema validation failed", failure_cases=failure_dict)
            report_data["errors"].append({"dataset": "vm_cpu_readings", "details": failure_dict})
            raise

        logger.info("Executing schema validation on VM metadata")
        try:
            validated_metadata = vm_metadata_schema.validate(vm_metadata_df, lazy=True)
            report_data["vm_metadata_valid"] = True
            logger.info("VM metadata schema validation passed", total_rows=len(validated_metadata))
        except pa.errors.SchemaErrors as err:
            failure_dict: dict = err.failure_cases.to_dict()
            logger.error("VM metadata schema validation failed", failure_cases=failure_dict)
            report_data["errors"].append({"dataset": "vm_metadata", "details": failure_dict})
            raise

        logger.info("Checking cross-dataset entity consistency")
        unmatched_vms = set(validated_cpu["vm_id"].unique()) - set(
            validated_metadata["vm_id"].unique()
        )

        if unmatched_vms:
            warning_msg = f"{len(unmatched_vms)} unique VMs found in CPU readings lack corresponding metadata entries."
            logger.warning(warning_msg)
            report_data["cross_validation_warning"] = warning_msg

        report_data["status"] = "success"
        logger.info("Dataset validation pipeline completed successfully with zero schema errors")

        return validated_cpu, validated_metadata

    except FileNotFoundError as fnf_err:
        logger.exception("File resolution error encountered", error=str(fnf_err))
        report_data["errors"].append({"error_type": "FileNotFoundError", "message": str(fnf_err)})
        raise
    except Exception as exc:
        logger.exception("Unexpected error encountered during data validation", error=str(exc))
        report_data["errors"].append({"error_type": type(exc).__name__, "message": str(exc)})
        raise

    finally:
        report_file = (
            get_settings().VALIDATION_REPORT_DIR
            / f"validation_{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
        )
        logger.debug("Persisting validation report artifact", report_path=str(report_file))

        with report_file.open("wb") as f:
            f.write(orjson.dumps(report_data, option=orjson.OPT_INDENT_2))

        logger.info("Validation report saved successfully", report_path=str(report_file))
