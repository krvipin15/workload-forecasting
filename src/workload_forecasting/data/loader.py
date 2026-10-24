"""Datasets Loader Module for Forecasting Project.

This module provides functionality to load VM CPU readings
and VM metadata datasets from a specified directory. It handles
the discovery of the appropriate files, reads them into pandas
DataFrames, and performs basic validation checks to ensure the
datasets are loaded correctly.
"""

from pathlib import Path

import pandas as pd

from workload_forecasting.config.logger import get_logger

# Initialize the logger
logger = get_logger(__name__)


def load_datasets(data_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load VM CPU readings and VM metadata datasets from a specified directory.

    This module searches for compressed CSV files containing CPU readings and
    VM metadata within the provided directory. It expects files to be in
    .csv.gz format.

    Parameters
    ----------
    data_dir : Path or str
        The path to the directory containing the dataset files.

    Returns
    -------
    cpu_readings_df : pd.DataFrame
        DataFrame containing the VM CPU utilization readings.
    vm_metadata_df : pd.DataFrame
        DataFrame containing the static VM metadata.

    Raises
    ------
    FileNotFoundError
        If either the CPU readings file or the metadata file is missing from
        the specified directory.
    pd.errors.EmptyDataError
        If one of the found files is empty.
    Exception
        For unexpected errors during the decompression or reading of the CSV files.
    """
    data_dir = Path(data_dir)
    logger.info("Initializing dataset loading", directory=str(data_dir))

    try:
        # Define the paths to the datasets
        cpu_files = list(data_dir.glob("*cpu*.csv.gz"))
        metadata_files = list(data_dir.glob("*metadata*.csv.gz"))

        # Validation of file existence
        if not cpu_files:
            logger.error("CPU readings file missing", directory=str(data_dir))
            raise FileNotFoundError(f"Could not find a CPU readings file (.csv.gz) in {data_dir}")

        if not metadata_files:
            logger.error("VM metadata file missing", directory=str(data_dir))
            raise FileNotFoundError(f"Could not find a metadata file (.csv.gz) in {data_dir}")

        # Use the first match found for each
        cpu_readings_file = cpu_files[0]
        metadata_file = metadata_files[0]

        # Load CPU readings
        logger.debug("Attempting to read CPU readings", file_path=str(cpu_readings_file))
        cpu_readings_df = pd.read_csv(cpu_readings_file, compression="gzip")

        if cpu_readings_df.empty:
            logger.warning("CPU readings file is empty", file_path=str(cpu_readings_file))

        logger.info("Successfully loaded CPU readings", shape=cpu_readings_df.shape)

        # Load VM metadata
        logger.debug("Attempting to read VM metadata", file_path=str(metadata_file))
        vm_metadata_df = pd.read_csv(metadata_file, compression="gzip")

        if vm_metadata_df.empty:
            logger.warning("VM metadata file is empty", file_path=str(metadata_file))

        logger.info("Successfully loaded VM metadata", shape=vm_metadata_df.shape)

        return cpu_readings_df, vm_metadata_df

    except pd.errors.EmptyDataError as e:
        logger.error("Dataset file was empty and could not be parsed", error=str(e))
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred while loading datasets", error=str(e))
        raise
