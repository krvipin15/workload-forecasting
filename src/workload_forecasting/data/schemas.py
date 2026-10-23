"""Data Validation Schemas for Workload Forecasting Project.

This module defines the data validation schemas for the workload
forecasting project using the Pandera library. The schemas are used
to validate the structure and content of the input datasets, ensuring
that they conform to expected formats and constraints.
"""

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

# Define the schema for CPU readings data
cpu_readings_schema = DataFrameSchema(
    {
        "timestamp": Column(
            pa.Int,
            Check.ge(0),
            nullable=False,
            description="Observation timestamp in seconds relative to the source trace",
        ),
        "vm_id": Column(
            pa.String,
            nullable=False,
            description="Anonymized VM identifier",
        ),
        "min_cpu": Column(
            pa.Float,
            Check.between(0.0, 100.0),
            nullable=False,
            description="Minimum CPU utilization during the five-minute interval",
        ),
        "max_cpu": Column(
            pa.Float,
            Check.between(0.0, 100.0),
            nullable=False,
            description="Maximum CPU utilization during the five-minute interval",
        ),
        "avg_cpu": Column(
            pa.Float,
            Check.between(0.0, 100.0),
            nullable=False,
            description="Average CPU utilization during the five-minute interval",
        ),
    },
    strict=True,
    coerce=True,
)

# Define the schema for VM metadata data
vm_metadata_schema = DataFrameSchema(
    {
        "vm_id": Column(
            pa.String,
            nullable=False,
            unique=True,
            description="Anonymized VM identifier (primary key)",
        ),
        "subscription_id": Column(
            pa.String,
            nullable=False,
            description="Anonymized subscription identifier",
        ),
        "deployment_id": Column(
            pa.String,
            nullable=False,
            description="Anonymized deployment identifier",
        ),
        "timestamp_vm_created": Column(
            pa.String,
            nullable=True,
            description="VM creation timestamp",
        ),
        "timestamp_vm_deleted": Column(
            pa.String,
            nullable=True,
            description="VM deletion timestamp",
        ),
        "max_cpu": Column(
            pa.Float,
            Check.between(0.0, 100.0),
            nullable=True,
            description="Source VM-level maximum CPU statistic",
        ),
        "avg_cpu": Column(
            pa.Float,
            Check.between(0.0, 100.0),
            nullable=True,
            description="Source VM-level average CPU statistic",
        ),
        "p95_max_cpu": Column(
            pa.Float,
            Check.between(0.0, 100.0),
            nullable=True,
            description="Source VM-level P95 maximum CPU statistic",
        ),
        "vm_category": Column(
            pa.String,
            nullable=True,
            description="Source VM workload category",
        ),
        "virtual_core_bucket": Column(
            pa.Object,
            nullable=True,
            description="Bucketed virtual-core count",
        ),
        "memory_gb_bucket": Column(
            pa.Object,
            nullable=True,
            description="Bucketed VM memory",
        ),
    },
    strict=False,
    coerce=True,
)
