"""Data Validation Schemas for Forecasting Project.

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
            checks=Check(
                lambda values: values.astype("string").str.strip().ne("").all(),
                error="values must not be blank",
            ),
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
    checks=[
        Check(lambda df: len(df) > 0, error="dataset must contain at least one row"),
        Check(
            lambda df: (
                (df["min_cpu"] <= df["avg_cpu"] + 1e-6) & (df["avg_cpu"] <= df["max_cpu"] + 1e-6)
            ),
            error="expected min_cpu <= avg_cpu <= max_cpu",
        ),
    ],
    unique=["vm_id", "timestamp"],
    strict=True,
    coerce=True,
)

# Define the schema for VM metadata data
vm_metadata_schema = DataFrameSchema(
    {
        "vm_id": Column(
            pa.String,
            checks=Check(
                lambda values: values.astype("string").str.strip().ne("").all(),
                error="values must not be blank",
            ),
            nullable=False,
            description="Anonymized VM identifier (primary key)",
        ),
        "subscription_id": Column(
            pa.String,
            checks=Check(
                lambda values: values.astype("string").str.strip().ne("").all(),
                error="values must not be blank",
            ),
            nullable=False,
            description="Anonymized subscription identifier",
        ),
        "deployment_id": Column(
            pa.String,
            checks=Check(
                lambda values: values.astype("string").str.strip().ne("").all(),
                error="values must not be blank",
            ),
            nullable=False,
            description="Anonymized deployment identifier",
        ),
        "timestamp_vm_created": Column(
            pa.Int,
            nullable=True,
            description="VM creation timestamp",
        ),
        "timestamp_vm_deleted": Column(
            pa.Int,
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
            pa.String,
            nullable=True,
            description="Bucketed virtual-core count",
        ),
        "memory_gb_bucket": Column(
            pa.String,
            nullable=True,
            description="Bucketed VM memory",
        ),
    },
    checks=Check(
        lambda df: (
            df["timestamp_vm_created"].isna()
            | df["timestamp_vm_deleted"].isna()
            | (df["timestamp_vm_deleted"] >= df["timestamp_vm_created"])
        ),
        error="timestamp_vm_deleted must be >= timestamp_vm_created when both are present",
    ),
    unique=["vm_id"],
    strict=False,
    coerce=True,
)

# Define the schema for aggregate targets data
aggregate_targets_schema = DataFrameSchema(
    {
        "timestamp": Column(
            pa.Int,
            Check.ge(0),
            nullable=False,
            description="Observation timestamp in seconds",
        ),
        "active_vm_count": Column(
            pa.Int,
            Check.ge(0),
            nullable=False,
            description="Number of active VMs",
        ),
        "aggregate_cpu_utilization": Column(
            pa.Float,
            Check.between(0.0, 100.0),
            nullable=False,
            description="Mean CPU utilization percentage",
        ),
    },
    checks=Check(
        lambda df: df["timestamp"].is_monotonic_increasing,
        error="Timestamps must be strictly chronological",
    ),
    unique=["timestamp"],
    strict=True,
    coerce=True,
)

# Submission contract: exactly these columns, finite, non-negative count, CPU in [0, 100].
predictions_schema = DataFrameSchema(
    {
        "timestamp": Column(pa.Int, nullable=False),
        "active_vm_count": Column(pa.Float, Check.in_range(0, 1e12), nullable=False),
        "aggregate_cpu_utilization": Column(pa.Float, Check.in_range(0, 100), nullable=False),
    },
    unique=["timestamp"],
    strict=True,
    coerce=True,
    ordered=True,
    name="predictions",
)
