"""Training Pipeline for the Forecasting Challenge.

This module implements the training pipeline for the forecasting challenge.
It includes data loading, validation, aggregation, model training, forecasting,
and evaluation. The pipeline is designed to be modular and configurable, allowing
for easy adjustments to the preprocessing and forecasting parameters.
"""

from pathlib import Path

import numpy as np
import orjson
import pandas as pd

from workload_forecasting.config.logger import get_logger
from workload_forecasting.config.settings import get_settings
from workload_forecasting.data.aggregation import aggregate_cpu_data
from workload_forecasting.data.loader import load_datasets
from workload_forecasting.data.schemas import predictions_schema
from workload_forecasting.data.validation import (
    ValidationReport,
    validate_datasets,
    validate_targets,
)
from workload_forecasting.modeling.evaluation import score_holdout
from workload_forecasting.modeling.forecasting import forecast_targets

# Initialize the logger for this module
logger = get_logger(__name__)


def train(
    data_dir: str | Path = "challenge_data",
    predictions_path: str | Path = "predictions/predictions.csv",
) -> None:
    """Train the forecasting model and generate predictions.

    Parameters
    ----------
    data_dir : str | Path, optional
        Directory containing the input datasets, by default "challenge_data"
    predictions_path : str | Path, optional
        Path to save the generated predictions, by default "predictions/predictions.csv"
    """
    settings = get_settings()
    params = settings.PARAMS
    report = ValidationReport()

    try:
        # 1. Load the time series data
        cpu_df, metadata_df = load_datasets(data_dir)

        # 2. Validate both the datasets
        validate_datasets(
            cpu_df,
            metadata_df,
            params.preprocessing.step_seconds,
            report,
        )

        # 3. Aggregate the CPU readings to create targets for forecasting
        y = aggregate_cpu_data(cpu_df, params.preprocessing.step_seconds)

        # 4. Validate the aggregated targets
        validate_targets(y, report)

    except Exception:
        report.status = "failed"
        report.save()
        raise
    report.status = "passed"
    report.save()

    # 5. Chronological split: everything below uses ``train`` only, never ``test`` values
    h = params.preprocessing.holdout
    if len(y) <= h:
        raise ValueError(f"Series has {len(y)} points; need more than the holdout ({h})")
    train, test = y.iloc[:-h].reset_index(drop=True), y.iloc[-h:].reset_index(drop=True)

    # 6. Select by rolling-origin CV inside ``train``, refit on ``train``, forecast h steps
    res = forecast_targets(train, h, params.forecasting)

    # 7. Write + validate the submission file (timestamps come from the train grid)
    preds = predictions_schema.validate(res.predictions)
    if not np.array_equal(preds["timestamp"], test["timestamp"]):
        raise ValueError("Forecast timestamps do not match the holdout timestamps")
    out = Path(predictions_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out, index=False)

    # 8. Report. Holdout scores are computed AFTER selection is frozen, for reporting only
    out_dir = settings.TRAINING_REPORT_DIR
    res.cv_scores.to_csv(out_dir / "cv_scores.csv", index=False)
    pd.concat(
        {"lower": res.lower.set_index("timestamp"), "upper": res.upper.set_index("timestamp")},
        axis=1,
    ).to_csv(out_dir / "intervals.csv")
    summary = {"selected": res.selected, "holdout": score_holdout(train, test, res)}
    (out_dir / "holdout_report.json").write_bytes(
        orjson.dumps(summary, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY)
    )
    logger.info("Predictions written", path=str(out), selected=res.selected)
