"""Model Selection and Forecasting with Statsforecast.

This module implements the model selection and forecasting logic for workload
forecasting using the Statsforecast library. It provides functions to convert
data to the required format, perform cross-validation, score models, and
generate forecasts with prediction intervals.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    HistoricAverage,
    Naive,
    SeasonalNaive,
    WindowAverage,
)

from workload_forecasting.config.logger import get_logger
from workload_forecasting.config.settings import ForecastingParams

logger = get_logger(__name__)

# Constants
TARGETS = ("active_vm_count", "aggregate_cpu_utilization")
_META = {"unique_id", "ds", "cutoff", "y"}
ENSEMBLE = "Ensemble_Naive_HistoricAverage"


@dataclass
class ForecastResult:
    """The result of a forecasting run."""

    predictions: pd.DataFrame
    lower: pd.DataFrame
    upper: pd.DataFrame
    selected: dict[str, str]
    cv_scores: pd.DataFrame


def _models(cfg: ForecastingParams) -> list:
    """Return a list of Statsforecast models to evaluate."""
    models = [
        Naive(),
        HistoricAverage(),
        *[WindowAverage(window_size=w, alias=f"WindowAverage_{w}") for w in cfg.window_sizes],
        *[SeasonalNaive(season_length=s, alias=f"SeasonalNaive_{s}") for s in cfg.season_lengths],
        AutoETS(),
        AutoARIMA(),
    ]

    logger.debug(
        "Initialized forecasting models",
        model_count=len(models),
        model_names=[model.alias for model in models],
    )

    return models


def _with_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ensemble column if both Naive and HistoricAverage are present."""
    if {"Naive", "HistoricAverage"} <= set(df.columns):
        df[ENSEMBLE] = (df["Naive"] + df["HistoricAverage"]) / 2

        logger.debug(
            "Added ensemble predictions",
            ensemble=ENSEMBLE,
            row_count=len(df),
        )
    else:
        logger.debug(
            "Skipped ensemble predictions because required models are missing",
            available_columns=list(df.columns),
        )

    return df


def to_long(y: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide-format DataFrame to long format for Statsforecast."""
    logger.debug(
        "Converting training data to long format",
        input_shape=y.shape,
        targets=list(TARGETS),
    )

    if y.empty:
        logger.warning("Cannot convert empty training data to long format")
        return pd.DataFrame(columns=["unique_id", "ds", "y"])

    required_columns = {"timestamp", *TARGETS}
    missing_columns = required_columns - set(y.columns)

    if missing_columns:
        logger.error(
            "Training data is missing required columns",
            missing_columns=sorted(missing_columns),
        )
        raise ValueError(f"Training data is missing required columns: {sorted(missing_columns)}")

    timestamps = y["timestamp"]

    if timestamps.isna().any():
        raise ValueError("Training timestamps must not contain missing values")

    if timestamps.duplicated().any():
        raise ValueError("Training timestamps must be unique")

    timestamps = timestamps.sort_values()

    if not timestamps.equals(y["timestamp"]):
        logger.debug("Sorting training data by timestamp")
        y = y.sort_values("timestamp").reset_index(drop=True)

    differences = y["timestamp"].diff().dropna()

    if not differences.empty:
        step = int(differences.iloc[0])

        if step <= 0 or not differences.eq(step).all():
            raise ValueError("Training timestamps must have a constant positive interval")
    else:
        step = 1

    t0 = y["timestamp"].min()

    long = y.melt(
        id_vars="timestamp",
        value_vars=list(TARGETS),
        var_name="unique_id",
        value_name="y",
    )

    long["ds"] = (long["timestamp"] - t0) // step

    result = long[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)

    logger.info(
        "Converted training data to long format",
        input_rows=len(y),
        output_rows=len(result),
        target_count=len(TARGETS),
        time_step_seconds=step,
    )

    logger.debug(
        "Long-format data summary",
        rows_per_target=result.groupby("unique_id").size().to_dict(),
        missing_values=result.isna().sum().to_dict(),
    )

    return result


def cross_validate(
    long: pd.DataFrame,
    h: int,
    cfg: ForecastingParams,
) -> pd.DataFrame | None:
    """Perform cross-validation on the long-format DataFrame.

    Returns cross-validation results or None if there is insufficient history.
    """
    if long.empty:
        logger.warning("Cross-validation skipped because input data is empty")
        return None

    if h < 1:
        raise ValueError("Forecast horizon must be at least 1")

    lengths = long.groupby("unique_id").size()
    n = lengths.min()

    n_windows = min(
        cfg.cv_windows,
        (n - h - cfg.min_train) // cfg.cv_step + 1,
    )

    logger.debug(
        "Calculated cross-validation window count",
        minimum_series_length=n,
        forecast_horizon=h,
        minimum_training_length=cfg.min_train,
        configured_windows=cfg.cv_windows,
        step_size=cfg.cv_step,
        calculated_windows=n_windows,
    )

    if n_windows < 1:
        logger.info(
            "Insufficient history for cross-validation; using baseline forecasting",
            minimum_series_length=n,
            forecast_horizon=h,
            minimum_training_length=cfg.min_train,
        )
        return None

    models = _models(cfg)

    logger.info(
        "Starting time-series cross-validation",
        series_count=len(lengths),
        model_count=len(models),
        forecast_horizon=h,
        windows=n_windows,
    )

    sf = StatsForecast(
        models=models,
        freq=1,
        n_jobs=1,
    )

    cv = sf.cross_validation(
        df=long,
        h=h,
        n_windows=n_windows,  # ty: ignore[invalid-argument-type]
        step_size=cfg.cv_step,
        refit=True,
    )

    cv = _with_ensemble(cv)

    logger.info(
        "Completed time-series cross-validation",
        result_rows=len(cv),
        result_columns=len(cv.columns),
    )

    logger.debug(
        "Cross-validation result summary",
        targets=cv["unique_id"].unique().tolist(),
        cutoffs=int(cv["cutoff"].nunique()) if "cutoff" in cv else None,
        missing_values=cv.isna().sum().to_dict(),
    )

    return cv


def score_cv(
    cv: pd.DataFrame,
    long: pd.DataFrame,
    cfg: ForecastingParams,
) -> pd.DataFrame:
    """Score cross-validation results using MAE, RMSE, MASE, and peak MAE."""
    if cv.empty:
        logger.warning("Cannot score empty cross-validation results")
        return pd.DataFrame()

    logger.info(
        "Scoring cross-validation predictions",
        result_rows=len(cv),
    )

    rows = []

    for uid, group in cv.groupby("unique_id"):
        y_train = long.loc[long["unique_id"] == uid, "y"]

        if y_train.empty:
            logger.warning(
                "Skipping target with no training observations",
                target=uid,
            )
            continue

        scale = np.abs(np.diff(y_train.to_numpy())).mean()

        peak_threshold = y_train.quantile(cfg.peak_quantile)
        peak = (group["y"] >= peak_threshold).to_numpy()

        model_columns = [column for column in group.columns if column not in _META]

        logger.debug(
            "Scoring target",
            target=uid,
            training_observations=len(y_train),
            validation_observations=len(group),
            peak_threshold=float(peak_threshold),
            models=model_columns,
        )

        for model_name in model_columns:
            errors = (group["y"] - group[model_name]).to_numpy()

            mae = float(np.abs(errors).mean())
            rmse = float(np.sqrt(np.mean(errors**2)))
            mase = mae / scale if scale > 0 else np.nan
            peak_mae = float(np.abs(errors[peak]).mean()) if peak.any() else np.nan

            rows.append(
                {
                    "target": uid,
                    "model": model_name,
                    "mae": mae,
                    "rmse": rmse,
                    "mase": mase,
                    "peak_mae": peak_mae,
                }
            )

            logger.debug(
                "Calculated model evaluation metrics",
                target=uid,
                model=model_name,
                mae=mae,
                rmse=rmse,
                mase=mase,
                peak_mae=peak_mae,
            )

    out = pd.DataFrame(rows)

    if out.empty:
        logger.warning("No cross-validation metrics were calculated")
        return out

    naive = out.loc[out["model"] == "Naive"].set_index("target")["mae"]

    # Relative improvement over Naive; NaN if the baseline is exact.
    baseline_mae = out["target"].map(naive).replace(0, np.nan)
    out["skill_vs_naive"] = 1 - out["mae"] / baseline_mae

    logger.info(
        "Completed cross-validation scoring",
        scored_models=len(out),
        targets=out["target"].nunique(),
    )

    logger.debug(
        "Cross-validation ranking",
        best_models=(
            out.sort_values("mae").groupby("target", sort=False).first()["model"].to_dict()
        ),
    )

    return out


def forecast_targets(  # noqa: PLR0915
    train: pd.DataFrame,
    h: int,
    cfg: ForecastingParams,
) -> ForecastResult:
    """Forecast target variables using Statsforecast models.

    Parameters
    ----------
    train : pd.DataFrame
        Training data in wide format containing timestamp,
        active_vm_count, and aggregate_cpu_utilization.
    h : int
        Forecast horizon in future time steps.
    cfg : ForecastingParams
        Model settings and cross-validation parameters.

    Returns
    -------
    ForecastResult
        Forecast predictions, lower and upper prediction intervals,
        selected models, and cross-validation scores.
    """
    if train.empty:
        raise ValueError("Training data must not be empty")

    if h < 1:
        raise ValueError("Forecast horizon must be at least 1")

    logger.info(
        "Starting target forecasting",
        training_rows=len(train),
        forecast_horizon=h,
        targets=list(TARGETS),
    )

    if len(train) > 1:
        differences = train["timestamp"].diff().dropna()

        if differences.empty or differences.iloc[0] <= 0:
            raise ValueError("Training timestamps must increase")

        if not differences.eq(differences.iloc[0]).all():
            raise ValueError("Training timestamps must have a constant interval")

        step = int(differences.iloc[0])
    else:
        step = 1

    long = to_long(train)
    cv = cross_validate(long, h, cfg)

    if len(train) > 1:
        differences = train["timestamp"].diff().dropna()

        if differences.empty or differences.iloc[0] <= 0:
            raise ValueError("Training timestamps must increase")

        if not differences.eq(differences.iloc[0]).all():
            raise ValueError("Training timestamps must have a constant interval")

        step = int(differences.iloc[0])
    else:
        step = 1

    timestamps = train["timestamp"].iloc[-1] + step * np.arange(1, h + 1)

    if cv is None:
        logger.info(
            "Using Naive baseline because cross-validation is unavailable",
        )

        scores = pd.DataFrame()
        selected = dict.fromkeys(TARGETS, "Naive")
        models = [Naive()]
    else:
        scores = score_cv(cv, long, cfg)

        if scores.empty:
            logger.warning("No valid model scores available; falling back to Naive")
            selected = dict.fromkeys(TARGETS, "Naive")
        else:
            selected = {
                target: group.sort_values(
                    "mae",
                    kind="stable",
                ).iloc[0]["model"]
                for target, group in scores.groupby(
                    "target",
                    sort=False,
                )
            }

            for target in TARGETS:
                if target not in selected:
                    logger.warning(
                        "No model score found for target; using Naive",
                        target=target,
                    )
                    selected[target] = "Naive"

        logger.info(
            "Selected forecasting models",
            selected_models=selected,
        )

        models = _models(cfg)

    logger.debug(
        "Generating forecasts",
        model_names=[model.alias for model in models],
        series_count=long["unique_id"].nunique(),
        forecast_horizon=h,
    )

    fc = StatsForecast(
        models=models,
        freq=1,
        n_jobs=1,
    ).forecast(
        df=long,
        h=h,
    )

    fc = _with_ensemble(fc)

    interval_tail = (1 - cfg.interval_level / 100) / 2

    predictions = {"timestamp": timestamps}
    lower = {"timestamp": timestamps}
    upper = {"timestamp": timestamps}

    for target in TARGETS:
        target_forecasts = fc.loc[fc["unique_id"] == target]

        if selected[target] not in target_forecasts.columns:
            logger.warning(
                "Selected model missing from forecast output; using Naive",
                target=target,
                selected_model=selected[target],
            )
            selected[target] = "Naive"

        point_forecast = target_forecasts[selected[target]].to_numpy()

        if cv is None:
            error_lower = error_upper = 0.0
        else:
            target_cv = cv.loc[cv["unique_id"] == target]

            if target_cv.empty:
                logger.warning(
                    "No cross-validation errors available; using zero-width interval",
                    target=target,
                )
                error_lower = error_upper = 0.0
            else:
                errors = (target_cv["y"] - target_cv[selected[target]]).dropna()

                if errors.empty:
                    logger.warning(
                        "Cross-validation errors are unavailable; using zero-width interval",
                        target=target,
                    )
                    error_lower = error_upper = 0.0
                else:
                    error_lower, error_upper = errors.quantile(
                        [
                            interval_tail,
                            1 - interval_tail,
                        ]
                    ).to_numpy()

        limits = (0, 100) if target == "aggregate_cpu_utilization" else (0, np.inf)

        predictions[target] = np.clip(
            point_forecast,
            *limits,
        )
        lower[target] = np.clip(
            point_forecast + error_lower,
            *limits,
        )
        upper[target] = np.clip(
            point_forecast + error_upper,
            *limits,
        )

        logger.debug(
            "Generated target forecast",
            target=target,
            selected_model=selected[target],
            forecast_steps=len(point_forecast),
            error_lower=float(error_lower),
            error_upper=float(error_upper),
            forecast_min=float(np.min(point_forecast)),
            forecast_max=float(np.max(point_forecast)),
        )

    result = ForecastResult(
        predictions=pd.DataFrame(predictions),
        lower=pd.DataFrame(lower),
        upper=pd.DataFrame(upper),
        selected=selected,  # ty: ignore[invalid-argument-type]
        cv_scores=scores,
    )

    logger.info(
        "Completed target forecasting",
        forecast_rows=len(result.predictions),
        selected_models=result.selected,
        cross_validation_rows=len(result.cv_scores),
    )

    return result
