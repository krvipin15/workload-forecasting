# Workload Forecasting: Azure VM trace

Short-horizon forecasting of two fleet-level targets from the Azure Public Dataset V2 subset (`data-v1`):

- `active_vm_count`: distinct VMs with a valid CPU reading at each 5-minute timestamp.
- `aggregate_cpu_utilization`: mean of VM-level `avg_cpu` (%) at each timestamp.

The final 24 observations (2 hours) are held out. The pipeline selects a model by rolling-origin cross-validation **inside the first 110 observations only**, refits on those 110, and forecasts the 24 holdout timestamps.

Full write-up: [`reports/final_report.md`](reports/final_report.md). Exploratory analysis: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
# 1. Install (runtime dependencies only)
uv sync

# 2. Produce predictions.csv
uv run workload-forecasting predict --data challenge_data --output predictions/predictions.csv
```

Equivalent module form: `uv run python -m workload_forecasting predict --data challenge_data --output predictions/predictions.csv`.

Runtime is a few seconds on the full data. The command writes the file the submission contract asks for (`timestamp,active_vm_count,aggregate_cpu_utilization`, 24 rows, no index) and validates it before saving; it does not need manual editing.

## Dataset

Extract the `data-v1` release so the repository contains:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

Verify integrity with `cd challenge_data && sha256sum -c SHA256SUMS`. The data files are git-ignored and are not redistributed here.

## What the pipeline does

1. **Load** both files and **validate** them with pandera schemas (types, ranges, uniqueness, `min <= avg <= max`, 300 s cadence, VM population match). Problems are written to `reports/validation/validation_report.json`.
1. **Aggregate** VM readings into the two targets on a complete 5-minute grid. Each VM is counted once per timestamp; gaps raise an error instead of being filled.
1. **Split** chronologically: the last `holdout` (24) points are the test window, everything before is `train`.
1. **Select** a model per target using rolling-origin cross-validation on `train` (4 folds, 24-step horizon, 12-step origin spacing, refit at every origin). Selection criterion: lowest MAE; ties go to the simpler model.
1. **Forecast** the 24 holdout timestamps from a refit on `train`. Timestamps are derived from the training grid and checked against the holdout timestamps.
1. **Report** holdout metrics once, after selection is frozen. They are never fed back into selection.

Candidates (all from `statsforecast`): `Naive` (baseline), `HistoricAverage`, `WindowAverage` (6 and 12 steps), `SeasonalNaive` (12 steps), `AutoETS`, `AutoARIMA`, and an equal-weight ensemble of `Naive` and `HistoricAverage`. Settings live in `params.yaml`.

Prediction intervals are empirical quantiles (5% and 95%) of the selected model's cross-validation errors, so they work for any candidate and use no holdout information.

## Outputs

| File                                        | Content                                                                    |
| ------------------------------------------- | -------------------------------------------------------------------------- |
| `predictions/predictions.csv`               | The submission file (24 rows).                                             |
| `reports/training/cv_scores.csv`            | CV MAE, RMSE, MASE, peak-MAE and skill versus Naive, per target and model. |
| `reports/training/intervals.csv`            | Lower and upper 90% band per holdout timestamp.                            |
| `reports/training/holdout_report.json`      | Selected models, holdout metrics, and the capacity-policy comparison.      |
| `reports/validation/validation_report.json` | Data validation status and any errors.                                     |
| `reports/eda_summary.json`                  | Numbers exported by the EDA notebook.                                      |

## Notebook

```bash
uv sync --group eda        # matplotlib, seaborn, ipykernel
```

Open `notebooks/01_eda.ipynb` with Jupyter or VS Code using the `.venv` kernel, and run all cells. Run the pipeline first so the last section can read its outputs. Figures are written to `reports/figures/`.

## Tests

```bash
uv sync --group test
uv run pytest
```

## Main modelling decisions

- **Baseline first.** A last-value forecast is evaluated with the same cross-validation as every other model. It turned out to be hard to beat.
- **No exogenous or calendar features.** Nothing about the holdout period is known at the forecast origin, and 11 hours cannot identify daily or weekly cycles. Only the target history is used.
- **Seasonality limited to one hour.** A 24-hour season cannot be estimated from an 11-hour trace.
- **No metadata as predictive features.** Metadata statistics such as `avg_cpu`, `p95_max_cpu` and deletion time cover periods that include the holdout, so they fail the "computable at the forecast origin" test. Metadata is used for description only.
- **`active_vm_count` is constant (10,000) by construction** because the release keeps only VMs with complete traces, so the persistence forecast is exact. This is a property of the sample, not evidence about real fleets.

## Reproducibility and smoke datasets

Nothing is hard-coded to 10,000 VMs, 134 observations or 110 training points: the holdout length (`holdout: 24`) and cadence (`step_seconds: 300`) come from `params.yaml`, and the training window is "everything except the last `holdout` points". If the history is too short for cross-validation, the pipeline falls back to the `Naive` baseline and a zero-width band instead of failing.

## Repository layout

```text
src/workload_forecasting/
├── cli.py, __main__.py        command-line entry point
├── config/                    settings, params loading, logging
├── data/                      loader, schemas, validation, aggregation
├── modeling/                  forecasting.py (CV, selection, forecast), evaluation.py (holdout, policy)
└── pipelines/training.py      end-to-end orchestration
notebooks/01_eda.ipynb
reports/final_report.md
tests/
```
