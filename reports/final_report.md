# Cloud Workload Forecasting: Azure VM Trace (`data-v1`)

## 1. Executive summary

**How predictable is this workload?** At fleet level, very predictable in the short term, but only because it barely moves. The mean CPU utilisation of the 10,000 sampled VMs stays between about 9.1% and 9.9% over the 9.2-hour training window, and the 10,000-VM count never changes.

**How accurate is the forecast?** On the 2-hour holdout, the selected model reaches MAE 0.124 and RMSE 0.154 percentage points for `aggregate_cpu_utilization`, against 0.147 and 0.185 for a last-value baseline (15% lower MAE, 17% lower RMSE). `active_vm_count` is forecast exactly (error 0) because it is constant in this sample.

**Read the 15% with care.** The model was chosen by rolling-origin cross-validation inside the training window, where its advantage over the baseline was only 3.6% in MAE (7.7% in RMSE). The holdout is 24 strongly autocorrelated points, effectively a single outcome. The honest summary is a small but consistent improvement over a baseline that is already hard to beat.

**What should an infrastructure team do?** Provision to the **upper 90% band** of the forecast rather than to the point forecast. On the holdout this removed all under-provisioning (37.5% of steps under-provisioned at the point forecast, 0% at the upper band) at an average over-provision of 0.46 percentage points (against 0.07 at the point forecast). At roughly 9% fleet utilisation, the fleet-level risk of running out is negligible. The decisions that matter operationally are per-segment and per-VM, which this sample can only describe.

**Two limits shape everything below.** (1) The release keeps only VMs with complete 134-point traces, so there is no VM churn in the data, and `active_vm_count` carries no forecasting signal. (2) The trace spans 11.1 hours, so daily or weekly seasonality cannot be learned.

## 2. Data understanding and preprocessing

The data is a curated subset of Azure Public Dataset V2: 10,000 VMs, 134 readings each at 300-second spacing (1,340,000 rows), plus one metadata row per VM.

**Quality checks** (loader and pandera schemas, `reports/validation/`; notebook section 1, `figures/eda_01_cadence_coverage.png`, `figures/du_01_cadence_coverage.png`):

- one cadence only (300 s), no gaps, no duplicate `(vm_id, timestamp)` pairs, no nulls, every VM present at every timestamp;
- readings inside `[0, 100]` and `min_cpu <= avg_cpu <= max_cpu` enforced by schema;
- CPU and metadata populations match one to one.

**Target construction.** `active_vm_count(t)` is the number of distinct VM IDs with a valid reading at `t`. `aggregate_cpu_utilization(t)` is the mean of `avg_cpu` over those VMs, in percent. Because the panel is complete there is nothing to impute: the pipeline builds a continuous five-minute grid and **raises an error** if any timestamp is missing, instead of inventing values. No resampling is performed, so the forecast is natively at the required cadence and needs no mapping back.

**Split.** The full 134-point target series is built first. The first 110 points (to 9.08 hours) are `train`; the last 24 (timestamps 33000 to 39900 s, 2 hours) are the holdout. The training window is defined as "all but the last `holdout` points", not as a fixed 110, so a smaller dataset still runs.

**Lifetimes and churn** (`figures/du_03_lifetimes_churn.png`, notebook section 2). All 10,000 VMs existed before the window and survive it: no creations or deletions occur inside the window. Roughly 8,000 of the 10,000 VMs fall in the longest lifetime bin, on the order of 600 to 700 hours (the source trace covers 30 days), far longer than the 11-hour window. The sample is therefore a **survivor population** of long-lived tenants.

## 3. Workload analysis

**Segments** (`figures/du_02_metadata_segments.png`). The sample is stratified by category and core count: 6,180 delay-insensitive, 3,091 interactive and 729 unknown VMs, with 2-vCPU VMs dominant in every category. The deployments are mostly small (a long tail up to about 100 VMs per deployment), and the largest subscriptions hold 65 to 96 sampled VMs each.

**Level and shape** (`figures/du_04_targets_train.png`, `figures/eda_04_targets.png`). The aggregate fluctuates around 9.4% with short-lived noise. There is one visible rise between roughly hour 4.4 and hour 6, to about 9.9%, followed by a return to 9.2 to 9.4%. The high-demand period is a broad shift lasting about 1.5 hours rather than an isolated spike, which matters for any trigger rule. The lowest readings (about 9.1%) recur at several points, and the final two hours of the training window sit at the low end (about 9.1 to 9.4%), so the forecast origin is in a low-demand stretch (notebook section 4).

**Autocorrelation and stationarity** (`figures/eda_05_acf.png`). The notebook reports the ACF of the level and of its first difference and an ADF test on each. \[FILL: ADF p-values, level and first difference, from `reports/eda_summary.json` key `adf_p`.\]

**Why persistence is not enough.** The notebook compares the error of the last value against that of always predicting the training mean (`figures/eda_06_persistence_vs_horizon.png`). \[FILL: horizon at which persistence stops beating the training mean, from key `persistence_vs_mean`.\] This is the evidence for the blended forecast chosen in section 6.

**VM-level skew** (notebook section 3). \[FILL: share of VMs with mean CPU below 5%, and share of total CPU from the top 10% of VMs, from key `cpu_stats`.\] The point is that a low fleet mean hides a small set of heavy VMs.

**Why the aggregate is smooth** (notebook section 7). The variance of a mean of N units equals `(sigma^2 / N) (1 + (N-1) rho)`, which lets us back out the implied average pairwise correlation between VMs. \[FILL: `implied_pairwise_corr` from `eda_summary.json`.\] A small positive value still leaves a visible common movement at fleet level, and that common component is what a fleet-level model can learn.

**Operational meaning.** (a) The fleet is lightly loaded, so the question is headroom reclamation and segment-level risk, not fleet-level shortage. (b) The aggregate hides VM and segment heterogeneity; any real scaling rule should be applied per segment. (c) Because the sample contains no churn, demand volatility from arriving and departing VMs is invisible here, so conclusions apply to a stable tenant base only.

## 4. Forecasting approach

**Candidates** (all `statsforecast`, chosen because the tutorial methods fit this problem): `Naive` (baseline), `HistoricAverage`, `WindowAverage` over 6 and 12 steps, `SeasonalNaive` with a one-hour season, `AutoETS`, `AutoARIMA`, and an equal-weight **ensemble of Naive and HistoricAverage**.

**Why these and not others.** With 110 training points and no future-known inputs, there is nothing for exogenous regressors to use, and a 24-hour season (288 steps) cannot be identified, so there are no calendar or Fourier features. Only the target history is used as input, which makes every feature available at the forecast origin by construction. Gradient boosting or neural models would have fewer than 110 rows to learn from and were not attempted.

**Ensemble rationale.** Naive tracks the latest level; HistoricAverage reverts to the mean. They make different kinds of error (level-following versus mean-reverting), so an equal-weight blend is a cheap hedge. No weights are tuned.

**Intervals.** For each target, the band is the empirical 5th and 95th percentile of the selected model's cross-validation errors (96 pooled errors from 4 folds of 24 steps), added to the point forecast and clipped to the valid range. The same quantiles are used at every horizon step, which is a simplification (section 9).

**Final forecast.** The selected model is refit on the 110 training points and predicts 24 steps. Forecasts are clipped to `[0, 100]` for utilisation and `>= 0` for counts.

## 5. Validation methodology

**Rolling-origin evaluation inside the training window.** Four folds, each forecasting 24 steps, with origins one hour (12 steps) apart. The models are refit at every origin. The earliest fold trains on 50 points and the last on 86, so every fold respects chronological order and uses a 2-hour horizon, the same as production. This mirrors the deployment problem: forecast the next two hours from everything seen so far.

**Metrics.** MAE, RMSE, MASE (scaled by the in-sample one-step naive MAE, so no percentage-error denominators near zero), relative improvement over Naive, and MAE restricted to peak points. A "peak" is a point at or above the 90th percentile of the training window.

**Selection rule.** Lowest pooled cross-validation MAE per target. Ties resolve to the simpler model, which matters for `active_vm_count`, where all candidates have zero error and Naive is chosen.

**Leakage controls.**

- Forecasting code receives only the training rows; it cannot read holdout values.
- Forecast timestamps are generated from the training grid and then checked against the holdout timestamps.
- Peak thresholds, volatility estimates and segment statistics in the analysis use training data only.
- Metadata statistics that cover the holdout period (`avg_cpu`, `p95_max_cpu`, deletion time) are not used as features.
- Holdout metrics are computed once, after selection, and are used only for reporting.

**Disclosure.** The release makes the holdout values public, and an earlier pipeline run, which contained only the baselines and statistical models, showed a flat Naive forecast for `aggregate_cpu_utilization`. The ensemble candidate was added afterwards to give the pipeline a hedge between level and mean, and it was kept because it **won cross-validation** (training window only), not because of its holdout score. Readers should still discount the holdout comparison accordingly.

## 6. Results versus baseline

**Cross-validation, `aggregate_cpu_utilization`** (training window only; 4 folds):

| Model                                  |        MAE |       RMSE | MASE |   Peak MAE | Skill vs Naive |
| -------------------------------------- | ---------: | ---------: | ---: | ---------: | -------------: |
| **Ensemble (Naive + HistoricAverage)** | **0.1982** | **0.2345** | 1.64 |     0.3470 |      **+3.6%** |
| Naive (baseline)                       |     0.2056 |     0.2540 | 1.70 | **0.2953** |           0.0% |
| HistoricAverage                        |     0.2141 |     0.2497 | 1.77 |     0.4159 |          -4.1% |
| AutoARIMA                              |     0.2334 |     0.2760 | 1.93 |     0.3247 |         -13.5% |
| AutoETS                                |     0.2355 |     0.2772 | 1.94 |     0.3247 |         -14.5% |
| WindowAverage (12)                     |     0.2545 |     0.2972 | 2.10 |     0.3607 |         -23.8% |
| SeasonalNaive (12)                     |     0.2634 |     0.3142 | 2.18 |     0.3976 |         -28.1% |
| WindowAverage (6)                      |     0.2659 |     0.3077 | 2.20 |     0.3340 |         -29.3% |

**Holdout, selected model** (24 steps, one-shot):

| Target                      | Model    |    MAE |   RMSE | MASE | Baseline MAE | Skill vs Naive | 90% band coverage |
| --------------------------- | -------- | -----: | -----: | ---: | -----------: | -------------: | ----------------: |
| `aggregate_cpu_utilization` | Ensemble | 0.1245 | 0.1538 | 1.03 |       0.1468 |         +15.2% |              100% |
| `active_vm_count`           | Naive    |      0 |      0 |  n/a |            0 |            n/a |              100% |

The Naive baseline on the same holdout had RMSE 0.1848 and MASE 1.21. MASE is undefined for `active_vm_count` because a constant series has zero one-step variation.

**Interpretation.**

- Every candidate other than the ensemble (ETS, ARIMA, window averages, seasonal naive, historic average) has **higher MAE than the baseline** in cross-validation. On a 110-point series with no seasonality or trend that is identifiable, extra flexibility fits noise.
- The forecast for `aggregate_cpu_utilization` is a **flat line at 9.289%** with a band of 8.964% to 9.724%. A flat forecast is the correct output of both candidates that win here, not a bug: neither Naive nor HistoricAverage has dynamics.
- The 15% holdout gain is larger than the 3.6% cross-validation gain. The holdout is a single 2-hour window of strongly autocorrelated points, so the true expected gain is closer to the cross-validation figure.

## 7. Error and failure analysis

**Where the model is weaker than the baseline: peaks.** In cross-validation the ensemble's MAE on peak points is 0.347 against 0.295 for Naive (17.5% worse). Pulling the forecast toward the training mean systematically under-predicts high load, which is when under-provisioning hurts. This is the model's main failure mode and the reason for provisioning to the upper band rather than the point forecast.

**Errors at long horizons.** All candidates have MASE well above 1 in cross-validation (1.64 to 2.20), meaning the 24-step forecast is worse than a one-step-ahead naive forecast in-sample. This is expected for a horizon of two hours and reflects the genuinely unpredictable short-term movement of the aggregate rather than a modelling defect.

**Bias on the holdout.** With a forecast of 9.289% and a flat prediction, point-forecast provisioning was exceeded on 37.5% of holdout steps (mean shortfall 0.052 points), while the actual value was below the forecast on the other 62.5% of steps (mean waste 0.072 points). [FILL: bias and horizon trend from the notebook's section 8 printout.]

**Why the interval over-covers.** The band covered 100% of holdout points against a 90% target (the earlier Naive-based run covered 95.8%). With only 96 pooled cross-validation errors from four overlapping folds, the quantiles are noisy, and a constant width across horizon steps ignores that errors grow with horizon.

**No structural breaks or anomalies were tested.** The 2-hour holdout contains no known regime change, so robustness to deployment surges or failures is untested.

## 8. Capacity-planning recommendation

**Policy.**

1. **Fleet CPU headroom:** provision against the **upper 90% band** of the 2-hour forecast, refreshed every 5 minutes with a refit on all data up to the current origin. In this sample the band sits about 0.44 percentage points above the point forecast.
1. **Peak-risk trigger:** if the observed aggregate exceeds the upper band for two consecutive 5-minute steps, raise an alert and widen the margin until the next refit. The sustained 1.5-hour rise seen in the training window is the kind of event this catches.
1. **Segment rule:** apply the same band logic per VM category and core bucket, because the aggregate hides which workload class is moving and smaller segments are noisier. This sample can only describe segments, not validate per-segment forecasts, so treat it as a design principle and test it on a longer trace.
1. **VM slots:** keep `active_vm_count` at the last observed value in this sample. In a real fleet add a buffer sized from observed arrival rates, which this dataset cannot provide.

**Trade-off, measured on the holdout:**

| Policy         | Under-provisioned steps | Mean shortfall (pts) | Mean waste (pts) |
| -------------- | ----------------------: | -------------------: | ---------------: |
| Point forecast |                   37.5% |                0.052 |            0.072 |
| Upper 90% band |                      0% |                0.000 |            0.455 |

Moving from the point forecast to the upper band removes 37.5 percentage points of shortfall events for about 0.38 additional points of average waste. At around 9% fleet utilisation the absolute amounts are small. The decision should be driven by what a shortfall costs: an SLA breach on interactive VMs justifies the band; a delay-insensitive batch pool can tolerate the point forecast or lower. The 100% interval coverage on this holdout indicates the band is conservative, so do not widen it further without evidence.

**Caution.** A utilisation near 9% might suggest large amounts of reclaimable capacity, but fleet means do not describe VM-level peaks. Check VM-level maxima and the heavy tail before consolidating anything.

## 9. Limitations and future work

- **Short trace.** 11.1 hours cannot reveal daily or weekly cycles, so the "predictable" conclusion applies to the next two hours only.
- **Survivor sample.** Only complete-trace VMs are included, so there is no churn and `active_vm_count` is constant. Results are not representative of fleets with short-lived VMs.
- **One holdout window.** 24 correlated points make a weak test. Rolling-origin results (4 folds) are the more reliable evidence and show a small gain.
- **Few CV folds.** Four overlapping folds make interval quantiles and model rankings noisy; the band uses the same width at every horizon.
- **Public holdout.** Honesty about the leakage rules is the only protection (section 5 discloses the one place where the holdout result was seen before the final candidate set was fixed).
- **Not done:** segment-level forecasts, anomaly detection, uncertainty that grows with horizon (conformal or quantile models), a comparison with another cloud trace.
- **Future work.** Use the full 30-day Azure trace to learn diurnal and weekly seasonality; model arrivals and departures explicitly; forecast per segment with hierarchical reconciliation; replace pooled-error bands with horizon-specific conformal intervals; evaluate a cost-weighted policy using real SLA penalties.

## Appendix A. Reproduction

```bash
uv sync
uv run workload-forecasting predict --data challenge_data --output predictions/predictions.csv
uv run pytest
```

Settings are in `params.yaml` (`holdout: 24`, `cv_windows: 4`, `cv_step: 12`, `interval_level: 90`). Outputs are listed in `README.md`.

## Appendix B. Figures

`figures/du_01` to `du_04` (data understanding) and `figures/eda_01` to `eda_08` (generated by `notebooks/01_eda.ipynb`).
