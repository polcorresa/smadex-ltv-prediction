# Smadex LTV Prediction System

Low-level engineering stack for forecasting 7-day in-app purchase revenue (`iap_revenue_d7`) on Smadex traffic. The system blends Dask-based ingestion, aggressive nested-feature parsing, histogram-aware sampling (HistOS/HistUS), an ODMN-inspired multi-horizon regressor, and a stacking ensemble calibrated on buyer intent.

> **Key ideas:** (1) multi-stage modeling of buyer intent and conditional revenue, (2) order-preserving constraints across D1/D7/D14 horizons, (3) stratified random splits to keep buyer ratios stable, and (4) fast inference for submissions or dashboards.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Environment Setup](#environment-setup)
3. [Configuration & Data Inputs](#configuration--data-inputs)
4. [Quick Iteration Workflow](#quick-iteration-workflow)
5. [End-to-End Pipeline](#end-to-end-pipeline)
6. [Operational Commands](#operational-commands)
7. [Testing & Quality Gates](#testing--quality-gates)
8. [Logging, Artifacts & Outputs](#logging-artifacts--outputs)
9. [Troubleshooting & Tuning](#troubleshooting--tuning)
10. [Research Lineage](#research-lineage)

---

## Repository Layout

| Path | Purpose |
|------|---------|
| `README.md` | This document – complete pipeline walkthrough. |
| `ITERATION_README.md` | Step-by-step data-scaling playbook (SMALL→XLARGE). |
| `pyproject.toml`, `setup.py` | Package metadata; installable `src/` module. |
| `config/config*.yaml` | Production + test configs (paths, splits, hyperparams). |
| `config/features.yaml` | Feature-groups catalog used by the engineer/selector. |
| `data/raw/` | Expected parquet inputs (`train/train/datetime=...`). Not tracked. |
| `data/processed/` | Cached feature matrices (created when `cache_features=true`). |
| `data/submissions/` | Model submissions (CSV) from `scripts/predict.py`. |
| `frontend/` | Streamlit dashboard (`frontend/app.py`). |
| `scripts/` | CLI utilities: `train.py`, `predict.py`, `test_model_validation.py`, etc. |
| `src/` | All reusable modules (data loader, preprocessing, models, training, inference, utils). |
| `tests/` | Pytest suite (data quality, pipeline sanity, regression tests). |

---

## Environment Setup

Python ≥ 3.10 is required. Commands below use [uv](https://docs.astral.sh/uv/) for reproducible environments; swap with `python -m venv` + `pip` if preferred.

```bash
# 1) Install uv (once per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Create & activate a virtualenv
uv venv
source .venv/bin/activate

# 3) Install project in editable mode
uv pip install -e .
```

GPU training is optional; LightGBM runs in CPU mode by default. Ensure `libomp` (macOS) or `build-essential` (Linux) is installed before compiling LightGBM/XGBoost.

---

## Configuration & Data Inputs

All critical knobs live in the YAML configs:

- **Paths & time ranges** (`data.*`): training/validation/test windows expressed as partition names (`datetime=YYYY-MM-DD-HH-MM`).
- **Randomized splits** (`training.split`): `strategy: stratified_random` loads a combined `model_start`→`model_end` window and applies `sklearn.model_selection.train_test_split` stratified on `buyer_d7`. Switch to `strategy: temporal` to revert to strictly chronological holdouts.
- **Sampling** (`training.sampling`): `train_frac`, `val_frac`, and `max_*_partitions` control Dask sampling prior to `.compute()`. `fallback_val_rows` prevents empty validation sets.
- **Histogram sampling** (`sampling.histos`): shared HistOS/HistUS hyperparameters (bins, window size, percentile targets).
- **Model hyperparams** (`models.*`): LightGBM settings for Stage 1, ODMN regressors, and stacking ensemble.

Feature documentation (group membership, on/off switches) is centralized in `config/features.yaml` for consistency with notebooks and EDA.

**Data expectations:** parquet files under `data/raw/train/train/datetime=*/part-*.parquet` (same pattern for test). Data is not stored in the repo; update the paths if your partitions live elsewhere.

---

## Quick Iteration Workflow

`ITERATION_README.md` describes a manual scaling loop. Each iteration simply plugs a different config into the validation harness:

```bash
uv run python scripts/test_model_validation.py config/config_test_small.yaml   # ~15s
uv run python scripts/test_model_validation.py config/config_test_medium.yaml  # ~1-2m
uv run python scripts/test_model_validation.py config/config_test_large.yaml   # ~3-4m
uv run python scripts/test_model_validation.py config/config_test_xlarge.yaml  # ~4-5m
uv run python scripts/compare_results.py                                       # tabular summary
```

Every `config_test_*.yaml` enables `training.split.strategy: stratified_random`, so both train and validation contain a consistent buyer mix (critical for the Stage 2 buyer-only regressor). Adjust `train_frac`, `model_(start|end)`, or estimator counts to trade off accuracy vs. runtime.

---

## End-to-End Pipeline

> The full workflow is orchestrated by `src/training/trainer.py::TrainingPipeline`. Each stage below references the concrete implementation files for low-level inspection.

### 0. Configuration ingestion

`TrainingPipeline.__init__` loads the YAML, instantiates:

- `DataLoader` (`src/data/loader.py`) – Dask-based parquet reader.
- `NestedFeatureParser` (`src/data/preprocessor.py`).
- `FeatureEngineer` (`src/features/engineer.py`).

### 1. Data loading & splitting (`prepare_data`)

1. **Dask read:** `DataLoader.load_train()` filters partitions by `data.train_*` / `data.val_*` or the combined modeling window when using the random strategy. Block size is auto-tuned to available RAM.
2. **Sampling:** `_apply_sampling` enforces `train_frac`, `val_frac`, and `max_*_partitions` per split before `.compute()`.
3. **Stratified random split (optional):** `_prepare_random_split_data` maps the combined Dask frame to pandas, then runs `train_test_split(..., stratify=buyer_d7)` with deterministic seeding.
4. **Fallback validation:** if the sampled validation set is empty, it draws `fallback_val_rows` from train to avoid training-time crashes.

### 2. Missingness & nested parsing (`src/data/preprocessor.py`)

`NestedFeatureParser.process_all` executes partition-safe transformations:

- Missing indicators via `MissingValueHandler` before altering the raw columns.
- **Dict columns:** aggregated stats (mean/std/max/min/count) for revenue/buy-count maps.
- **List columns:** lengths + uniqueness counts for bundle/app taxonomy lists.
- **Session tuples:** total/mean/max counts extracted from `(hash, count)` sequences.
- **Histograms:** top-K frequencies and entropy for country/region/device hist fields.
- **Timestamps:** recency features (`days_ago`, exponential decay weights).
- Final missing-value imputation optionally fitted on train and reused for val/test.

### 3. Feature engineering (`src/features/engineer.py`)

- **Interaction features:** whale × sessions, purchase × recency, install velocity, average purchase value, device-price ratios.
- **Local distribution features:** percentile ranks, local means, and deviations within `country`, `dev_os`, `advertiser_category` groups.
- **Temporal encoding:** cyclical sine/cosine for hour/weekday and coarse segments (morning/afternoon/evening/night, weekend flag).
- **Behavioral aggregates:** engagement score, session consistency, Wi-Fi engagement.
- Categorical encoding currently uses on-the-fly one-hot (extendable via `categorical_cols`).

Critical label columns (`buyer_d*`, `iap_revenue_d*`, `row_id`, `datetime`) are preserved throughout the engineer to keep downstream supervision intact.

### 4. Histogram-based sampling (`src/sampling/histos.py`)

- **HistOS (oversampling):** applied before Stage 1 training to rebalance rare buyers. It builds a histogram of target counts (buyers) and duplicates samples from under-populated bins until every bin reaches the `target_percentile` count.
- **HistUS (undersampling):** applied on Stage 2’s buyer-only dataset to de-emphasize zero-revenue dominance. It trims bins above the `target_percentile` while keeping the whale-rich tails intact.

### 5. Stage 1 – Buyer classifier (`src/models/buyer_classifier.py`)

- Model: LightGBM with AUC objective, limited depth (`max_depth` ≈ 5–8), and `n_estimators` tuned per config.
- Training: uses HistOS-resampled features, logs evaluation metrics, and persists to `models/buyer_classifier.txt`.
- Outputs: `predict_proba` probabilities used both for evaluation and as meta-features later.

### 6. Stage 2 – ODMN revenue regressor (`src/models/revenue_regressor.py`)

- Input subset: `buyer_d7 == 1` rows only (actual buyers).
- Three LightGBM regressors (`d1`, `d7`, `d14`) trained with RMSLE/MAE custom eval functions.
- Order enforcement: predictions pass through `RevenuePredictions.enforce_order_constraints()` to guarantee `d1 ≤ d7 ≤ d14`.
- HistUS keeps the training distribution focused on informative revenue bands; sample weights hook is available for further tuning.
- Horizon auto-selection: any horizon whose labels are missing (most often `d1`) is logged and skipped, so training never crashes when the column drops from certain slices.
- Objective auto-switch: even if configs still declare `objective: "huber"`, the trainer overrides it with LightGBM's `regression` objective because we now rely solely on the custom RMSLE/MAE callbacks for guidance.

### 7. Stage 3 – Stacking ensemble (`src/models/ensemble.py`)

- Base learners: ElasticNet, RandomForest, XGBoost.
- Meta-features: buyer probability, each horizon prediction, weighted horizon blend, and buyer × revenue interaction.
- Meta-learner: LightGBM optimized with a Huber objective to limit whale-induced outliers.
- Persisted to `models/stacking_ensemble.pkl`.

### 8. Training orchestration (`TrainingPipeline.run`)

1. `prepare_data()` → `train_df`, `val_df` (with logs and optional parquet caching).
2. `train_stage1_buyer()` – includes HistOS resampling and saves booster.
3. `train_stage2_revenue()` – filters buyers, applies HistUS, trains three horizons, saves boosters to `models/odmn_revenue_*.txt`.
4. `train_stage3_ensemble()` – builds meta-features from both train/val, fits the stacker, and saves to disk.

### 9. Validation & metrics (`scripts/test_model_validation.py`)

- Reuses `TrainingPipeline` but adds evaluation layers:
	- Stage 1 metrics: Accuracy/Precision/Recall/F1/ROC-AUC on train + validation.
	- Stage 2 metrics: RMSLE, RMSE, MAE, R² on buyer-only revenue.
	- Full pipeline: multiplies buyer probability by D7 revenue to get unconditional revenue and logs RMSLE/RMSE/R²/MAE.
- Prints dataset stats (rows, memory, buyer rate, revenue distribution) and prediction histograms.
- Logs stored at `logs/validation_<config>.log` for `scripts/compare_results.py` to summarize.

### 10. Inference & serving (`src/inference/predictor.py` + `scripts/predict.py`)

- `FastPredictor` reloads trained artefacts, runs the same preprocessing + feature engineering stack (with `fit=False`), and produces predictions for any dataframe.
- `predict_test_set()` loads the official test partitions via `DataLoader`, runs inference, and writes `data/submissions/submission.csv` with `row_id` + `iap_revenue_d7`.
- Latency stats (seconds + ms per sample) are logged for capacity planning.

### 11. Dashboard (`frontend/app.py`)

Streamlit UI for analysts:

- Load trained models, pick a dataset slice, and display buyer probabilities vs. revenue predictions.
- Visual diagnostics (histograms, scatter plots) can be extended via `frontend/components/`.

---

## Operational Commands

| Action | Command |
|--------|---------|
| Full training run | ```bash
uv run python scripts/train.py
``` |
| Validation harness (any config) | ```bash
uv run python scripts/test_model_validation.py config/config_test_medium.yaml
``` |
| Result comparison | ```bash
uv run python scripts/compare_results.py
``` |
| Test-set prediction | ```bash
uv run python scripts/predict.py
``` |
| Cached evaluation demo | ```bash
uv run python scripts/evaluate.py
``` |
| Streamlit dashboard | ```bash
uv run streamlit run frontend/app.py
``` |

Adjust the config path or CLI flags as needed. All scripts assume the repo root as the working directory.

---

## Testing & Quality Gates

- Pytest suite lives under `tests/` (column sanity, dataset structure, missing values, Dask optimizations, pipeline regressions). Run all tests via:

```bash
uv run pytest -q
```

- Add focused checks by targeting individual files (e.g., `uv run pytest tests/test_data_loader.py`).
- `src/utils/metrics.evaluate_predictions` centralizes RMSLE/RMSE/MAE/R² so unit tests and validation scripts stay aligned.

---

## Logging, Artifacts & Outputs

- **Logs:** `logs/*.log` (training, prediction, validation). They capture timestamps, dataset sizes, metric tables, and warnings (e.g., residual object columns after encoding).
- **Models:** saved under `models/` (`buyer_classifier.txt`, `odmn_revenue_{d1,d7,d14}.txt`, `stacking_ensemble.pkl`).
- **Processed data:** optional parquet caches `data/processed/train_processed.parquet` / `val_processed.parquet` when `training.cache_features=true`.
- **Submissions:** `data/submissions/submission.csv` from the latest inference run.

Artifacts are overwritten per run; archive `models/` or `logs/` externally if you need historical checkpoints.

---

## Troubleshooting & Tuning

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Validation RMSLE >> Train RMSLE | Distribution shift or too few buyers in val | Use `training.split.strategy: stratified_random`, widen `model_(start|end)`, or increase `train_frac`/`val_frac`. |
| Stage 2 cannot train (0 buyers) | Sampling window too small | Increase modeling window, lower HistUS target percentile, or temporarily disable filtering. |
| Memory errors during `.compute()` | Dask partitions too large | Reduce `train_frac`, lower `max_*_partitions`, or trim the datetime window. |
| Pandas `FutureWarning` about `groupby(..., observed=False)` | Upcoming pandas default change | The warning is safe; set `observed=True` inside `FeatureEngineer` if you want silence. |
| Inference output contains NaN | Object columns not encoded before inference | Check warning in `FastPredictor`; ensure preprocessing removed all string columns or extend encoder coverage. |

Additional tuning levers:

- LightGBM depth/leaves per config scale with data size (`config_test_small` uses 15 leaves, XLARGE bumps to 31).
- ODMN `loss.lambda_*` weights favor D7; tweak if business shifts toward D1 or D14 accuracy.
- HistOS/HistUS bin counts can be raised for smoother distributions when data volumes grow.

---

## Research Lineage

- **Order-preserving Deep Multitask Network (ODMN)** – Li et al., CIKM 2022. Provides the multi-horizon + order-constraint blueprint implemented in `src/models/revenue_regressor.py` and `src/models/losses.py`.
- **OptDist Sub-distribution Selection** – Weng et al., CIKM 2024. Inspires the buyer-focused resampling and horizon-specific weighting.
- **HistOS / HistUS** – Aminian et al., Machine Learning Journal 2025. Drives the histogram-aware over/under-sampling routines in `src/sampling/histos.py`.
- **Hybrid stacking ensembles for zero-inflated regression** – ScienceDirect Hybrid Ensemble Survey 2025. Basis for the Stage 3 blending strategy.

*Citation:* “We adapt the ODMN framework (Li et al., 2022) with OptDist sub-distribution selection (Weng et al., 2024) and extend it with HistOS sampling plus a stacking ensemble for Smadex 2025.”

---

This README walks the entire pipeline so ML engineers can navigate from raw parquet to final predictions. Dive into the referenced modules for implementation details or extend the configs/tests to suit new experiments.