# Smadex LTV Prediction System

Production-ready pipeline for forecasting 7-day in-app purchase revenue (`iap_revenue_d7`) for Smadex mobile app users. The stack combines advanced preprocessing, histogram-aware sampling (HistOS/HistUS), an ODMN-inspired multi-horizon regressor, and a hybrid stacking ensemble for final calibration.

> **Key ideas:** two-stage modeling of buyer intent and revenue, order-preserving multi-horizon losses, and lightweight inference that can serve millisecond predictions or drive a Streamlit dashboard.

---

## Why this project matters

- **Business objective** – identify buyers and estimate their 7-day revenue to unlock smarter UA bidding and LTV multipliers.
- **Modeling requirements** – cope with zero-inflated targets, whale users, and strict order constraints across 1/7/14-day horizons.
- **Operational constraints** – training on tens of millions of impressions using Dask + parquet, fast offline batch scoring, and a friendly analytics frontend.

---

## Stack compliance & research lineage

- **Primary framework:** The Stage‑2 regressor is built on the Kuaishou **Order-preserving Deep Multitask Network (ODMN)** described by Li et al., CIKM 2022. We retain the multi-horizon order constraints (D1 ≤ D7 ≤ D14) and ODMN-inspired losses in `src/models/revenue_regressor.py` and `src/models/losses.py`.
- **OptDist sub-distribution selection:** Buyer-focused resampling and horizon-specific weighting follow the OptDist strategy from Weng et al., CIKM 2024, emphasizing sub-distribution targeting for high-value cohorts.
- **2025 enhancements:** We layer **HistOS/HistUS histogram-aware sampling** (Aminian et al., MLJ 2025) and a **Hybrid stacking ensemble** (ScienceDirect Hybrid Ensemble, 2025) to stabilize zero-inflated targets and calibrate final predictions. These appear in `src/sampling/histos.py`, `src/training/trainer.py`, and `src/models/ensemble.py`.
- **Citation:** *“We adapt the ODMN framework (Li et al., CIKM 2022) with OptDist sub-distribution selection (Weng et al., CIKM 2024) and extend it with HistOS sampling plus a stacking ensemble for Smadex 2025.”*

---

## Quick start

### 1. Environment setup (uv recommended)

```bash
# install uv once (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# create & activate a virtualenv managed by uv
uv venv
source .venv/bin/activate

# install project dependencies from pyproject.toml
uv pip install -e .
```

> Prefer pip? Replace the last two commands with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train the full pipeline

```bash
python scripts/train.py
```

**Low-memory tip:** Set `training.sampling.train_frac` / `val_frac` or `max_*_partitions` in `config/config.yaml` (e.g., `train_frac: 0.2`) to downsample the Dask DataFrame before calling `.compute()`. This keeps RAM usage manageable on 8 GB laptops while you debug.

### 3. (Optional) Evaluate cached validation data

```bash
python scripts/evaluate.py
```

### 4. Generate test predictions / submission

```bash
python scripts/predict.py
```

### 5. Launch the Streamlit dashboard

```bash
streamlit run frontend/app.py
```

All paths, sampling knobs, and model hyperparameters are controlled through `config/config.yaml`. Processed parquet artefacts are cached under `data/processed/` when `training.cache_features` is enabled.

---

## Modeling pipeline overview

| Stage | Purpose | Key files |
|-------|---------|-----------|
| Data ingestion & splits | Read raw parquet with temporal filters for train/val/test | `src/data/loader.py`, `config/config.yaml`
| Nested feature parsing | Flatten dict/list/timestamp fields into dense numeric aggregates | `src/data/preprocessor.py`
| Feature engineering | Interaction, local percentile, categorical encoding, temporal cycles | `src/features/engineer.py`, `config/features.yaml`
| Imbalance handling | HistOS oversampling for buyers, HistUS undersampling for rare-spend regression | `src/sampling/histos.py`
| Stage 1 – Buyer classifier | LightGBM classifier + HistOS samples predictive of `buyer_d7` | `src/models/buyer_classifier.py`
| Stage 2 – Revenue regressor | ODMN-style multi-horizon LightGBM (D1/D7/D14) + order constraints + custom losses | `src/models/revenue_regressor.py`, `src/models/losses.py`
| Stage 3 – Stacking ensemble | ElasticNet + RF + XGBoost stacked into LightGBM meta learner | `src/models/ensemble.py`
| Training orchestrator | Executes the 3 stages, caches artefacts, logs metrics | `src/training/trainer.py`, `scripts/train.py`
| Inference & serving | Fast predictor for batch/test scoring and dashboard usage | `src/inference/predictor.py`, `scripts/predict.py`, `frontend/app.py`

**Pipeline flow:**

1. **Load & split** parquet data with Dask (Oct 1–6 train, Oct 7 validation, Oct 8–12 test).
2. **Parse nested fields** (dicts/lists/histograms/timestamps) into scalars, then engineer behavior, purchase, contextual, temporal and whale features.
3. **Stage 1 – Buyer intent:** LightGBM classifier trained on HistOS-resampled positives outputs `P(buyer_d7)`.
4. **Stage 2 – Revenue:** Filter to predicted buyers, apply HistUS, train three LightGBM regressors (D1/D7/D14) with order-preserving post-processing (ODMN-inspired).
5. **Stage 3 – Stacking:** Combine buyer probabilities and multi-horizon revenues into meta-features; stack ElasticNet, RandomForest, XGBoost under a LightGBM meta-learner optimized for Huber/MSLE.
6. **Inference & serving:** Reproduce feature steps, compute meta-features, run ensemble, and expose predictions through CLI scripts or Streamlit dashboard.

---

## Configuration & data locations

- `config/config.yaml` – single source of truth for paths, splits, sampler knobs, and model hyperparameters.
- `config/features.yaml` – documentation of engineered feature groups, ablation slices, and selection heuristics.
- `data/raw/` – expected parquet inputs (not tracked). Organize subfolders for `train/` and `test/` matching config paths.
- `data/processed/` – cached feature matrices from `TrainingPipeline.prepare_data()`.
- `data/submissions/` – outputs from `scripts/predict.py`.

Update `config/config.yaml` when data paths change or when experimenting with hyperparameters. Feature toggles (e.g., removal thresholds) live under `features.yaml`.

---

## Repository guide (file-by-file)

### Root files

| Path | Description |
|------|-------------|
| `README.md` | You are here – full project overview and file reference. |
| `requirements.txt` | Python dependencies (LightGBM, XGBoost, Dask, Streamlit, etc.). |
| `setup.py` | Packaging hook for installing `src/` as `smadex-ltv-prediction`. |

### `config/`

| Path | Description |
|------|-------------|
| `config/config.yaml` | Global configuration (paths, temporal splits, sampler parameters, model hyperparams, training flags). |
| `config/features.yaml` | Detailed catalog of engineered features, grouped by behavior/purchase/context, plus selection strategies. |

### `data/`

| Path | Description |
|------|-------------|
| `data/raw/` | Placeholder for parquet training/testing data (not tracked). |
| `data/processed/` | Cached, fully engineered datasets written during training. |
| `data/submissions/` | Generated CSV submissions from inference. |

### `frontend/`

| Path | Description |
|------|-------------|
| `frontend/app.py` | Streamlit dashboard for loading trained models, generating predictions, and visualizing metrics/distributions. |
| `frontend/components/` | Slot for custom Streamlit components (empty scaffold for future charts/widgets). |

### `notebooks/`

| Path | Description |
|------|-------------|
| `notebooks/eda.ipynb` | Exploratory data analysis notebook covering distributional insights, whales, correlations, and feature ideas. |

### `scripts/`

| Path | Description |
|------|-------------|
| `scripts/train.py` | CLI entry point that instantiates `TrainingPipeline` and runs the full training workflow. |
| `scripts/predict.py` | Loads trained artefacts via `FastPredictor`, scores the official test set, and writes `data/submissions/submission.csv`. |
| `scripts/evaluate.py` | Utility to compute metrics on cached validation data (baseline vs. model predictions). |

### `src/data/`

| Path | Description |
|------|-------------|
| `src/data/loader.py` | Dask-powered parquet loader with temporal filters and batch computation helpers. |
| `src/data/preprocessor.py` | `NestedFeatureParser` that flattens dict/list/timestamp/histogram fields into numeric aggregates. |

### `src/features/`

| Path | Description |
|------|-------------|
| `src/features/engineer.py` | `FeatureEngineer` composing interaction, local distribution, categorical encoding, temporal, and behavioral aggregations. |
| `src/features/selector.py` | Toolkit for variance/correlation filtering, model-based importance, and RFE/SHAP-style selection. |

### `src/sampling/`

| Path | Description |
|------|-------------|
| `src/sampling/histos.py` | HistOS oversampling + HistUS undersampling implementations tailored for imbalanced regression/classification. |

### `src/models/`

| Path | Description |
|------|-------------|
| `src/models/buyer_classifier.py` | LightGBM-based Stage‑1 binary classifier for `buyer_d7`. |
| `src/models/revenue_regressor.py` | ODMN-inspired multi-horizon LightGBM regressor with order-preserving post-processing. |
| `src/models/losses.py` | Custom PyTorch/LightGBM losses enforcing Huber robustness and order constraints. |
| `src/models/ensemble.py` | Stacking ensemble that blends ElasticNet, RandomForest, XGBoost, and a LightGBM meta-learner. |

### `src/inference/`

| Path | Description |
|------|-------------|
| `src/inference/predictor.py` | `FastPredictor` – end-to-end inference engine (preprocess ⇒ feature engineering ⇒ models ⇒ ensemble). |

### `src/training/`

| Path | Description |
|------|-------------|
| `src/training/trainer.py` | `TrainingPipeline` orchestrating data prep, stage-wise training, sampling, caching, and model persistence. |

### `src/utils/`

| Path | Description |
|------|-------------|
| `src/utils/logger.py` | Consistent logging setup (console + file handlers). |
| `src/utils/metrics.py` | MSLE/RMSLE/MAE/R2 helper functions for evaluation and monitoring. |

---

## Extending the system

- **Alternative objectives** – plug custom ODMN losses from `src/models/losses.py` into LightGBM or neural backbones.
- **Feature ablations** – adjust `config/features.yaml` groups for quick experiments or run the `FeatureSelector` utilities.
- **Serving** – wrap `FastPredictor` inside an API, or continue using `frontend/app.py` for analyst-friendly exploration.

For questions, new experiments, or deployment hooks, start from the references above – every document in the repo ties back to one of the pipeline stages described here.