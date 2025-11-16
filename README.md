# Smadex LTV Prediction

## 1. Project Overview

Smadex LTV Prediction estimates 7-day in-app purchase revenue for each ad impression. The architecture merges a LightGBM buyer classifier, histogram-aware sampling (HistOS/HHISTUS), an ODMN-inspired LightGBM regressor conditioned on buyers, and a stacking ensemble that blends calibrated probabilities with revenue forecasts. Buyer intent flows through gating logic that caps overconfident probabilities, while the revenue branch keeps predictions consistent with historical spend patterns. Meta-features splice these signals together with recency and whale diagnostics to deliver robust D7 revenue estimates suitable for either Kaggle-style submissions or real-time dashboards.

**How to use and test:** create a uv environment, install the project in editable mode, and execute `uv run python scripts/train.py --config config/config.yaml` to train all models, `uv run python scripts/predict.py --config config/config.yaml` to score the test partitions, and `uv run python scripts/evaluate_holdout.py --config config/config.yaml --split val` to report RMSLE/RMSE/MAE on labeled data; swapping in `config/config_test.yaml` or the chunked scripts keeps turnaround times manageable during development.

## 2. Data Processing

**Data loading & purging.** `src/data/loader.py` streams parquet partitions via Dask according to the windows set in `config/config.yaml`. Each split (train, validation, test) can apply fractional sampling and a maximum partition cap before materializing to pandas. The loader enforces schema alignment, drops rows lacking essential keys (`row_id`, revenue labels, buyer flags), and logs any unexpected columns so that downstream feature builders remain deterministic.

**Feature extraction.** `src/data/preprocessor.py` and `src/features/engineer.py` transform nested Smadex fields into flat numeric tensors. Dictionaries of country/device histograms become entropy scores, sparse session tuples become total and peak session counts, lists of bundle IDs become length/uniqueness statistics, and timestamps are recast into recency deltas plus cyclical encodings. Missing-value guards create binary indicators before filling numeric gaps, preserving leakage-free signals. The engineer then composes interaction terms (e.g., whale-friction × session depth, buyer probability × ARPU) and localized aggregates (per-country percentiles, deviation-from-geo-median) that feed every model stage.

**Histogram-aware resampling.** `src/sampling/histos.py` implements two complementary routines. HistOS oversamples rare buyers by equalizing histogram bins derived from the target rate, ensuring the classifier sees balanced evidence across spend brackets. HHISTUS (heavy-tail histogram undersampling) trims redundant zero-revenue buyers before Stage 2 so the ODMN regressors emphasize informative spenders while reserving room for whale preservation. Both steps honor partition boundaries to avoid leakage and can be tuned per split through the YAML configuration.

## 3. Training the Models

**Buyer classifier (`src/models/buyer_classifier.py`).** A LightGBM gradient-boosted tree predicts the probability that a user purchases within 7 days. HistOS-balanced data prevents collapse toward the majority negative class, calibrated probabilities are persisted for inference, and gating thresholds ensure that the zero-rate seen in validation matches the holdout requirement. This classifier’s outputs seed every subsequent component.

**High-value gate (`src/models/buyer_classifier.py` + `src/models/threshold_optimizer.py`).** A secondary LightGBM or calibrated thresholding module flags likely whales. Its role is to throttle the regression branch when the classifier yields low confidence while still allowing high-revenue anomalies to pass. The optimizer writes `buyer_threshold.json` / `buyer_gating.json`, which inference consumes to clamp or boost predictions.

**ODMN-inspired revenue regressor (`src/models/revenue_regressor.py`).** A LightGBM model conditioned on confirmed buyers predicts 7-day revenue. Custom RMSLE/MAE callbacks reward smooth targets, while regularization penalties deter overestimation. HHISTUS prunes flat zero-revenue sections so the regressor concentrates on informative spenders without sacrificing whale recall.

**Ensemble meta-learner (`src/models/ensemble.py`).** ElasticNet, RandomForest, and gradient boosting base learners feed a LightGBM stacker that predicts unconditional D7 revenue. The meta-feature set includes buyer probability, regressor output, whale gate signals, device/geo risk factors, and engineered caps that prevent any single component from dominating. The final prediction multiplies the buyer probability curve against the revenue expectation, optionally applying calibration for submission alignment.

**Inference orchestrator (`src/inference/predictor.py`).** FastPredictor reloads all fitted artefacts, replays preprocessing/feature engineering, and streams predictions in configurable chunks. The predictor honors gating constraints, applies the probability calibrator, executes the ensemble, and writes submission-ready CSVs.

## 4. Key Files

`config/config.yaml` -- Production configuration covering data windows, sampling fractions, model hyperparameters, and gating thresholds.

`config/config_test.yaml` -- Lightweight configuration with narrow date ranges and aggressive sampling for smoke tests.

`config/features.yaml` -- Canonical registry of feature groups toggled during preprocessing and engineering.

`scripts/train.py` -- CLI entrypoint that loads the config, runs the TrainingPipeline, and persists all model artefacts.

`scripts/predict.py` -- Batch inference script for standard RAM footprints; supports sampling overrides for experimentation.

`scripts/predict_chunked.py` -- Memory-optimized predictor that streams data through FastPredictor using adjustable chunk sizes.

`scripts/evaluate_holdout.py` -- Holdout scoring utility that reloads trained models and computes RMSLE/RMSE/MAE on labeled splits.

`src/data/loader.py` -- Dask-based parquet ingestion plus row-level purging logic and sampling enforcement.

`src/data/preprocessor.py` -- Nested-structure flattener, missing-value handler, and base feature constructor.

`src/features/engineer.py` -- Higher-order feature composer that adds interactions, localized aggregates, and temporal encodings.

`src/models/buyer_classifier.py` -- LightGBM classifier definition, calibration hooks, and whale gating utilities.

`src/models/revenue_regressor.py` -- ODMN-inspired buyer-conditioned regressor with custom RMSLE/MAE callbacks for 7-day revenue.

`src/models/ensemble.py` -- Stacking ensemble builder plus meta-feature generation used during training and inference.

`src/inference/predictor.py` -- FastPredictor class responsible for preprocessing, model chaining, and submission creation.

`src/training/trainer.py` -- TrainingPipeline implementation orchestrating data prep, model fitting, logging, and artefact storage.

`src/sampling/histos.py` -- HistOS oversampling and HHISTUS undersampling algorithms for buyer/revenue balancing.

`src/utils/metrics.py` -- Shared RMSLE/RMSE/MAE/R² computations invoked by training, evaluation, and tests.

## 5. Next Steps

1. **Tune model hyperparameters.** Start with `config/config_test.yaml` to probe LightGBM depth, learning rate, and regularization for both the buyer classifier and ODMN regressors. Once a candidate set emerges, copy it into `config/config.yaml` and repeat the training/predict/evaluate cycle to verify stability on the full window.
2. **Adjust zero-rate clamping.** Review `models/buyer_threshold.json` and `models/buyer_gating.json` after each training run to ensure predicted zero ratios align with the holdout target. If over-clamping suppresses whale recall, widen the `buyer_probability.cap` range or retrain the calibrator (`src/models/calibration.py`) using the latest validation curves.
3. **Reiterate on large datasets.** When satisfied with small-scale behavior, increase `training.sampling.frac`, `max_train_partitions`, and the datetime windows in `config/config.yaml` to cover the entire production horizon. Use `scripts/train.py`, `scripts/predict_chunked.py`, and `scripts/evaluate_holdout.py` sequentially, monitoring `logs/training.log` for chunk pressure; reduce `chunk_size` if memory spikes appear during this final verification pass.