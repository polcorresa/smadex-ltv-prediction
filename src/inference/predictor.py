"""
Fast inference engine for production deployment.

Following ArjanCodes best practices:
- Complete type hints
- Proper use of dataclasses
- Assertions for validation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import time
from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Optional

from src.data.loader import DataLoader
from src.data.preprocessor import NestedFeatureParser
from src.features.engineer import FeatureEngineer
from src.models.buyer_classifier import BuyerClassifier
from src.models.revenue_regressor import ODMNRevenueRegressor
from src.models.ensemble import StackingEnsemble, build_meta_features
from src.types import RevenuePredictions

logger = logging.getLogger(__name__)


class FastPredictor:
    """Optimized inference pipeline."""
    
    def __init__(self, config_path: str) -> None:
        """
        Load trained models.
        
        Args:
            config_path: Path to configuration file
        """
        assert config_path is not None, "Config path must not be None"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        assert self.config is not None, "Config must not be None"
        
        # Initialize components
        self.preprocessor = NestedFeatureParser()
        self.feature_engineer = FeatureEngineer(self.config)
        
        # Load models
        logger.info("Loading trained models...")
        
        self.buyer_model = BuyerClassifier(self.config)
        self.buyer_model.load('models/buyer_classifier.txt')
        
        self.revenue_model = ODMNRevenueRegressor(self.config)
        self.revenue_model.load('models/odmn')
        
        self.ensemble_model = StackingEnsemble(self.config)
        self.ensemble_model.load('models/stacking_ensemble.pkl')
        
        # Load optimal buyer threshold
        self.optimal_buyer_threshold = self._load_buyer_threshold()
        
        logger.info("Models loaded successfully")

    def _scale_probabilities(self, buyer_proba: np.ndarray) -> np.ndarray:
        """Apply calibrated scaling using the saved optimal threshold."""
        alpha = float(self.optimal_buyer_threshold or 1.0)
        buyer_proba = np.clip(buyer_proba, 0.0, 1.0)
        if alpha <= 0.0 or np.isclose(alpha, 1.0):
            return buyer_proba
        return buyer_proba ** alpha

    def _gate_revenue_predictions(
        self,
        buyer_proba: np.ndarray,
        predictions: RevenuePredictions
    ) -> RevenuePredictions:
        """Apply buyer-probability gating to revenue predictions when enabled."""
        gating_cfg = (
            self.config
            .get('models', {})
            .get('stage2_revenue', {})
            .get('gating', {})
        )
        if not gating_cfg or not gating_cfg.get('enabled', False):
            return predictions

        alpha = float(gating_cfg.get('alpha', 1.0))
        floor = float(gating_cfg.get('floor', 0.0))
        prob_cap = float(gating_cfg.get('probability_cap', 1.0))
        prob_cutoff = float(gating_cfg.get('probability_cutoff', 0.0))
        logger.debug(
            "Applying revenue gating during inference: alpha=%.3f, floor=%.3f, cap=%.3f, cutoff=%.3f",
            alpha,
            floor,
            prob_cap,
            prob_cutoff
        )
        return predictions.gate_with_probability(
            buyer_proba,
            power=alpha,
            floor=floor,
            probability_cap=prob_cap,
            probability_cutoff=prob_cutoff
        )
    
    def _load_buyer_threshold(self) -> float:
        """Load optimized buyer threshold, fallback to 1.0 if not found."""
        import json
        threshold_path = Path('models/buyer_threshold.json')
        
        if threshold_path.exists():
            try:
                with open(threshold_path, 'r') as f:
                    data = json.load(f)
                threshold = float(data.get('optimal_threshold', 1.0))
                logger.info(f"Loaded optimal buyer threshold: {threshold:.4f}")
                return threshold
            except Exception as e:
                logger.warning(f"Failed to load threshold from {threshold_path}: {e}")
                return 1.0
        else:
            logger.info("No threshold file found; using default alpha=1.0 (no scaling)")
            return 1.0
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fast inference on test data.
        
        Args:
            df: Raw test data
        
        Returns:
            Array of predicted iap_revenue_d7
        """
        assert df is not None, "DataFrame must not be None"
        assert len(df) > 0, "DataFrame must not be empty"
        
        # Preprocess
        df = self.preprocessor.process_all(df)
        
        # Feature engineering (fit=False to use cached encoders)
        # Note: pass any available target column or None for test data
        target_col = 'iap_revenue_d7' if 'iap_revenue_d7' in df.columns else None
        df = self.feature_engineer.engineer_all(df, target_col=target_col, fit=False)
        
        # Select features (exclude metadata and targets)
        excluded_cols = {'row_id', 'datetime', 'iap_revenue_d7', 'iap_revenue_d1', 
                        'iap_revenue_d14', 'iap_revenue_d28', 'buyer_d1', 'buyer_d7', 
                        'buyer_d14', 'buyer_d28'}
        feature_cols = [col for col in df.columns if col not in excluded_cols]
        
        # Use only numeric columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # Check for remaining object columns
        obj_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
        if obj_cols:
            logger.warning(f"Found {len(obj_cols)} object columns during inference, dropping them: {obj_cols[:5]}")
        
        X = df[numeric_cols].fillna(0)
        
        assert len(X.columns) > 0, "Must have at least one feature column"
        
        # Stage 1: Buyer probability + calibration
        buyer_proba = self.buyer_model.predict_proba(X)
        scaled_buyer_proba = self._scale_probabilities(buyer_proba)
        
        # Stage 2: Revenue predictions
        revenue_preds = self.revenue_model.predict(X, enforce_order=True)
        revenue_preds = self._gate_revenue_predictions(scaled_buyer_proba, revenue_preds)
        
        # Stage 3: Ensemble meta-features (use scaled probabilities)
        loss_config = self.config['models']['stage2_revenue']['loss']
        gating_cfg = (
            self.config
            .get('models', {})
            .get('stage2_revenue', {})
            .get('gating', {})
        )
        X_meta = build_meta_features(
            scaled_buyer_proba,
            revenue_preds,
            loss_config,
            gating_cfg
        )
        
        # Final prediction using ensemble
        final_preds = self.ensemble_model.predict(X_meta)
        cutoff = 0.0
        if gating_cfg:
            cutoff = float(gating_cfg.get('probability_cutoff', 0.0))
        final_preds = np.where(scaled_buyer_proba <= cutoff, 0.0, final_preds)
        
        logger.debug(
            f"Ensemble predictions: mean=${final_preds.mean():.2f}, std=${final_preds.std():.2f}"
        )
        
        # Postcondition
        assert len(final_preds) == len(df), \
            "Predictions length must match input length"
        assert np.all(final_preds >= 0), \
            "Predictions must be non-negative"
        
        return final_preds
    
    def predict_test_set(
        self,
        *,
        max_partitions: Optional[int] = None,
        sample_frac: Optional[float] = None,
        limit_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Predict on official test set.
        
        Args:
            max_partitions: Optional cap on number of Dask partitions loaded from disk.
            sample_frac: Optional fraction (0, 1] to down-sample the computed test dataframe.
            limit_rows: Optional hard cap on number of rows kept after sampling.

        Returns:
            Submission dataframe with row_id and iap_revenue_d7
        """
        inference_cfg = self.config.get('inference', {})
        if max_partitions is None:
            max_partitions = inference_cfg.get('max_partitions')
        if sample_frac is None:
            sample_frac = inference_cfg.get('sample_frac')
        if limit_rows is None:
            limit_rows = inference_cfg.get('limit_rows')

        logger.info(
            "Loading test data (max_partitions=%s, sample_frac=%s, limit_rows=%s)...",
            max_partitions,
            sample_frac,
            limit_rows
        )
        
        data_loader = DataLoader(self.config)
        test_ddf = data_loader.load_test(max_partitions=max_partitions)
        test_df = test_ddf.compute()
        
        assert len(test_df) > 0, "Test data must not be empty"

        if sample_frac is not None:
            assert 0.0 < sample_frac <= 1.0, "sample_frac must be in (0, 1]"
            if sample_frac < 1.0:
                random_state = self.config.get('training', {}).get('random_state', 42)
                logger.info("Sampling test data with frac=%.3f (random_state=%s)", sample_frac, random_state)
                test_df = test_df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
                logger.info("Sampled test size: %d", len(test_df))

        if limit_rows is not None:
            assert limit_rows > 0, "limit_rows must be positive"
            if len(test_df) > limit_rows:
                logger.info(
                    "Limiting test dataframe from %d to first %d rows for quicker inference",
                    len(test_df),
                    limit_rows
                )
                test_df = test_df.head(limit_rows).reset_index(drop=True)
        
        logger.info(f"Test size: {len(test_df)}")
        
        # Predict
        start_time = time.time()
        predictions = self.predict(test_df)
        elapsed = time.time() - start_time
        
        logger.info(f"Prediction completed in {elapsed:.2f}s")
        logger.info(f"Average latency: {elapsed / len(test_df) * 1000:.2f}ms per sample")
        
        # Create submission
        submission = pd.DataFrame({
            'row_id': test_df['row_id'],
            'iap_revenue_d7': predictions
        })
        
        # Postcondition
        assert len(submission) == len(test_df), \
            "Submission length must match test data"
        assert 'row_id' in submission.columns, \
            "Submission must contain row_id column"
        assert 'iap_revenue_d7' in submission.columns, \
            "Submission must contain iap_revenue_d7 column"
        
        return submission