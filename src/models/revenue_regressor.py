"""
Stage 2: Multi-Horizon Revenue Regression (ODMN).

Predicts [iap_revenue_d1, iap_revenue_d7, iap_revenue_d14] jointly.

Following ArjanCodes best practices:
- Complete type hints
- Precondition/postcondition assertions
- Dataclass for structured results
- Enum for time horizons
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Any, Optional, Tuple
import logging

from pathlib import Path

import json

from src.types import TimeHorizon, RevenuePredictions

logger = logging.getLogger(__name__)


def rmsle_feval(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
    """
    Custom RMSLE evaluation function for LightGBM sklearn API.
    
    Note: For sklearn API, the signature is (y_true, y_pred), not (y_pred, data)
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Tuple of (metric_name, metric_value, is_higher_better)
    """
    # Ensure non-negative predictions
    y_pred = np.clip(y_pred, 0, None)
    
    # Calculate RMSLE
    rmsle = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
    
    return ('rmsle', rmsle, False)  # False = lower is better


def mae_feval(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
    """
    Custom MAE evaluation function.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Tuple of (metric_name, metric_value, is_higher_better)
    """
    y_pred = np.clip(y_pred, 0, None)
    mae = np.mean(np.abs(y_true - y_pred))
    return ('mae', mae, False)


class ODMNRevenueRegressor:
    """
    Order-preserving multi-task revenue regressor.
    
    Based on Kuaishou ODMN (CIKM 2022).
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize ODMN regressor.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        assert config is not None, "Config must not be None"
        assert 'models' in config, "Config must contain 'models' key"
        assert 'stage2_revenue' in config['models'], \
            "Config must contain stage2_revenue settings"
        
        self.config = config
        self.models: Dict[TimeHorizon, lgb.LGBMRegressor | lgb.Booster] = {}
        self.feature_names: Optional[list[str]] = None
        raw_horizons = config['models']['stage2_revenue'].get('horizons', [1, 7, 14])
        self.horizons: list[TimeHorizon] = self._resolve_horizons(raw_horizons)
        self.trained_horizons: list[TimeHorizon] = []
        
        assert len(self.horizons) > 0, "Must have at least one horizon"
    
    def train(
        self,
        X_train: pd.DataFrame,
        targets_train: Dict[TimeHorizon, np.ndarray],
        X_val: pd.DataFrame,
        targets_val: Dict[TimeHorizon, np.ndarray],
        sample_weight: Optional[np.ndarray] = None
    ) -> None:
        """
        Train multi-horizon revenue models.
        
        Note: For simplicity, we train separate models per horizon
        with post-processing order enforcement. For full ODMN,
        would need custom objective function.
        
        Args:
            X_train: Training features
            y_train_d1: D1 revenue targets
            y_train_d7: D7 revenue targets
            y_train_d14: D14 revenue targets
            X_val: Validation features
            y_val_d1: D1 validation targets
            y_val_d7: D7 validation targets
            y_val_d14: D14 validation targets
            sample_weight: Optional sample weights
        """
        # Preconditions
        assert len(X_train) > 0, "Training data must not be empty"
        assert len(targets_train) > 0, "Need at least one revenue horizon to train"
        assert len(targets_val) > 0, "Need at least one revenue horizon for validation"
        train_lengths = {len(arr) for arr in targets_train.values()}
        val_lengths = {len(arr) for arr in targets_val.values()}
        assert train_lengths == {len(X_train)}, "Training targets must align with X_train"
        assert val_lengths == {len(X_val)}, "Validation targets must align with X_val"
        for horizon, arr in targets_train.items():
            assert np.all(arr >= 0), f"{horizon.display_name()} training targets must be non-negative"
        for horizon, arr in targets_val.items():
            assert np.all(arr >= 0), f"{horizon.display_name()} validation targets must be non-negative"
        
        if sample_weight is not None:
            assert len(sample_weight) == len(X_train), \
                "Sample weights must match training size"
            assert np.all(sample_weight >= 0), "Sample weights must be non-negative"
        
        logger.info("Training Stage 2: Revenue Regressor (Multi-Horizon)")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Get model config
        model_config = self.config['models']['stage2_revenue']['params'].copy()
        
        # Train models for each horizon
        available_horizons = [
            horizon for horizon in self.horizons
            if horizon in targets_train and horizon in targets_val
        ]
        if not available_horizons:
            raise ValueError("No overlapping horizons found between training and validation sets")

        self.trained_horizons = available_horizons
        
        for horizon in available_horizons:
            y_train = targets_train[horizon]
            y_val = targets_val[horizon]
            logger.info(f"Training model for {horizon.value}...")
            
            # Use 'regression' objective which works better for RMSLE
            # We'll use custom feval to monitor RMSLE during training
            model_config_copy = model_config.copy()
            if model_config_copy.get('objective') == 'huber':
                logger.info(f"  ⚠️  Changing objective from 'huber' to 'regression' for RMSLE optimization")
                model_config_copy['objective'] = 'regression'
            
            # Use standard regression without complex sample weighting
            # The full pipeline performance is what matters, not Stage 2 in isolation
            logger.info("  Using standard regression objective")
            
            model = lgb.LGBMRegressor(**model_config_copy)
            
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=[rmsle_feval, mae_feval],  # Monitor both RMSLE and MAE
                sample_weight=sample_weight,  # Use provided weights (from HistUS)
                callbacks=[
                    lgb.early_stopping(model_config_copy.get('early_stopping_rounds', 100)),
                    lgb.log_evaluation(50)  # Log every 50 iterations
                ]
            )
            
            self.models[horizon] = model
            
            # Validation metrics
            y_val_pred = np.clip(model.predict(X_val), 0, None)  # Ensure non-negative
            rmse = self._rmse(y_val, y_val_pred)
            mae = mean_absolute_error(y_val, y_val_pred)
            rmsle_val = np.sqrt(np.mean((np.log1p(y_val) - np.log1p(y_val_pred)) ** 2))
            
            # Additional diagnostics
            mean_actual = y_val.mean()
            mean_pred = y_val_pred.mean()
            median_actual = np.median(y_val)
            median_pred = np.median(y_val_pred)
            
            logger.info(f"{horizon.value} - Validation Metrics:")
            logger.info(f"  RMSLE: {rmsle_val:.6f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            logger.info(f"  Mean: actual=${mean_actual:.2f}, pred=${mean_pred:.2f}")
            logger.info(f"  Median: actual=${median_actual:.2f}, pred=${median_pred:.2f}")
    
    def predict(
        self, 
        X: pd.DataFrame, 
        enforce_order: bool = True
    ) -> RevenuePredictions:
        """
        Predict revenue for all horizons.
        
        Args:
            X: Features
            enforce_order: Apply order-preserving post-processing
        
        Returns:
            RevenuePredictions dataclass with d1, d7, d14 arrays
        """
        assert len(self.models) > 0, "Models not trained yet"
        assert len(X) > 0, "Input data must not be empty"
        
        if self.feature_names is not None:
            missing = [col for col in self.feature_names if col not in X.columns]
            if missing:
                logger.warning(
                    "Revenue regressor missing %d feature columns (examples: %s); filling with zeros.",
                    len(missing),
                    missing[:5]
                )
                X = X.copy()
                for col in missing:
                    X[col] = 0.0
            X = X[self.feature_names]
        else:
            logger.warning("Revenue regressor lacks feature metadata; using raw input columns (%d).", len(X.columns))
        
        predictions_dict: Dict[TimeHorizon, np.ndarray] = {}
        
        horizons_to_use = self.trained_horizons or list(self.models.keys())
        for horizon in horizons_to_use:
            model = self.models[horizon]
            predictions_dict[horizon] = np.clip(model.predict(X), 0, None)
        
        # Create RevenuePredictions object
        predictions = RevenuePredictions(predictions_dict)
        
        # Enforce order constraints: ensure non-decreasing horizons when requested
        if enforce_order:
            predictions = predictions.enforce_order_constraints()
        
        # Postcondition handled inside RevenuePredictions
        return predictions
    
    def _enforce_order_constraints(
        self, 
        predictions: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Post-processing to ensure D1 ≤ D7 ≤ D14.
        
        Uses isotonic regression approach.
        
        Args:
            predictions: Dictionary with horizon predictions
            
        Returns:
            Dictionary with corrected predictions
        """
        d1 = predictions[TimeHorizon.D1.value]
        d7 = predictions[TimeHorizon.D7.value]
        d14 = predictions[TimeHorizon.D14.value]
        
        # Clip D7 to be between D1 and D14
        d7_corrected = np.clip(d7, d1, d14)
        
        # If D1 > D7, adjust D1
        d1_corrected = np.minimum(d1, d7_corrected)
        
        # If D7 > D14, adjust D14
        d14_corrected = np.maximum(d14, d7_corrected)
        
        return {
            TimeHorizon.D1.value: d1_corrected,
            TimeHorizon.D7.value: d7_corrected,
            TimeHorizon.D14.value: d14_corrected
        }
    
    def save(self, base_path: str) -> None:
        """
        Save all models.
        
        Args:
            base_path: Base path for model files
        """
        assert len(self.models) > 0, "No models to save"
        
        saved_horizons: list[TimeHorizon] = []
        for horizon, model in self.models.items():
            path = f"{base_path}_revenue_{horizon.value}.txt"
            model.booster_.save_model(path)
            logger.info(f"Revenue model {horizon.value} saved to {path}")
            saved_horizons.append(horizon)

        metadata_path = Path(f"{base_path}_revenue_metadata.json")
        metadata = {
            "horizons": [horizon.value for horizon in saved_horizons],
            "feature_names": self.feature_names,
            "generated_at": pd.Timestamp.utcnow().isoformat()
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
        logger.info("Revenue metadata written to %s", metadata_path)
    
    def load(self, base_path: str) -> None:
        """
        Load all models.
        
        Args:
            base_path: Base path for model files
        """
        metadata = self._load_metadata(base_path)
        metadata_horizons = metadata.get("horizons") if metadata else None
        horizons_to_load = self._resolve_horizons(metadata_horizons) if metadata_horizons else self.horizons

        for horizon in horizons_to_load:
            path = Path(f"{base_path}_revenue_{horizon.value}.txt")
            if not path.exists():
                logger.warning(
                    "Revenue model file %s not found; skipping horizon %s",
                    path,
                    horizon.value
                )
                continue
            self.models[horizon] = lgb.Booster(model_file=str(path))
            logger.info(f"Revenue model {horizon.value} loaded from {path}")
        
        self.trained_horizons = list(self.models.keys())
        if metadata and metadata.get("feature_names"):
            self.feature_names = metadata["feature_names"]
            logger.info("Loaded %d revenue feature columns from metadata", len(self.feature_names))
        elif not self.feature_names:
            logger.warning("Revenue feature metadata missing; inference will use raw columns from input data.")
        assert self.models, "No revenue models were loaded"

    def _load_metadata(self, base_path: str) -> dict[str, Any] | None:
        """Load saved metadata if available (horizons + feature names)."""
        metadata_path = Path(f"{base_path}_revenue_metadata.json")
        if not metadata_path.exists():
            legacy_path = Path(f"{base_path}_revenue_horizons.json")
            if legacy_path.exists():
                logger.info("Using legacy horizon metadata at %s", legacy_path)
                try:
                    return json.loads(legacy_path.read_text())
                except json.JSONDecodeError:
                    logger.warning("Unable to parse legacy metadata at %s", legacy_path)
            else:
                logger.info("No revenue metadata found near %s", metadata_path)
            return None

        try:
            return json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("Corrupted revenue metadata at %s (%s)", metadata_path, exc)
            return None

    @staticmethod
    def _resolve_horizons(raw_horizons: list[Any]) -> list[TimeHorizon]:
        """Normalize stage 2 horizon configuration."""
        normalized: list[TimeHorizon] = []
        mapping = {
            1: TimeHorizon.D1,
            7: TimeHorizon.D7,
            14: TimeHorizon.D14,
            28: TimeHorizon.D28
        }
        for value in raw_horizons:
            if isinstance(value, TimeHorizon):
                normalized.append(value)
                continue
            if isinstance(value, str):
                key = value.strip().lower()
                if key.startswith('d') and key[1:].isdigit():
                    value = int(key[1:])
                elif key.isdigit():
                    value = int(key)
                else:
                    raise ValueError(f"Unsupported horizon specifier: {value}")
            if isinstance(value, int):
                mapped = mapping.get(value)
                if mapped is None:
                    raise ValueError(f"Unsupported horizon: {value}")
                normalized.append(mapped)
            else:
                raise ValueError(f"Unsupported horizon specifier type: {type(value)}")

        unique_horizons: list[TimeHorizon] = []
        for horizon in sorted(normalized, key=lambda h: h.to_days()):
            if horizon not in unique_horizons:
                unique_horizons.append(horizon)
        return unique_horizons

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute RMSE with backward compatibility for older sklearn versions."""
        try:
            return mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))