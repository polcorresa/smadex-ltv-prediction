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
        self.models: Dict[str, lgb.LGBMRegressor] = {}
        self.feature_names: Optional[list[str]] = None
        self.horizons: list[str] = config['models']['stage2_revenue']['horizons']
        
        assert len(self.horizons) > 0, "Must have at least one horizon"
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train_d1: np.ndarray,
        y_train_d7: np.ndarray,
        y_train_d14: np.ndarray,
        X_val: pd.DataFrame,
        y_val_d1: np.ndarray,
        y_val_d7: np.ndarray,
        y_val_d14: np.ndarray,
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
        assert len(X_train) == len(y_train_d1) == len(y_train_d7) == len(y_train_d14), \
            "All training arrays must have same length"
        assert len(X_val) == len(y_val_d1) == len(y_val_d7) == len(y_val_d14), \
            "All validation arrays must have same length"
        assert np.all(y_train_d1 >= 0), "D1 targets must be non-negative"
        assert np.all(y_train_d7 >= 0), "D7 targets must be non-negative"
        assert np.all(y_train_d14 >= 0), "D14 targets must be non-negative"
        
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
        horizons_data = {
            'd1': (y_train_d1, y_val_d1),
            'd7': (y_train_d7, y_val_d7),
            'd14': (y_train_d14, y_val_d14)
        }
        
        for horizon, (y_train, y_val) in horizons_data.items():
            logger.info(f"Training model for {horizon}...")
            
            # Use 'regression' objective which works better for RMSLE
            # We'll use custom feval to monitor RMSLE during training
            model_config_copy = model_config.copy()
            if model_config_copy.get('objective') == 'huber':
                logger.info(f"  ⚠️  Changing objective from 'huber' to 'regression' for RMSLE optimization")
                model_config_copy['objective'] = 'regression'
            
            # Use standard regression without complex sample weighting
            # The full pipeline performance is what matters, not Stage 2 in isolation
            logger.info(f"  Using standard regression objective")
            
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
            
            logger.info(f"{horizon} - Validation Metrics:")
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
        
        predictions_dict: Dict[str, np.ndarray] = {}
        
        for horizon, model in self.models.items():
            # Clip predictions to ensure non-negative values
            predictions_dict[horizon] = np.clip(model.predict(X), 0, None)
        
        # Create RevenuePredictions object
        predictions = RevenuePredictions(
            d1=predictions_dict[TimeHorizon.D1.value],
            d7=predictions_dict[TimeHorizon.D7.value],
            d14=predictions_dict[TimeHorizon.D14.value]
        )
        
        # Enforce order constraints: D1 ≤ D7 ≤ D14
        if enforce_order:
            predictions = predictions.enforce_order_constraints()
        
        # Postcondition
        assert np.all(predictions.d1 >= 0), "D1 predictions must be non-negative"
        assert np.all(predictions.d7 >= 0), "D7 predictions must be non-negative"
        assert np.all(predictions.d14 >= 0), "D14 predictions must be non-negative"
        
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
        
        for horizon, model in self.models.items():
            path = f"{base_path}_revenue_{horizon}.txt"
            model.booster_.save_model(path)
            logger.info(f"Revenue model {horizon} saved to {path}")
    
    def load(self, base_path: str) -> None:
        """
        Load all models.
        
        Args:
            base_path: Base path for model files
        """
        for horizon_enum in [TimeHorizon.D1, TimeHorizon.D7, TimeHorizon.D14]:
            horizon = horizon_enum.value
            path = f"{base_path}_revenue_{horizon}.txt"
            self.models[horizon] = lgb.Booster(model_file=path)
            logger.info(f"Revenue model {horizon} loaded from {path}")
        
        assert len(self.models) == 3, "Should have loaded 3 models"

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute RMSE with backward compatibility for older sklearn versions."""
        try:
            return mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))