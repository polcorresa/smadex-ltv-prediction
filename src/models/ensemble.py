"""
Stage 3: Stacking Ensemble.

Combines predictions from buyer classifier and revenue regressor.
Based on ScienceDirect 2025 Hybrid Ensemble approach.

Following ArjanCodes best practices:
- Complete type hints
- Precondition/postcondition assertions
- Clear error messages
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import lightgbm as lgb
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

from src.types import RevenuePredictions, TimeHorizon


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute RMSE with compatibility for older sklearn versions.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        RMSE score
    """
    assert len(y_true) == len(y_pred), "Arrays must have same length"
    assert len(y_true) > 0, "Arrays must not be empty"
    
    try:
        result = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        result = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    assert result >= 0.0, "RMSE must be non-negative"
    return result


class StackingEnsemble:
    """
    Stacking ensemble for final revenue prediction.
    
    Combines buyer probability + multi-horizon revenue predictions.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize stacking ensemble.
        
        Args:
            config: Configuration dictionary
        """
        assert config is not None, "Config must not be None"
        
        self.config = config
        self.ensemble: Optional[StackingRegressor] = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray
    ) -> None:
        """
        Train stacking ensemble.
        
        Args:
            X_train: Meta-features (buyer_proba, d1_pred, d7_pred, d14_pred, ...)
            y_train: Target (iap_revenue_d7)
            X_val: Validation meta-features
            y_val: Validation target
        """
        # Preconditions
        assert len(X_train) > 0, "Training data must not be empty"
        assert len(X_train) == len(y_train), \
            "X_train and y_train must have same length"
        assert len(X_val) == len(y_val), \
            "X_val and y_val must have same length"
        assert np.all(y_train >= 0), "Revenue targets must be non-negative"
        assert np.all(y_val >= 0), "Revenue targets must be non-negative"
        
        logger.info("Training Stage 3: Stacking Ensemble")
        
        # Base models
        base_models = [
            ('elastic', ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                max_iter=1000
            )),
            ('rf', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )),
            ('xgb', XGBRegressor(
                objective='reg:squarederror',
                max_depth=6,
                learning_rate=0.05,
                n_estimators=200,
                random_state=42
            ))
        ]
        
        # Meta-learner (LightGBM with Huber loss)
        meta_learner = lgb.LGBMRegressor(
            objective='huber',
            alpha=0.9,
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=200,
            random_state=42
        )
        
        # Stacking ensemble
        self.ensemble = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        # Train
        self.ensemble.fit(X_train, y_train)
        
        # Validation metrics
        y_val_pred = self.ensemble.predict(X_val)
        
        rmse = _rmse(y_val, y_val_pred)
        mae = mean_absolute_error(y_val, y_val_pred)
        
        # MSLE
        msle = mean_squared_error(
            np.log1p(y_val),
            np.log1p(np.clip(y_val_pred, 0, None))
        )
        
        # Postconditions
        assert rmse >= 0.0, "RMSE must be non-negative"
        assert mae >= 0.0, "MAE must be non-negative"
        assert msle >= 0.0, "MSLE must be non-negative"
        
        logger.info(f"Ensemble - Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}, MSLE: {msle:.6f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict final revenue.
        
        Args:
            X: Meta-features
            
        Returns:
            Array of revenue predictions
        """
        assert self.ensemble is not None, "Ensemble not trained yet"
        assert len(X) > 0, "Input data must not be empty"
        
        predictions = self.ensemble.predict(X)
        
        # Ensure non-negative
        predictions = np.clip(predictions, 0, None)
        
        # Postconditions
        assert len(predictions) == len(X), \
            "Output length must match input length"
        assert np.all(predictions >= 0), \
            "Predictions must be non-negative"
        
        return predictions
    
    def save(self, path: str) -> None:
        """
        Save ensemble.
        
        Args:
            path: File path for saving ensemble
        """
        assert self.ensemble is not None, "Ensemble not trained yet"
        
        import joblib
        joblib.dump(self.ensemble, path)
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load ensemble.
        
        Args:
            path: File path for loading ensemble
        """
        import joblib
        self.ensemble = joblib.load(path)
        logger.info(f"Ensemble loaded from {path}")


def build_meta_features(
    buyer_proba: np.ndarray,
    revenue_preds: RevenuePredictions,
    loss_config: Dict[str, Any]
) -> pd.DataFrame:
    """Create Stage 3 meta-features shared by training and inference."""
    assert revenue_preds.length == len(buyer_proba), \
        "Buyer probabilities and revenue predictions must align"

    data: Dict[str, np.ndarray] = {'buyer_proba': buyer_proba}

    horizons = revenue_preds.available_horizons()
    for horizon in horizons:
        horizon_preds = revenue_preds.get(horizon)
        assert horizon_preds is not None, "Missing predictions for configured horizon"
        data[f'revenue_{horizon.value}'] = horizon_preds

    weights = np.zeros_like(buyer_proba, dtype=float)
    for horizon in horizons:
        horizon_preds = revenue_preds.get(horizon)
        assert horizon_preds is not None
        lambda_key = f'lambda_{horizon.value}'
        weight = float(loss_config.get(lambda_key, 0.0))
        weights += weight * horizon_preds
    data['weighted_revenue'] = weights

    primary_horizon = revenue_preds.primary_horizon()
    primary_preds = revenue_preds.get(primary_horizon)
    assert primary_preds is not None
    data['buyer_x_revenue'] = buyer_proba * primary_preds

    return pd.DataFrame(data)