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
from typing import Dict, Any, Optional, List, Tuple
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
        
        ensemble_cfg = (self.config.get('models', {})
                         .get('ensemble', {}))

        base_models = self._build_base_models(
            ensemble_cfg.get('base_models', [])
        )

        meta_learner = self._build_meta_learner(
            ensemble_cfg.get('meta_learner', {})
        )
        
        # Stacking ensemble
        self.ensemble = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=int(ensemble_cfg.get('cv', 5)),
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

    def _build_base_models(
        self,
        configurations: List[Dict[str, Any]]
    ) -> List[Tuple[str, Any]]:
        """Instantiate configured base models with sensible fallbacks."""
        if not configurations:
            configurations = [
                {'type': 'elastic_net', 'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 1000},
                {'type': 'random_forest', 'n_estimators': 100, 'max_depth': 10},
                {
                    'type': 'xgboost',
                    'objective': 'reg:squarederror',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'n_estimators': 200
                }
            ]

        models: List[Tuple[str, Any]] = []
        for cfg in configurations:
            model_type = cfg.get('type', '').lower()
            if model_type == 'elastic_net':
                alpha = float(cfg.get('alpha', 0.1))
                l1_ratio = float(cfg.get('l1_ratio', 0.5))
                max_iter = int(cfg.get('max_iter', 1000))
                tol = float(cfg.get('tol', 1e-4))
                models.append((
                    'elastic',
                    ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=max_iter,
                        tol=tol
                    )
                ))
            elif model_type == 'random_forest':
                models.append((
                    'rf',
                    RandomForestRegressor(
                        n_estimators=int(cfg.get('n_estimators', 100)),
                        max_depth=cfg.get('max_depth', 10),
                        random_state=int(cfg.get('random_state', 42)),
                        n_jobs=-1
                    )
                ))
            elif model_type == 'xgboost':
                params = cfg.copy()
                params.setdefault('objective', 'reg:squarederror')
                params.setdefault('learning_rate', 0.05)
                params.setdefault('n_estimators', 200)
                params.setdefault('max_depth', 6)
                params.setdefault('random_state', 42)
                params.pop('type', None)
                models.append(('xgb', XGBRegressor(**params)))
            else:
                logger.warning(
                    "Unknown base model type '%s'; skipping",
                    cfg.get('type')
                )

        if not models:
            raise ValueError("Stacking ensemble requires at least one base model")

        return models

    def _build_meta_learner(self, config: Dict[str, Any]) -> lgb.LGBMRegressor:
        """Instantiate meta learner (currently LightGBM only)."""
        meta_type = config.get('type', 'lightgbm').lower()
        if meta_type != 'lightgbm':
            raise ValueError(f"Unsupported meta learner type: {meta_type}")

        params = config.copy()
        params.setdefault('objective', 'huber')
        params.setdefault('alpha', 0.9)
        params.setdefault('num_leaves', 31)
        params.setdefault('learning_rate', 0.05)
        params.setdefault('n_estimators', 200)
        params.setdefault('random_state', 42)
        params.pop('type', None)

        return lgb.LGBMRegressor(**params)
    
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
    loss_config: Dict[str, Any],
    gating_config: Dict[str, Any] | None = None
) -> pd.DataFrame:
    """Create Stage 3 meta-features shared by training and inference."""
    assert revenue_preds.length == len(buyer_proba), \
        "Buyer probabilities and revenue predictions must align"

    data: Dict[str, np.ndarray] = {'buyer_proba': buyer_proba}

    if gating_config:
        cap = float(gating_config.get('probability_cap', 1.0))
        cutoff = float(gating_config.get('probability_cutoff', 0.0))
        alpha = float(gating_config.get('alpha', 1.0))
    else:
        cap = 1.0
        cutoff = 0.0
        alpha = 1.0

    clipped = np.clip(buyer_proba, 0.0, cap)
    gate_factor = np.where(
        clipped <= cutoff,
        0.0,
        clipped ** alpha
    )
    data['buyer_proba_clipped'] = clipped
    data['buyer_gate_mask'] = (clipped > cutoff).astype(float)
    data['buyer_gate_powered'] = gate_factor

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
    data['gated_revenue'] = gate_factor * primary_preds

    return pd.DataFrame(data)