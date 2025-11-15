"""
Stage 3: Stacking Ensemble
Combines predictions from buyer classifier and revenue regressor
Based on ScienceDirect 2025 Hybrid Ensemble approach
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
import lightgbm as lgb
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """
    Stacking ensemble for final revenue prediction
    Combines buyer probability + multi-horizon revenue predictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ensemble = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray
    ):
        """
        Train stacking ensemble
        
        Args:
            X_train: Meta-features (buyer_proba, d1_pred, d7_pred, d14_pred, ...)
            y_train: Target (iap_revenue_d7)
            X_val: Validation meta-features
            y_val: Validation target
        """
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
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = mean_squared_error(y_val, y_val_pred, squared=False)
        mae = mean_absolute_error(y_val, y_val_pred)
        
        # MSLE
        msle = mean_squared_error(
            np.log1p(y_val),
            np.log1p(np.clip(y_val_pred, 0, None))
        )
        
        logger.info(f"Ensemble - Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}, MSLE: {msle:.6f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict final revenue"""
        if self.ensemble is None:
            raise ValueError("Ensemble not trained yet")
        
        predictions = self.ensemble.predict(X)
        
        # Ensure non-negative
        return np.clip(predictions, 0, None)
    
    def save(self, path: str):
        """Save ensemble"""
        import joblib
        joblib.dump(self.ensemble, path)
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: str):
        """Load ensemble"""
        import joblib
        self.ensemble = joblib.load(path)
        logger.info(f"Ensemble loaded from {path}")