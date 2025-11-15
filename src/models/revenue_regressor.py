"""
Stage 2: Multi-Horizon Revenue Regression (ODMN)
Predicts [iap_revenue_d1, iap_revenue_d7, iap_revenue_d14] jointly
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ODMNRevenueRegressor:
    """
    Order-preserving multi-task revenue regressor
    Based on Kuaishou ODMN (CIKM 2022)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}  # Separate model for each horizon
        self.feature_names = None
        self.horizons = config['models']['stage2_revenue']['horizons']
    
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
        sample_weight: np.ndarray = None
    ):
        """
        Train multi-horizon revenue models
        
        Note: For simplicity, we train separate models per horizon
        with post-processing order enforcement. For full ODMN,
        would need custom objective function.
        """
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
            
            model = lgb.LGBMRegressor(**model_config)
            
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=['huber', 'rmse'],
                sample_weight=sample_weight,
                callbacks=[
                    lgb.early_stopping(model_config.get('early_stopping_rounds', 100)),
                    lgb.log_evaluation(100)
                ]
            )
            
            self.models[horizon] = model
            
            # Validation metrics
            y_val_pred = model.predict(X_val)
            rmse = self._rmse(y_val, y_val_pred)
            mae = mean_absolute_error(y_val, y_val_pred)
            
            logger.info(f"{horizon} - Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    def predict(self, X: pd.DataFrame, enforce_order: bool = True) -> Dict[str, np.ndarray]:
        """
        Predict revenue for all horizons
        
        Args:
            X: Features
            enforce_order: Apply order-preserving post-processing
        
        Returns:
            Dictionary with keys 'd1', 'd7', 'd14'
        """
        if not self.models:
            raise ValueError("Models not trained yet")
        
        predictions = {}
        
        for horizon, model in self.models.items():
            predictions[horizon] = model.predict(X)
        
        # Enforce order constraints: D1 ≤ D7 ≤ D14
        if enforce_order:
            predictions = self._enforce_order_constraints(predictions)
        
        return predictions
    
    def _enforce_order_constraints(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Post-processing to ensure D1 ≤ D7 ≤ D14
        Uses isotonic regression approach
        """
        d1 = predictions['d1']
        d7 = predictions['d7']
        d14 = predictions['d14']
        
        # Clip D7 to be between D1 and D14
        d7_corrected = np.clip(d7, d1, d14)
        
        # If D1 > D7, adjust D1
        d1_corrected = np.minimum(d1, d7_corrected)
        
        # If D7 > D14, adjust D14
        d14_corrected = np.maximum(d14, d7_corrected)
        
        return {
            'd1': d1_corrected,
            'd7': d7_corrected,
            'd14': d14_corrected
        }
    
    def save(self, base_path: str):
        """Save all models"""
        for horizon, model in self.models.items():
            path = f"{base_path}_revenue_{horizon}.txt"
            model.booster_.save_model(path)
            logger.info(f"Revenue model {horizon} saved to {path}")
    
    def load(self, base_path: str):
        """Load all models"""
        for horizon in ['d1', 'd7', 'd14']:
            path = f"{base_path}_revenue_{horizon}.txt"
            self.models[horizon] = lgb.Booster(model_file=path)
            logger.info(f"Revenue model {horizon} loaded from {path}")

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute RMSE with backward compatibility for older sklearn versions."""
        try:
            return mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))