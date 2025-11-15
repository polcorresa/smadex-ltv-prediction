"""
Stage 1: Buyer Classification Model
Predicts P(buyer_d7 | features)
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BuyerClassifier:
    """Binary classifier for buyer prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_names = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        sample_weight: np.ndarray = None
    ):
        """
        Train buyer classification model
        
        Args:
            X_train: Training features
            y_train: Binary labels (buyer_d7)
            X_val: Validation features
            y_val: Validation labels
            sample_weight: Optional sample weights (from HistOS)
        """
        logger.info("Training Stage 1: Buyer Classifier")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Get model config
        model_config = self.config['models']['stage1_buyer']['params'].copy()
        
        # Compute class imbalance ratio
        n_buyers = np.sum(y_train == 1)
        n_non_buyers = np.sum(y_train == 0)
        scale_pos_weight = n_non_buyers / n_buyers if n_buyers > 0 else 1.0
        
        logger.info(f"Class imbalance: {n_non_buyers} non-buyers, {n_buyers} buyers")
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        model_config['scale_pos_weight'] = scale_pos_weight
        
        # Initialize model
        self.model = lgb.LGBMClassifier(**model_config)
        
        # Train with early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc', 'average_precision'],
            sample_weight=sample_weight,
            callbacks=[
                lgb.early_stopping(model_config.get('early_stopping_rounds', 50)),
                lgb.log_evaluation(100)
            ]
        )
        
        # Validation metrics
        y_val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred_proba)
        ap = average_precision_score(y_val, y_val_pred_proba)
        
        logger.info(f"Validation AUC: {auc:.4f}")
        logger.info(f"Validation Average Precision: {ap:.4f}")
        
        # Feature importance
        importances = self.model.feature_importances_
        top_features = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(20)
        
        logger.info("Top 20 features:")
        logger.info(f"\n{top_features}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict buyer probability"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.booster_.save_model(path)
        logger.info(f"Buyer classifier saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        self.model = lgb.Booster(model_file=path)
        logger.info(f"Buyer classifier loaded from {path}")