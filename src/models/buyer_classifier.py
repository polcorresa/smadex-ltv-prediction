"""
Stage 1: Buyer Classification Model.

Predicts P(buyer_d7 | features).

Following ArjanCodes best practices:
- Complete type hints with Optional
- Precondition/postcondition assertions
- Clear error messages
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BuyerClassifier:
    """Binary classifier for buyer prediction."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize buyer classifier.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        assert config is not None, "Config must not be None"
        assert 'models' in config, "Config must contain 'models' key"
        assert 'stage1_buyer' in config['models'], \
            "Config must contain stage1_buyer settings"
        
        self.config = config
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: Optional[list[str]] = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> None:
        """
        Train buyer classification model.
        
        Args:
            X_train: Training features
            y_train: Binary labels (buyer_d7)
            X_val: Validation features
            y_val: Validation labels
            sample_weight: Optional sample weights (from HistOS)
        """
        # Preconditions
        assert len(X_train) > 0, "Training data must not be empty"
        assert len(X_train) == len(y_train), \
            "X_train and y_train must have same length"
        assert len(X_val) == len(y_val), \
            "X_val and y_val must have same length"
        assert set(np.unique(y_train)).issubset({0, 1}), \
            "y_train must be binary (0 or 1)"
        assert set(np.unique(y_val)).issubset({0, 1}), \
            "y_val must be binary (0 or 1)"
        
        if sample_weight is not None:
            assert len(sample_weight) == len(X_train), \
                "Sample weights must match training size"
            assert np.all(sample_weight >= 0), \
                "Sample weights must be non-negative"
        
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
        
        # Postconditions
        assert 0.0 <= auc <= 1.0, f"AUC must be in [0, 1], got {auc}"
        assert 0.0 <= ap <= 1.0, f"Average Precision must be in [0, 1], got {ap}"
        
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
        """
        Predict buyer probability.
        
        Args:
            X: Features
            
        Returns:
            Array of probabilities for positive class
        """
        assert self.model is not None, "Model not trained yet"
        assert len(X) > 0, "Input data must not be empty"
        
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Postconditions
        assert len(probabilities) == len(X), \
            "Output length must match input length"
        assert np.all((probabilities >= 0) & (probabilities <= 1)), \
            "Probabilities must be in [0, 1]"
        
        return probabilities
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: File path for saving model
        """
        assert self.model is not None, "Model not trained yet"
        
        self.model.booster_.save_model(path)
        logger.info(f"Buyer classifier saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: File path for loading model
        """
        self.model = lgb.Booster(model_file=path)
        logger.info(f"Buyer classifier loaded from {path}")