"""
Fast inference engine for production deployment
"""
import numpy as np
import pandas as pd
import time
from pathlib import Path
import yaml
import logging
from typing import Dict, Any

from src.data.loader import DataLoader
from src.data.preprocessor import NestedFeatureParser
from src.features.engineer import FeatureEngineer
from src.models.buyer_classifier import BuyerClassifier
from src.models.revenue_regressor import ODMNRevenueRegressor
from src.models.ensemble import StackingEnsemble

logger = logging.getLogger(__name__)


class FastPredictor:
    """Optimized inference pipeline"""
    
    def __init__(self, config_path: str):
        """Load trained models"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
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
        
        logger.info("Models loaded successfully")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fast inference on test data
        
        Args:
            df: Raw test data
        
        Returns:
            Array of predicted iap_revenue_d7
        """
        # Preprocess
        df = self.preprocessor.process_all(df)
        
        # Feature engineering (fit=False to use cached encoders)
        df = self.feature_engineer.engineer_all(df, target_col=None, fit=False)
        
        # Select features
        feature_cols = [col for col in df.columns if col not in [
            'row_id', 'datetime'
        ]]
        
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_cols].fillna(0)
        
        # Stage 1: Buyer probability
        buyer_proba = self.buyer_model.predict_proba(X)
        
        # Stage 2: Revenue predictions
        revenue_preds = self.revenue_model.predict(X, enforce_order=True)
        
        # Stage 3: Ensemble meta-features
        loss_config = self.config['models']['stage2_revenue']['loss']
        
        X_meta = pd.DataFrame({
            'buyer_proba': buyer_proba,
            'revenue_d1': revenue_preds['d1'],
            'revenue_d7': revenue_preds['d7'],
            'revenue_d14': revenue_preds['d14'],
            'weighted_revenue': (
                loss_config['lambda_d1'] * revenue_preds['d1'] +
                loss_config['lambda_d7'] * revenue_preds['d7'] +
                loss_config['lambda_d14'] * revenue_preds['d14']
            ),
            'buyer_x_revenue': buyer_proba * revenue_preds['d7']
        })
        
        # Final prediction
        final_preds = self.ensemble_model.predict(X_meta)
        
        return final_preds
    
    def predict_test_set(self) -> pd.DataFrame:
        """Predict on official test set"""
        logger.info("Loading test data...")
        
        data_loader = DataLoader(self.config)
        test_ddf = data_loader.load_test()
        test_df = test_ddf.compute()
        
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
        
        return submission