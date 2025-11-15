"""
Orchestrates the complete training pipeline
"""
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Tuple

from src.data.loader import DataLoader
from src.data.preprocessor import NestedFeatureParser
from src.features.engineer import FeatureEngineer
from src.sampling.histos import HistogramOversampling, HistogramUndersampling
from src.models.buyer_classifier import BuyerClassifier
from src.models.revenue_regressor import ODMNRevenueRegressor
from src.models.ensemble import StackingEnsemble

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline for Smadex LTV prediction"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to config.yaml
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = DataLoader(self.config)
        self.preprocessor = NestedFeatureParser()
        self.feature_engineer = FeatureEngineer(self.config)
        
        self.buyer_model = None
        self.revenue_model = None
        self.ensemble_model = None
        
        # Setup directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories"""
        Path(self.config['data']['processed_path']).mkdir(parents=True, exist_ok=True)
        Path(self.config['data']['submission_path']).mkdir(parents=True, exist_ok=True)
        Path('models').mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess data
        
        Returns:
            train_df, val_df
        """
        logger.info("=" * 80)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("=" * 80)
        
        # Load data
        train_ddf, val_ddf = self.data_loader.load_train(validation_split=True)
        
        # Compute to pandas (batched if needed)
        logger.info("Computing train data...")
        train_df = train_ddf.compute()
        
        logger.info("Computing validation data...")
        val_df = val_ddf.compute()
        
        logger.info(f"Train size: {len(train_df)}")
        logger.info(f"Validation size: {len(val_df)}")
        
        # Preprocess nested features
        logger.info("Preprocessing nested features (train)...")
        train_df = self.preprocessor.process_all(train_df)
        
        logger.info("Preprocessing nested features (val)...")
        val_df = self.preprocessor.process_all(val_df)
        
        # Feature engineering
        logger.info("Engineering features (train)...")
        train_df = self.feature_engineer.engineer_all(
            train_df, 
            target_col='iap_revenue_d7',
            fit=True
        )
        
        logger.info("Engineering features (val)...")
        val_df = self.feature_engineer.engineer_all(
            val_df,
            target_col='iap_revenue_d7',
            fit=False
        )
        
        # Cache processed data
        if self.config['training']['cache_features']:
            cache_path = Path(self.config['data']['processed_path'])
            train_df.to_parquet(cache_path / 'train_processed.parquet')
            val_df.to_parquet(cache_path / 'val_processed.parquet')
            logger.info("Processed data cached")
        
        return train_df, val_df
    
    def train_stage1_buyer(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ):
        """Train Stage 1: Buyer Classifier"""
        logger.info("=" * 80)
        logger.info("STEP 2: STAGE 1 - BUYER CLASSIFICATION")
        logger.info("=" * 80)
        
        # Select features
        feature_cols = [col for col in train_df.columns if col not in [
            'row_id', 'datetime', 'iap_revenue_d7', 'iap_revenue_d1', 'iap_revenue_d14',
            'iap_revenue_d28', 'buyer_d1', 'buyer_d7', 'buyer_d14', 'buyer_d28',
            'buy_d7', 'buy_d14', 'buy_d28', 'registration', 
            'retention_d1', 'retention_d7', 'retention_d14'
        ]]
        
        # Remove any remaining non-numeric
        numeric_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X_train = train_df[numeric_cols].fillna(0)
        y_train = train_df['buyer_d7'].astype(int).values
        
        X_val = val_df[numeric_cols].fillna(0)
        y_val = val_df['buyer_d7'].astype(int).values
        
        # Apply HistOS sampling
        logger.info("Applying HistOS sampling...")
        histos_config = self.config['sampling']['histos']
        sampler = HistogramOversampling(
            n_bins=histos_config['n_bins'],
            window_size=histos_config['window_size'],
            target_percentile=histos_config['target_percentile']
        )
        
        X_train_resampled, y_train_resampled = sampler.fit_resample(
            X_train.values,
            y_train
        )
        
        X_train_resampled = pd.DataFrame(X_train_resampled, columns=numeric_cols)
        
        # Train buyer classifier
        self.buyer_model = BuyerClassifier(self.config)
        self.buyer_model.train(
            X_train_resampled,
            y_train_resampled,
            X_val,
            y_val
        )
        
        # Save model
        self.buyer_model.save('models/buyer_classifier.txt')
    
    def train_stage2_revenue(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ):
        """Train Stage 2: Revenue Regressor (ODMN)"""
        logger.info("=" * 80)
        logger.info("STEP 3: STAGE 2 - REVENUE REGRESSION (ODMN)")
        logger.info("=" * 80)
        
        # Filter to buyers only
        train_buyers = train_df[train_df['buyer_d7'] == 1].copy()
        val_buyers = val_df[val_df['buyer_d7'] == 1].copy()
        
        logger.info(f"Training on {len(train_buyers)} buyers")
        logger.info(f"Validation on {len(val_buyers)} buyers")
        
        # Select features (same as stage 1)
        feature_cols = [col for col in train_df.columns if col not in [
            'row_id', 'datetime', 'iap_revenue_d7', 'iap_revenue_d1', 'iap_revenue_d14',
            'iap_revenue_d28', 'buyer_d1', 'buyer_d7', 'buyer_d14', 'buyer_d28',
            'buy_d7', 'buy_d14', 'buy_d28', 'registration',
            'retention_d1', 'retention_d7', 'retention_d14'
        ]]
        
        numeric_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X_train = train_buyers[numeric_cols].fillna(0)
        y_train_d1 = train_buyers['iap_revenue_d1'].values
        y_train_d7 = train_buyers['iap_revenue_d7'].values
        y_train_d14 = train_buyers['iap_revenue_d14'].values
        
        X_val = val_buyers[numeric_cols].fillna(0)
        y_val_d1 = val_buyers['iap_revenue_d1'].values
        y_val_d7 = val_buyers['iap_revenue_d7'].values
        y_val_d14 = val_buyers['iap_revenue_d14'].values
        
        # Apply HistUS undersampling (reduce zero-heavy distribution)
        logger.info("Applying HistUS undersampling...")
        histus_sampler = HistogramUndersampling(
            n_bins=self.config['sampling']['histos']['n_bins'],
            target_percentile=25.0
        )
        
        X_train_resampled, y_train_d7_resampled = histus_sampler.fit_resample(
            X_train.values,
            y_train_d7
        )
        
        X_train_resampled = pd.DataFrame(X_train_resampled, columns=numeric_cols)
        
        # Resample other horizons accordingly
        indices = histus_sampler.fit_resample(
            np.arange(len(X_train)).reshape(-1, 1),
            y_train_d7
        )[0].flatten().astype(int)
        
        y_train_d1_resampled = y_train_d1[indices]
        y_train_d14_resampled = y_train_d14[indices]
        
        # Train ODMN revenue regressor
        self.revenue_model = ODMNRevenueRegressor(self.config)
        self.revenue_model.train(
            X_train_resampled,
            y_train_d1_resampled,
            y_train_d7_resampled,
            y_train_d14_resampled,
            X_val,
            y_val_d1,
            y_val_d7,
            y_val_d14
        )
        
        # Save models
        self.revenue_model.save('models/odmn')
    
    def train_stage3_ensemble(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ):
        """Train Stage 3: Stacking Ensemble"""
        logger.info("=" * 80)
        logger.info("STEP 4: STAGE 3 - STACKING ENSEMBLE")
        logger.info("=" * 80)
        
        # Get predictions from Stage 1 and Stage 2
        feature_cols = [col for col in train_df.columns if col not in [
            'row_id', 'datetime', 'iap_revenue_d7', 'iap_revenue_d1', 'iap_revenue_d14',
            'iap_revenue_d28', 'buyer_d1', 'buyer_d7', 'buyer_d14', 'buyer_d28',
            'buy_d7', 'buy_d14', 'buy_d28', 'registration',
            'retention_d1', 'retention_d7', 'retention_d14'
        ]]
        
        numeric_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # Train meta-features
        X_train_raw = train_df[numeric_cols].fillna(0)
        buyer_proba_train = self.buyer_model.predict_proba(X_train_raw)
        revenue_preds_train = self.revenue_model.predict(X_train_raw, enforce_order=True)
        
        # Validation meta-features
        X_val_raw = val_df[numeric_cols].fillna(0)
        buyer_proba_val = self.buyer_model.predict_proba(X_val_raw)
        revenue_preds_val = self.revenue_model.predict(X_val_raw, enforce_order=True)
        
        # Create meta-features
        loss_config = self.config['models']['stage2_revenue']['loss']
        
        X_train_meta = pd.DataFrame({
            'buyer_proba': buyer_proba_train,
            'revenue_d1': revenue_preds_train['d1'],
            'revenue_d7': revenue_preds_train['d7'],
            'revenue_d14': revenue_preds_train['d14'],
            'weighted_revenue': (
                loss_config['lambda_d1'] * revenue_preds_train['d1'] +
                loss_config['lambda_d7'] * revenue_preds_train['d7'] +
                loss_config['lambda_d14'] * revenue_preds_train['d14']
            ),
            'buyer_x_revenue': buyer_proba_train * revenue_preds_train['d7']
        })
        
        X_val_meta = pd.DataFrame({
            'buyer_proba': buyer_proba_val,
            'revenue_d1': revenue_preds_val['d1'],
            'revenue_d7': revenue_preds_val['d7'],
            'revenue_d14': revenue_preds_val['d14'],
            'weighted_revenue': (
                loss_config['lambda_d1'] * revenue_preds_val['d1'] +
                loss_config['lambda_d7'] * revenue_preds_val['d7'] +
                loss_config['lambda_d14'] * revenue_preds_val['d14']
            ),
            'buyer_x_revenue': buyer_proba_val * revenue_preds_val['d7']
        })
        
        y_train = train_df['iap_revenue_d7'].values
        y_val = val_df['iap_revenue_d7'].values
        
        # Train ensemble
        self.ensemble_model = StackingEnsemble(self.config)
        self.ensemble_model.train(
            X_train_meta,
            y_train,
            X_val_meta,
            y_val
        )
        
        # Save ensemble
        self.ensemble_model.save('models/stacking_ensemble.pkl')
    
    def run(self):
        """Execute complete training pipeline"""
        logger.info("Starting Smadex LTV Prediction Training Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Prepare data
        train_df, val_df = self.prepare_data()
        
        # Step 2: Train Stage 1 (Buyer Classifier)
        self.train_stage1_buyer(train_df, val_df)
        
        # Step 3: Train Stage 2 (Revenue Regressor)
        self.train_stage2_revenue(train_df, val_df)
        
        # Step 4: Train Stage 3 (Ensemble)
        self.train_stage3_ensemble(train_df, val_df)
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)