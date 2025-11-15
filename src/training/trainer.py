"""
Orchestrates the complete training pipeline
"""
import numpy as np
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Tuple, List

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
    
    def _apply_sampling(self, ddf: dd.DataFrame, dataset: str) -> dd.DataFrame:
        """Optionally sample or cap partitions to reduce memory usage"""
        if ddf is None:
            return None

        sampling_cfg = self.config['training'].get('sampling', {}) or {}
        random_state = sampling_cfg.get('random_state', 42)

        frac_key = f'{dataset}_frac'
        frac = sampling_cfg.get(frac_key, sampling_cfg.get('frac', 1.0))
        if frac is not None and frac < 1.0:
            logger.info(f"Sampling {dataset} dataset with frac={frac}")
            ddf = ddf.sample(frac=frac, random_state=random_state)

        max_parts_key = f'max_{dataset}_partitions'
        max_parts = sampling_cfg.get(max_parts_key)
        if isinstance(max_parts, int) and max_parts > 0:
            logger.info(f"Restricting {dataset} dataset to first {max_parts} partitions")
            ddf = ddf.partitions[:max_parts]

        return ddf

    def _numeric_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return numeric feature columns excluding labels/metadata."""
        excluded = {
            'row_id', 'datetime', 'iap_revenue_d7', 'iap_revenue_d1', 'iap_revenue_d14',
            'iap_revenue_d28', 'buyer_d1', 'buyer_d7', 'buyer_d14', 'buyer_d28',
            'buy_d7', 'buy_d14', 'buy_d28', 'registration',
            'retention_d1', 'retention_d7', 'retention_d14'
        }

        feature_cols = [col for col in df.columns if col not in excluded]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            raise ValueError(
                "No numeric feature columns available after preprocessing. "
                "Increase sampling fractions or inspect feature engineering."
            )

        return numeric_cols

    def _prepare_feature_matrix(
        self,
        df: pd.DataFrame,
        columns: List[str],
        context: str
    ) -> pd.DataFrame:
        """Align dataframe with expected columns, filling missing values."""
        missing = [col for col in columns if col not in df.columns]
        if missing:
            preview = ', '.join(missing[:5])
            logger.warning(
                "%s missing %d feature columns (%s); filling with zeros.",
                context,
                len(missing),
                preview
            )
        return df.reindex(columns=columns).fillna(0)

    def _get_target_array(self, df: pd.DataFrame, column: str, context: str) -> np.ndarray:
        """Return target column values, defaulting to zeros if missing."""
        if column in df.columns:
            return df[column].values
        logger.warning("%s missing target column %s; returning zeros.", context, column)
        return np.zeros(len(df), dtype=float)

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess data
        
        Returns:
            train_df, val_df
        """
        logger.info("=" * 80)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("=" * 80)

        sampling_cfg = self.config['training'].get('sampling', {}) or {}
        random_state = sampling_cfg.get('random_state', 42)
        
        # Load data
        train_ddf, val_ddf = self.data_loader.load_train(validation_split=True)
        train_ddf = self._apply_sampling(train_ddf, 'train')
        val_ddf = self._apply_sampling(val_ddf, 'val') if val_ddf is not None else None
        
        # Compute to pandas (batched if needed)
        logger.info("Computing train data...")
        train_df = train_ddf.compute()
        train_df = train_df.reset_index(drop=True)
        
        logger.info("Computing validation data...")
        val_df = val_ddf.compute()
        val_df = val_df.reset_index(drop=True)

        if train_df.empty:
            raise ValueError("Training dataframe is empty after sampling; increase sampling parameters.")

        if val_df.empty:
            fallback_rows = sampling_cfg.get('fallback_val_rows', min(512, len(train_df)))
            fallback_rows = max(1, min(len(train_df), fallback_rows))
            logger.warning(
                "Validation dataframe empty after sampling; using %d train rows as proxy validation set.",
                fallback_rows
            )
            val_df = train_df.sample(n=fallback_rows, random_state=random_state).copy()
            val_df = val_df.reset_index(drop=True)
        
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
        
        numeric_cols = self._numeric_feature_columns(train_df)

        X_train = self._prepare_feature_matrix(train_df, numeric_cols, "Stage 1 train")
        y_train = train_df['buyer_d7'].astype(int).values

        X_val = self._prepare_feature_matrix(val_df, numeric_cols, "Stage 1 validation")
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
        
        numeric_cols = self._numeric_feature_columns(train_df)
        
        X_train = self._prepare_feature_matrix(train_buyers, numeric_cols, "Stage 2 train buyers")
        # Use iap_revenue_d7 as proxy for d1 if d1 is not available
        y_train_d1 = self._get_target_array(train_buyers, 'iap_revenue_d1', "Stage 2 train buyers")
        if np.all(y_train_d1 == 0) and 'iap_revenue_d7' in train_buyers.columns:
            logger.info("Using iap_revenue_d7 * 0.3 as proxy for iap_revenue_d1")
            y_train_d1 = train_buyers['iap_revenue_d7'].values * 0.3
        y_train_d7 = self._get_target_array(train_buyers, 'iap_revenue_d7', "Stage 2 train buyers")
        y_train_d14 = self._get_target_array(train_buyers, 'iap_revenue_d14', "Stage 2 train buyers")

        X_val = self._prepare_feature_matrix(val_buyers, numeric_cols, "Stage 2 validation buyers")
        y_val_d1 = self._get_target_array(val_buyers, 'iap_revenue_d1', "Stage 2 validation buyers")
        if np.all(y_val_d1 == 0) and 'iap_revenue_d7' in val_buyers.columns:
            y_val_d1 = val_buyers['iap_revenue_d7'].values * 0.3
        y_val_d7 = self._get_target_array(val_buyers, 'iap_revenue_d7', "Stage 2 validation buyers")
        y_val_d14 = self._get_target_array(val_buyers, 'iap_revenue_d14', "Stage 2 validation buyers")
        
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
        numeric_cols = self._numeric_feature_columns(train_df)

        # Train meta-features
        X_train_raw = self._prepare_feature_matrix(train_df, numeric_cols, "Stage 3 train meta")
        buyer_proba_train = self.buyer_model.predict_proba(X_train_raw)
        revenue_preds_train = self.revenue_model.predict(X_train_raw, enforce_order=True)
        
        # Validation meta-features
        X_val_raw = self._prepare_feature_matrix(val_df, numeric_cols, "Stage 3 validation meta")
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