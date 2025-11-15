"""
Orchestrates the complete training pipeline.

Following ArjanCodes best practices:
- Complete type hints
- Proper use of dataclasses and enums
- Assertions for validation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Tuple, Optional

from src.data.loader import DataLoader
from src.data.preprocessor import NestedFeatureParser
from src.features.engineer import FeatureEngineer
from src.sampling.histos import HistogramOversampling, HistogramUndersampling
from src.models.buyer_classifier import BuyerClassifier
from src.models.revenue_regressor import ODMNRevenueRegressor
from src.models.ensemble import StackingEnsemble
from src.utils.logger import log_section, log_subsection, log_metric
from src.types import TimeHorizon

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
    
    def _remove_constant_columns(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove columns that have constant values (no information)
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            
        Returns:
            Filtered train_df, val_df
        """
        # Preserve critical columns
        critical_cols = [
            'row_id', 'datetime',
            'buyer_d1', 'buyer_d7', 'buyer_d14', 'buyer_d28',
            'iap_revenue_d1', 'iap_revenue_d7', 'iap_revenue_d14', 'iap_revenue_d28'
        ]
        
        # Find constant columns in training data
        constant_cols = []
        for col in train_df.columns:
            if col in critical_cols:
                continue
            try:
                # Check if column has only 1 unique value (including NaN handling)
                if train_df[col].nunique(dropna=False) <= 1:
                    constant_cols.append(col)
            except:
                # Handle unhashable types by converting to string
                try:
                    if train_df[col].astype(str).nunique(dropna=False) <= 1:
                        constant_cols.append(col)
                except:
                    logger.warning(f"Could not check if {col} is constant, keeping it")
        
        if constant_cols:
            logger.info(f"Removing {len(constant_cols)} constant columns")
            logger.info(f"  Examples: {constant_cols[:5]}")
            train_df = train_df.drop(columns=constant_cols)
            # Remove from validation as well (if they exist)
            val_constant_cols = [c for c in constant_cols if c in val_df.columns]
            val_df = val_df.drop(columns=val_constant_cols)
        else:
            logger.info("No constant columns found")
        
        return train_df, val_df

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess data
        
        Returns:
            train_df, val_df
        """
        log_section(logger, "STEP 1: DATA PREPARATION")

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
        
        # Log detailed data information after loading
        log_subsection(logger, "📊 DATA LOADED - Summary")
        logger.info(f"  📚 Train size: {len(train_df):,} rows × {len(train_df.columns)} columns")
        logger.info(f"  🎯 Validation size: {len(val_df):,} rows × {len(val_df.columns)} columns")
        logger.info(f"  💾 Train memory: {train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info(f"  💾 Validation memory: {val_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info(f"  ❓ Train missing cells: {train_df.isna().sum().sum():,} ({train_df.isna().sum().sum() / train_df.size * 100:.2f}%)")
        logger.info(f"  ❓ Validation missing cells: {val_df.isna().sum().sum():,} ({val_df.isna().sum().sum() / val_df.size * 100:.2f}%)")
        
        # Log target variable statistics
        if 'buyer_d7' in train_df.columns:
            buyer_rate = train_df['buyer_d7'].mean() * 100
            logger.info(f"  🎯 Buyer rate (D7): {buyer_rate:.2f}%")
        if 'iap_revenue_d7' in train_df.columns:
            avg_revenue = train_df['iap_revenue_d7'].mean()
            logger.info(f"  💰 Average revenue (D7): ${avg_revenue:.4f}")
        logger.info("-" * 80)
        
        # Preprocess nested features
        logger.info("")
        logger.info("  🔧 Preprocessing nested features (train)...")
        train_df = self.preprocessor.process_all(train_df)
        
        logger.info("  🔧 Preprocessing nested features (val)...")
        val_df = self.preprocessor.process_all(val_df)
        
        # Log data after preprocessing
        logger.info("")
        log_subsection(logger, "🔧 DATA PREPROCESSED - Summary")
        logger.info(f"  📚 Train size: {len(train_df):,} rows × {len(train_df.columns)} columns")
        logger.info(f"  🎯 Validation size: {len(val_df):,} rows × {len(val_df.columns)} columns")
        logger.info(f"  💾 Train memory: {train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info(f"  💾 Validation memory: {val_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info(f"  ❓ Train missing cells: {train_df.isna().sum().sum():,} ({train_df.isna().sum().sum() / train_df.size * 100:.2f}%)")
        logger.info(f"  ❓ Validation missing cells: {val_df.isna().sum().sum():,} ({val_df.isna().sum().sum() / val_df.size * 100:.2f}%)")
        
        # Log column types
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        cat_cols = train_df.select_dtypes(include=['object', 'category']).columns
        logger.info(f"  🔢 Numeric columns: {len(numeric_cols)}")
        logger.info(f"  📝 Categorical columns: {len(cat_cols)}")
        logger.info("-" * 80)
        
        # Feature engineering
        logger.info("")
        logger.info("  ⚙️  Engineering features (train)...")
        train_df = self.feature_engineer.engineer_all(
            train_df, 
            target_col='iap_revenue_d7',
            fit=True
        )
        
        logger.info("  ⚙️  Engineering features (val)...")
        val_df = self.feature_engineer.engineer_all(
            val_df,
            target_col='iap_revenue_d7',
            fit=False
        )
        
        # Log data after feature engineering
        logger.info("")
        log_subsection(logger, "⚙️  FEATURE ENGINEERING COMPLETE - Summary")
        logger.info(f"  📚 Train size: {len(train_df):,} rows × {len(train_df.columns)} columns")
        logger.info(f"  🎯 Validation size: {len(val_df):,} rows × {len(val_df.columns)} columns")
        logger.info(f"  💾 Train memory: {train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info(f"  💾 Validation memory: {val_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Log feature groups
        feature_groups = {
            '💰 Revenue': len([c for c in train_df.columns if 'revenue' in c.lower()]),
            '🛒 Buyer': len([c for c in train_df.columns if 'buyer' in c.lower() or 'buy' in c.lower()]),
            '📱 Session': len([c for c in train_df.columns if 'session' in c.lower()]),
            '📱 Device': len([c for c in train_df.columns if 'dev_' in c.lower()]),
            '🕐 Temporal': len([c for c in train_df.columns if any(x in c.lower() for x in ['hour', 'day', 'weekday'])]),
            '❓ Missing indicators': len([c for c in train_df.columns if '_is_missing' in c.lower()]),
            '📊 Aggregated': len([c for c in train_df.columns if any(x in c for x in ['_mean', '_std', '_max', '_min', '_count'])]),
        }
        for group, count in feature_groups.items():
            logger.info(f"  {group} features: {count}")
        logger.info("-" * 80)
        
        # Remove constant columns
        logger.info("")
        logger.info("  🧹 Removing constant columns...")
        train_df, val_df = self._remove_constant_columns(train_df, val_df)
        logger.info(f"  ✓ After constant removal: {len(train_df.columns)} columns")
        
        # Check for remaining object columns
        obj_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        metadata_cols = {'row_id', 'datetime'}
        obj_cols = [c for c in obj_cols if c not in metadata_cols]
        if obj_cols:
            logger.warning(f"  ⚠️  {len(obj_cols)} object columns still present after encoding: {obj_cols[:5]}")
        else:
            logger.info("  ✓ All categorical columns properly encoded")
        
        logger.info("")
        logger.info("=" * 80)
        
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
        log_section(logger, "STEP 2: STAGE 1 - BUYER CLASSIFICATION")
        
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
        log_section(logger, "STEP 3: STAGE 2 - REVENUE REGRESSION (ODMN)")
        
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
        log_section(logger, "STEP 4: STAGE 3 - STACKING ENSEMBLE")
        
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
            'revenue_d1': revenue_preds_train.d1,
            'revenue_d7': revenue_preds_train.d7,
            'revenue_d14': revenue_preds_train.d14,
            'weighted_revenue': (
                loss_config['lambda_d1'] * revenue_preds_train.d1 +
                loss_config['lambda_d7'] * revenue_preds_train.d7 +
                loss_config['lambda_d14'] * revenue_preds_train.d14
            ),
            'buyer_x_revenue': buyer_proba_train * revenue_preds_train.d7
        })
        
        X_val_meta = pd.DataFrame({
            'buyer_proba': buyer_proba_val,
            'revenue_d1': revenue_preds_val.d1,
            'revenue_d7': revenue_preds_val.d7,
            'revenue_d14': revenue_preds_val.d14,
            'weighted_revenue': (
                loss_config['lambda_d1'] * revenue_preds_val.d1 +
                loss_config['lambda_d7'] * revenue_preds_val.d7 +
                loss_config['lambda_d14'] * revenue_preds_val.d14
            ),
            'buyer_x_revenue': buyer_proba_val * revenue_preds_val.d7
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