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
from typing import Dict, Any, Tuple, Optional, List

from sklearn.model_selection import train_test_split

from src.data.loader import DataLoader
from src.data.preprocessor import NestedFeatureParser
from src.features.engineer import FeatureEngineer
from src.sampling.histos import HistogramOversampling, HistogramUndersampling
from src.models.buyer_classifier import BuyerClassifier
from src.models.revenue_regressor import ODMNRevenueRegressor
from src.models.ensemble import StackingEnsemble, build_meta_features
from src.models.threshold_optimizer import BuyerThresholdOptimizer
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
        self.optimal_buyer_threshold = 1.0  # Default: use continuous probabilities
        
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

        leakage_patterns = ('retention', 'registration', 'local_spend', 'local_avg', 'spend_deviation')
        feature_cols = [
            col for col in df.columns
            if col not in excluded and not any(pattern in col for pattern in leakage_patterns)
        ]
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
        """Return target column values, raising if missing."""
        if column not in df.columns:
            raise ValueError(
                f"{context} missing required target column '{column}'. "
                "Ensure ingestion and preprocessing preserve this target."
            )
        return df[column].astype(float).values

    def _get_optional_target_array(
        self,
        df: pd.DataFrame,
        column: str,
        context: str
    ) -> np.ndarray | None:
        """Return target column values when available, otherwise None."""
        if column not in df.columns:
            logger.warning(
                "%s missing optional target column '%s'; skipping related horizon.",
                context.capitalize(),
                column
            )
            return None
        return df[column].astype(float).values

    def _ensure_target_columns(self, df: pd.DataFrame, context: str) -> pd.DataFrame:
        """Ensure essential revenue targets exist before continuing."""
        required = ['iap_revenue_d7', 'iap_revenue_d14']

        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"{context} dataframe missing required target columns: {missing}. "
                "Provide real revenue data instead of relying on proxies."
            )
        if 'iap_revenue_d1' not in df.columns:
            logger.info(
                "%s dataframe lacks iap_revenue_d1; Stage 2 will continue without D1 horizon.",
                context.capitalize()
            )

        return df
    
    def _preprocess_revenue_targets(self, df: pd.DataFrame, context: str) -> pd.DataFrame:
        """
        Preprocess revenue targets: cap outliers and optionally transform.
        
        Args:
            df: Dataframe with revenue columns
            context: Description for logging
            
        Returns:
            Dataframe with preprocessed revenue
        """
        preprocessing_cfg = self.config.get('preprocessing', {})
        revenue_cap = preprocessing_cfg.get('revenue_cap', None)
        
        if revenue_cap is not None and revenue_cap > 0:
            logger.info(f"Capping {context} revenue at ${revenue_cap:.2f}")
            for col in ['iap_revenue_d1', 'iap_revenue_d7', 'iap_revenue_d14']:
                if col in df.columns:
                    original_mean = df[col].mean()
                    original_max = df[col].max()
                    capped = np.sum(df[col] > revenue_cap)
                    df[col] = np.clip(df[col], 0, revenue_cap)
                    new_mean = df[col].mean()
                    logger.info(
                        f"  {col}: mean ${original_mean:.2f}→${new_mean:.2f}, "
                        f"max ${original_max:.2f}, capped {capped} samples ({capped/len(df)*100:.2f}%)"
                    )
        
        return df
    
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

    def _prepare_temporal_split_data(
        self,
        sampling_cfg: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data using temporal boundaries defined in the config."""
        random_state = sampling_cfg.get('random_state', 42)
        train_ddf, val_ddf = self.data_loader.load_train(validation_split=True)
        train_ddf = self._apply_sampling(train_ddf, 'train')
        val_ddf = self._apply_sampling(val_ddf, 'val') if val_ddf is not None else None

        logger.info("Computing train data...")
        train_df = train_ddf.compute().reset_index(drop=True)
        logger.info("Computing validation data...")
        val_df = val_ddf.compute().reset_index(drop=True) if val_ddf is not None else pd.DataFrame()

        if train_df.empty:
            raise ValueError("Training dataframe is empty after sampling; increase sampling parameters.")

        if val_df.empty:
            fallback_rows = sampling_cfg.get('fallback_val_rows', min(512, len(train_df)))
            fallback_rows = max(1, min(len(train_df), fallback_rows))
            logger.warning(
                "Validation dataframe empty after sampling; using %d train rows as proxy validation set.",
                fallback_rows
            )
            val_df = train_df.sample(n=fallback_rows, random_state=random_state).copy().reset_index(drop=True)

        train_df = self._ensure_target_columns(train_df, "train")
        val_df = self._ensure_target_columns(val_df, "validation")

        return train_df, val_df
    
    def _prepare_random_split_data_chunked(
        self,
        sampling_cfg: Dict[str, Any],
        split_cfg: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and split data using chunked processing to avoid RAM issues.
        
        This collects chunks incrementally and performs stratified split on the combined data.
        """
        random_state = split_cfg.get('random_state', sampling_cfg.get('random_state', 42))
        train_fraction = split_cfg.get('train_fraction', 0.8)
        assert 0.0 < train_fraction < 1.0, "train_fraction must be in (0, 1)"

        model_start = split_cfg.get('model_start') or self.config['data'].get('model_start') or self.config['data']['train_start']
        model_end = split_cfg.get('model_end') or self.config['data'].get('model_end') or self.config['data']['val_end']
        
        max_partitions = sampling_cfg.get('max_train_partitions', None)
        chunk_size = self.config.get('training', {}).get('chunk_size', 50000)

        logger.info(
            "Using chunked stratified random split between %s and %s (train_fraction=%.2f, chunk_size=%d)",
            model_start,
            model_end,
            train_fraction,
            chunk_size
        )
        
        # Collect all chunks
        all_chunks = []
        total_rows = 0
        
        logger.info("Loading data in chunks...")
        for chunk_df, _ in self.data_loader.iter_train_chunks(
            chunk_size=chunk_size,
            validation_split=False,
            start_dt=model_start,
            end_dt=model_end,
            max_partitions=max_partitions
        ):
            all_chunks.append(chunk_df)
            total_rows += len(chunk_df)
            
            if len(all_chunks) % 10 == 0:
                logger.info(f"  Loaded {len(all_chunks)} chunks, {total_rows:,} rows")
        
        logger.info(f"Loaded {len(all_chunks)} total chunks with {total_rows:,} rows")
        
        # Combine chunks
        logger.info("Combining chunks...")
        combined_df = pd.concat(all_chunks, ignore_index=True)
        del all_chunks  # Free memory
        
        # Apply sampling if needed
        frac = sampling_cfg.get('frac', 1.0)
        if frac < 1.0:
            logger.info(f"Sampling {frac*100:.1f}% of combined data...")
            combined_df = combined_df.sample(frac=frac, random_state=random_state).reset_index(drop=True)
            logger.info(f"After sampling: {len(combined_df):,} rows")

        if combined_df.empty:
            raise ValueError("Combined modeling dataframe is empty; adjust modeling window or sampling fraction.")

        # Stratified split
        stratify_column = split_cfg.get('stratify_column', 'buyer_d7')
        stratify_values: Optional[pd.Series] = None
        if stratify_column and stratify_column in combined_df.columns:
            unique_values = combined_df[stratify_column].nunique(dropna=False)
            if unique_values > 1:
                stratify_values = combined_df[stratify_column]
            else:
                logger.warning(
                    "Stratify column %s has %d unique values; proceeding without stratification.",
                    stratify_column,
                    unique_values
                )
        else:
            logger.warning(
                "Stratify column %s missing; proceeding without stratification.",
                stratify_column
            )

        logger.info("Performing stratified split...")
        train_df, val_df = train_test_split(
            combined_df,
            train_size=train_fraction,
            random_state=random_state,
            stratify=stratify_values if stratify_values is not None else None,
            shuffle=split_cfg.get('shuffle', True)
        )

        logger.info(
            "Random split complete ➜ train: %d rows, val: %d rows (stratify=%s)",
            len(train_df),
            len(val_df),
            stratify_column if stratify_values is not None else "none"
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        train_df = self._ensure_target_columns(train_df, "train")
        val_df = self._ensure_target_columns(val_df, "validation")

        return train_df, val_df

    def _prepare_random_split_data(
        self,
        sampling_cfg: Dict[str, Any],
        split_cfg: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load a combined modeling window and create a stratified random split."""
        random_state = split_cfg.get('random_state', sampling_cfg.get('random_state', 42))
        train_fraction = split_cfg.get('train_fraction', 0.8)
        assert 0.0 < train_fraction < 1.0, "train_fraction must be in (0, 1)"

        model_start = split_cfg.get('model_start') or self.config['data'].get('model_start') or self.config['data']['train_start']
        model_end = split_cfg.get('model_end') or self.config['data'].get('model_end') or self.config['data']['val_end']

        logger.info(
            "Using stratified random split between %s and %s (train_fraction=%.2f)",
            model_start,
            model_end,
            train_fraction
        )

        combined_ddf, _ = self.data_loader.load_train(
            validation_split=False,
            start_dt=model_start,
            end_dt=model_end
        )
        combined_ddf = self._apply_sampling(combined_ddf, 'train')

        logger.info("Computing combined modeling data...")
        combined_df = combined_ddf.compute().reset_index(drop=True)

        if combined_df.empty:
            raise ValueError("Combined modeling dataframe is empty; adjust modeling window or sampling fraction.")

        stratify_column = split_cfg.get('stratify_column', 'buyer_d7')
        stratify_values: Optional[pd.Series] = None
        if stratify_column and stratify_column in combined_df.columns:
            unique_values = combined_df[stratify_column].nunique(dropna=False)
            if unique_values > 1:
                stratify_values = combined_df[stratify_column]
            else:
                logger.warning(
                    "Stratify column %s has %d unique values; proceeding without stratification.",
                    stratify_column,
                    unique_values
                )
        else:
            logger.warning(
                "Stratify column %s missing; proceeding without stratification.",
                stratify_column
            )

        train_df, val_df = train_test_split(
            combined_df,
            train_size=train_fraction,
            random_state=random_state,
            stratify=stratify_values if stratify_values is not None else None,
            shuffle=split_cfg.get('shuffle', True)
        )

        logger.info(
            "Random split complete ➜ train: %d rows, val: %d rows (stratify=%s)",
            len(train_df),
            len(val_df),
            stratify_column if stratify_values is not None else "none"
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        train_df = self._ensure_target_columns(train_df, "train")
        val_df = self._ensure_target_columns(val_df, "validation")

        return train_df, val_df

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess data
        
        Returns:
            train_df, val_df
        """
        log_section(logger, "STEP 1: DATA PREPARATION")

        sampling_cfg = self.config['training'].get('sampling', {}) or {}
        split_cfg = self.config['training'].get('split', {}) or {}
        strategy = split_cfg.get('strategy', 'temporal').lower()
        
        # Use chunked loading to avoid RAM issues
        use_chunked = self.config.get('training', {}).get('use_chunked_loading', True)

        if strategy == 'stratified_random':
            if use_chunked:
                logger.info("Using chunked data loading (memory-efficient)")
                train_df, val_df = self._prepare_random_split_data_chunked(sampling_cfg, split_cfg)
            else:
                logger.info("Using standard Dask-based loading")
                train_df, val_df = self._prepare_random_split_data(sampling_cfg, split_cfg)
        else:
            train_df, val_df = self._prepare_temporal_split_data(sampling_cfg)
        
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
        
        # Preprocess revenue targets (cap outliers)
        logger.info("")
        logger.info("  💰 Preprocessing revenue targets...")
        train_df = self._preprocess_revenue_targets(train_df, "train")
        val_df = self._preprocess_revenue_targets(val_df, "validation")
        
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

        total_buyers = int(y_train.sum())
        total_non_buyers = int(len(y_train) - total_buyers)
        if total_buyers == 0:
            logger.warning(
                "Stage 1 train contains zero buyers; overriding scale_pos_weight to 1.0"
            )
            scale_pos_weight_override = 1.0
        else:
            scale_pos_weight_override = total_non_buyers / total_buyers
        logger.info(
            "Buyer distribution before oversampling: %d buyers (%.2f%%) vs %d non-buyers",
            total_buyers,
            (total_buyers / max(len(y_train), 1)) * 100.0,
            total_non_buyers
        )

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
            y_val,
            scale_pos_weight_override=scale_pos_weight_override
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
        
        logger.info(f"Buyers before zero-revenue filtering: {len(train_buyers)}")
        
        # Remove buyers with zero revenue (haven't spent yet)
        sampling_cfg = self.config.get('sampling', {}).get('histus', {})
        remove_zero_revenue = sampling_cfg.get('remove_zero_revenue', True)
        
        if remove_zero_revenue:
            zero_revenue_train = (train_buyers['iap_revenue_d7'] == 0).sum()
            train_buyers = train_buyers[train_buyers['iap_revenue_d7'] > 0].copy()
            logger.info(
                f"Removed {zero_revenue_train} zero-revenue buyers from training "
                f"({zero_revenue_train / len(train_df[train_df['buyer_d7'] == 1]) * 100:.2f}%)"
            )
            
            zero_revenue_val = (val_buyers['iap_revenue_d7'] == 0).sum()
            val_buyers = val_buyers[val_buyers['iap_revenue_d7'] > 0].copy()
            logger.info(
                f"Removed {zero_revenue_val} zero-revenue buyers from validation "
                f"({zero_revenue_val / len(val_df[val_df['buyer_d7'] == 1]) * 100:.2f}%)"
            )
        
        logger.info(f"Training on {len(train_buyers)} buyers with revenue > 0")
        logger.info(f"Validation on {len(val_buyers)} buyers with revenue > 0")
        
        # Log revenue distribution
        logger.info(f"\nTraining buyer revenue (D7): mean=${train_buyers['iap_revenue_d7'].mean():.4f}, median=${train_buyers['iap_revenue_d7'].median():.4f}")
        logger.info(f"Validation buyer revenue (D7): mean=${val_buyers['iap_revenue_d7'].mean():.4f}, median=${val_buyers['iap_revenue_d7'].median():.4f}")
        
        numeric_cols = self._numeric_feature_columns(train_df)
        
        X_train = self._prepare_feature_matrix(train_buyers, numeric_cols, "Stage 2 train buyers")
        y_train_d7 = self._get_target_array(train_buyers, 'iap_revenue_d7', "Stage 2 train buyers")
        y_train_d14 = self._get_target_array(train_buyers, 'iap_revenue_d14', "Stage 2 train buyers")
        y_train_d1 = self._get_optional_target_array(train_buyers, 'iap_revenue_d1', "Stage 2 train buyers")

        X_val = self._prepare_feature_matrix(val_buyers, numeric_cols, "Stage 2 validation buyers")
        y_val_d7 = self._get_target_array(val_buyers, 'iap_revenue_d7', "Stage 2 validation buyers")
        y_val_d14 = self._get_target_array(val_buyers, 'iap_revenue_d14', "Stage 2 validation buyers")
        y_val_d1 = self._get_optional_target_array(val_buyers, 'iap_revenue_d1', "Stage 2 validation buyers")

        targets_train = {
            TimeHorizon.D7: y_train_d7,
            TimeHorizon.D14: y_train_d14
        }
        targets_val = {
            TimeHorizon.D7: y_val_d7,
            TimeHorizon.D14: y_val_d14
        }

        if y_train_d1 is not None and y_val_d1 is not None:
            targets_train[TimeHorizon.D1] = y_train_d1
            targets_val[TimeHorizon.D1] = y_val_d1
        else:
            logger.info("Skipping D1 revenue horizon due to missing inputs.")
        
        # Apply HistUS undersampling (reduce zero-heavy distribution)
        logger.info("Applying HistUS undersampling...")
        histus_cfg = self.config.get('sampling', {}).get('histus', {})
        histus_n_bins = histus_cfg.get('n_bins', self.config['sampling']['histos']['n_bins'])
        histus_percentile = histus_cfg.get('target_percentile', 10.0)  # More aggressive
        
        logger.info(f"  Using n_bins={histus_n_bins}, target_percentile={histus_percentile}")
        histus_sampler = HistogramUndersampling(
            n_bins=histus_n_bins,
            target_percentile=histus_percentile
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
        
        resampled_targets = {
            TimeHorizon.D7: y_train_d7_resampled,
            TimeHorizon.D14: y_train_d14[indices]
        }
        if y_train_d1 is not None:
            resampled_targets[TimeHorizon.D1] = y_train_d1[indices]
        
        # Train ODMN revenue regressor
        self.revenue_model = ODMNRevenueRegressor(self.config)
        self.revenue_model.train(
            X_train_resampled,
            resampled_targets,
            X_val,
            targets_val
        )
        
        # Save models
        self.revenue_model.save('models/odmn')
    
    def optimize_buyer_threshold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> None:
        """
        Optimize buyer probability threshold/scaling to maximize revenue prediction.
        
        Uses validation data to find optimal continuous scaling factor (alpha)
        such that final_pred = (P(buyer)^alpha) * E(revenue|buyer) minimizes RMSLE.
        """
        log_section(logger, "STEP 2.5: BUYER THRESHOLD OPTIMIZATION")
        
        assert self.buyer_model is not None, "Buyer model must be trained first"
        assert self.revenue_model is not None, "Revenue model must be trained first"
        
        numeric_cols = self._numeric_feature_columns(val_df)
        X_val = self._prepare_feature_matrix(val_df, numeric_cols, "Threshold optimization")
        y_val_revenue = val_df['iap_revenue_d7'].values
        
        # Get buyer probabilities
        buyer_proba_val = self.buyer_model.predict_proba(X_val)
        
        # Get revenue predictions (for all samples)
        revenue_preds_val = self.revenue_model.predict(X_val, enforce_order=True)
        revenue_val_d7 = revenue_preds_val.get(TimeHorizon.D7)
        
        if revenue_val_d7 is None:
            logger.warning("No D7 predictions available; skipping threshold optimization")
            return
        
        # Optimize using continuous scaling
        optimizer = BuyerThresholdOptimizer(
            metric="rmsle",
            n_thresholds=100,
            threshold_range=(0.01, 0.99)
        )
        
        result = optimizer.optimize_continuous(
            buyer_proba=buyer_proba_val,
            revenue_pred=revenue_val_d7,
            y_true_revenue=y_val_revenue
        )
        
        self.optimal_buyer_threshold = result.optimal_threshold
        
        logger.info(f"Optimal buyer scaling factor (alpha): {self.optimal_buyer_threshold:.4f}")
        logger.info(f"Will apply: final_pred = (P(buyer)^{self.optimal_buyer_threshold:.4f}) * E(revenue|buyer)")
        
        # Save threshold
        import json
        threshold_path = Path('models/buyer_threshold.json')
        with open(threshold_path, 'w') as f:
            json.dump({
                'optimal_threshold': self.optimal_buyer_threshold,
                'optimization_metric': 'rmsle',
                'method': 'continuous_scaling'
            }, f, indent=2)
        logger.info(f"Threshold saved to {threshold_path}")
    
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
        
        loss_config = self.config['models']['stage2_revenue']['loss']
        X_train_meta = build_meta_features(buyer_proba_train, revenue_preds_train, loss_config)
        X_val_meta = build_meta_features(buyer_proba_val, revenue_preds_val, loss_config)
        
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
        
        # Step 2.5: Optimize buyer threshold
        self.optimize_buyer_threshold(train_df, val_df)
        
        # Step 4: Train Stage 3 (Ensemble)
        self.train_stage3_ensemble(train_df, val_df)
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)