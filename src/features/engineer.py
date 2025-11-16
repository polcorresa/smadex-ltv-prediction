"""
Advanced feature engineering for LTV prediction.

Following ArjanCodes best practices:
- Complete type hints
- Precondition/postcondition assertions
- Clear parameter validation
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from pandas.util import hash_pandas_object
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create predictive features from preprocessed data."""
    
    def __init__(self, config: Dict[str, any]) -> None:
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary with feature settings
        """
        assert config is not None, "Config must not be None"
        
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoders: Dict[str, TargetEncoder] = {}
        features_cfg = config.get('features', {}) if isinstance(config, dict) else {}
        self.enable_local_distributions: bool = bool(
            features_cfg.get('enable_local_distributions', False)
        )

    @staticmethod
    def _to_numeric(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
        """
        Safely coerce a series to numeric for arithmetic operations.
        
        Args:
            series: Input series
            fill_value: Value to use for NaN after conversion
            
        Returns:
            Numeric series with NaN filled
        """
        assert series is not None, "Series must not be None"
        assert np.isfinite(fill_value), "fill_value must be finite"
        
        result = pd.to_numeric(series, errors='coerce').fillna(fill_value)
        
        assert len(result) == len(series), "Result length must match input"
        return result
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        
        Args:
            df: Input dataframe with base features
            
        Returns:
            DataFrame with additional interaction features
        """
        assert df is not None, "DataFrame must not be None"
        assert len(df) > 0, "DataFrame must not be empty"
        
        logger.info("Creating interaction features...")
        
        # Whale user frequency
        if 'whale_users_bundle_revenue_prank_mean' in df.columns and 'avg_daily_sessions_total' in df.columns:
            whale_score = self._to_numeric(df['whale_users_bundle_revenue_prank_mean'])
            freq = self._to_numeric(df['avg_daily_sessions_total'])
            df['whale_x_freq'] = whale_score * freq
        
        # Purchase history × Recency
        if 'num_buys_bundle_mean' in df.columns and 'last_buy_recency_weight' in df.columns:
            num_buys = self._to_numeric(df['num_buys_bundle_mean'])
            recency_weight = self._to_numeric(df['last_buy_recency_weight'])
            df['purchase_recency_score'] = num_buys * recency_weight
        
        # Install velocity
        if 'new_bundles_count' in df.columns and 'weeks_since_first_seen' in df.columns:
            new_bundles = self._to_numeric(df['new_bundles_count'])
            weeks_seen = self._to_numeric(df['weeks_since_first_seen'])
            df['install_velocity'] = new_bundles / (weeks_seen + 1)
        
        # Category spending affinity
        if 'iap_revenue_usd_category_mean' in df.columns and 'num_buys_category_count' in df.columns:
            cat_rev = self._to_numeric(df['iap_revenue_usd_category_mean'])
            cat_count = self._to_numeric(df['num_buys_category_count'])
            df['category_avg_purchase_value'] = cat_rev / (cat_count + 1)
        
        # Device price × Purchase history (affordability proxy)
        if 'release_msrp' in df.columns and 'iap_revenue_usd_bundle_mean' in df.columns:
            bundle_rev = self._to_numeric(df['iap_revenue_usd_bundle_mean'])
            msrp = self._to_numeric(df['release_msrp'])
            df['device_spending_ratio'] = bundle_rev / (msrp + 1)
        
        return df

    def create_zero_activity_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add binary indicators when key behavioral counts are zero."""
        assert df is not None, "DataFrame must not be None"
        assert len(df) > 0, "DataFrame must not be empty"

        logger.info("Creating zero-activity indicators...")

        zero_feature_map: Dict[str, str] = {
            'num_buys_bundle_count': 'no_bundle_buys',
            'num_buys_category_count': 'no_category_buys',
            'num_buys_category_bottom_taxonomy_count': 'no_bottom_tax_buys',
            'iap_revenue_usd_bundle_count': 'no_bundle_revenue',
            'iap_revenue_usd_category_count': 'no_category_revenue',
            'iap_revenue_usd_category_bottom_taxonomy_count': 'no_bottom_tax_revenue',
            'new_bundles_count': 'no_new_bundles',
            'user_bundles_count': 'no_install_history',
            'avg_daily_sessions_total': 'no_sessions_recorded',
            'avg_duration_mean': 'no_session_duration',
        }

        for source_col, flag_col in zero_feature_map.items():
            if source_col in df.columns:
                numeric_values = self._to_numeric(df[source_col])
                df[flag_col] = (numeric_values <= 0).astype(int)

        histogram_prefixes = (
            'country_hist', 'region_hist', 'city_hist',
            'dev_osv_hist', 'dev_language_hist'
        )
        for prefix in histogram_prefixes:
            top_freq_col = f'{prefix}_top1_freq'
            if top_freq_col in df.columns:
                freq_values = self._to_numeric(df[top_freq_col])
                df[f'{prefix}_missing_hist'] = (freq_values <= 0).astype(int)

        timestamp_flags = {
            'last_buy_days_ago': 'never_bought_before',
            'last_install_ts_bundle_days_ago': 'never_installed_bundle_before'
        }
        for source_col, flag_col in timestamp_flags.items():
            if source_col in df.columns:
                numeric_values = self._to_numeric(df[source_col])
                df[flag_col] = (numeric_values >= 9990).astype(int)

        return df
    
    def create_local_distribution_features(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'iap_revenue_d7'
    ) -> pd.DataFrame:
        """
        Local percentile ranks (LDAO paper, 2025).
        Compute whale indicators within local contexts.
        
        Args:
            df: Input dataframe
            target_col: Target column for computing local distributions
            
        Returns:
            DataFrame with local distribution features
        """
        assert df is not None, "DataFrame must not be None"
        assert len(df) > 0, "DataFrame must not be empty"
        if target_col is None or target_col not in df.columns:
            logger.info(
                "Skipping local distribution features because target column '%s' is unavailable",
                target_col or "<missing>"
            )
            return df
        
        logger.info("Creating local distribution features...")
        
        # Local contexts to consider
        contexts = ['country', 'dev_os', 'advertiser_category']
        
        for context in contexts:
            if context in df.columns and target_col in df.columns:
                grouped = df.groupby(context, observed=False)[target_col]

                # Local spending rank
                df[f'{context}_local_spend_rank'] = (
                    grouped.rank(pct=True)
                    .fillna(0.5)
                )
                
                # Local average
                df[f'{context}_local_avg_spend'] = (
                    grouped.transform('mean')
                    .fillna(0)
                )
                
                # Deviation from local mean
                df[f'{context}_spend_deviation'] = (
                    df[target_col] - df[f'{context}_local_avg_spend']
                ) / (df[f'{context}_local_avg_spend'] + 1)
        
        return df
    
    def encode_categorical_features(
        self, 
        df: pd.DataFrame, 
        categorical_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """
        One-hot encoding for categorical features.
        
        Args:
            df: Input dataframe
            categorical_cols: List of categorical column names to encode
            
        Returns:
            DataFrame with encoded categorical features
        """
        assert df is not None, "DataFrame must not be None"
        assert len(df) > 0, "DataFrame must not be empty"
        
        metadata_cols = {'row_id', 'datetime'}

        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        else:
            categorical_cols = [col for col in categorical_cols if col in df.columns]
            categorical_cols.extend(
                col for col in df.select_dtypes(include=['object']).columns
                if col not in categorical_cols
            )

        # Remove metadata/critical columns from encoding
        categorical_cols = [col for col in categorical_cols if col not in metadata_cols]

        if not categorical_cols:
            logger.info("No categorical columns to encode")
            return df
            
        logger.info(f"Encoding categorical features (hash) for {len(categorical_cols)} columns")
        
        for col in categorical_cols:
            series = df[col].fillna("__missing__").astype(str)
            hashed = hash_pandas_object(series, index=False).astype('int64')
            df[col] = hashed
        
        return df
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced temporal features with cyclical encoding.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with temporal features
        """
        assert df is not None, "DataFrame must not be None"
        assert len(df) > 0, "DataFrame must not be empty"
        
        logger.info("Creating temporal features...")
        
        if 'hour' in df.columns:
            hour = self._to_numeric(df['hour'])
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            # Time segments
            df['is_morning'] = (hour >= 6) & (hour < 12)
            df['is_afternoon'] = (hour >= 12) & (hour < 18)
            df['is_evening'] = (hour >= 18) & (hour < 22)
            df['is_night'] = (hour >= 22) | (hour < 6)
        
        if 'weekday' in df.columns:
            weekday = self._to_numeric(df['weekday'])
            # Cyclical encoding
            df['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
            df['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
            
            # Weekend flag
            df['is_weekend'] = weekday.isin([5, 6]) if hasattr(weekday, 'isin') else df['weekday'].isin([5, 6])
        
        return df
    
    def create_aggregated_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated behavioral patterns.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with aggregated behavioral features
        """
        assert df is not None, "DataFrame must not be None"
        assert len(df) > 0, "DataFrame must not be empty"
        
        logger.info("Creating aggregated behavioral features...")
        
        # Session consistency
        if 'avg_daily_sessions_total' in df.columns and 'avg_act_days' in df.columns:
            sessions = self._to_numeric(df['avg_daily_sessions_total'])
            act_days = self._to_numeric(df['avg_act_days'])
            df['session_consistency'] = sessions * act_days
        
        # Engagement score
        if 'avg_duration' in df.columns and 'avg_daily_sessions_total' in df.columns:
            duration = self._to_numeric(df['avg_duration'])
            sessions = self._to_numeric(df['avg_daily_sessions_total'])
            df['engagement_score'] = duration * sessions
        
        # WiFi preference score
        if 'wifi_ratio' in df.columns and 'avg_daily_sessions_total' in df.columns:
            wifi_ratio = self._to_numeric(df['wifi_ratio'])
            sessions = self._to_numeric(df['avg_daily_sessions_total'])
            df['wifi_engagement'] = wifi_ratio * sessions
        
        return df
    
    def engineer_all(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'iap_revenue_d7',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        CRITICAL: Preserves essential columns for training/inference:
        - row_id: Unique identifier
        - datetime: Temporal information  
        - Target columns: buyer_d1, buyer_d7, buyer_d14, buyer_d28,
                         iap_revenue_d1, iap_revenue_d7, iap_revenue_d14, iap_revenue_d28
                         
        Args:
            df: Input dataframe
            target_col: Target column for local distribution features
            fit: Whether to fit encoders (True for training, False for inference)
            
        Returns:
            DataFrame with all engineered features
        """
        assert df is not None, "DataFrame must not be None"
        assert len(df) > 0, "DataFrame must not be empty"
        if fit:
            assert target_col is not None, "Target column must be provided when fit=True"
        assert isinstance(fit, bool), "fit must be a boolean"
        
        # Identify critical columns to preserve
        critical_cols = [
            'row_id', 'datetime',
            'buyer_d1', 'buyer_d7', 'buyer_d14', 'buyer_d28',
            'iap_revenue_d1', 'iap_revenue_d7', 'iap_revenue_d14', 'iap_revenue_d28'
        ]
        existing_critical = [col for col in critical_cols if col in df.columns]
        
        logger.info(f"Preserving {len(existing_critical)} critical columns: {existing_critical}")
        
        # 1. Interaction features
        df = self.create_interaction_features(df)
        
        # 2. Local distribution features (optional; disabled by default to avoid target leakage)
        if self.enable_local_distributions:
            if fit and target_col is not None and target_col in df.columns:
                df = self.create_local_distribution_features(df, target_col)
            else:
                logger.info(
                    "Skipping local distribution features (fit=%s, target_col=%s present=%s)",
                    fit,
                    target_col,
                    target_col in df.columns if target_col is not None else False
                )
        
        # 3. Categorical encoding (using empty list for now - can be extended if needed)
        df = self.encode_categorical_features(df, categorical_cols=None)
        
        # 4. Temporal features
        df = self.create_temporal_features(df)
        
        # 5. Aggregated behavioral features
        df = self.create_aggregated_behavioral_features(df)

        # 6. Zero-activity indicators
        df = self.create_zero_activity_flags(df)
        
        return df
