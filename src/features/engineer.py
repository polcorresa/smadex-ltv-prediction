"""
Advanced feature engineering for LTV prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create predictive features from preprocessed data"""
    
    def __init__(self, config: dict):
        self.config = config
        self.label_encoders = {}
        self.target_encoders = {}
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables"""
        logger.info("Creating interaction features...")
        
        # Whale × Activity frequency
        if 'whale_users_bundle_revenue_prank_mean' in df.columns and 'avg_daily_sessions' in df.columns:
            df['whale_x_freq'] = (
                df['whale_users_bundle_revenue_prank_mean'] * 
                df['avg_daily_sessions']
            )
        
        # Purchase history × Recency
        if 'num_buys_bundle_mean' in df.columns and 'last_buy_days_ago' in df.columns:
            df['purchase_recency_score'] = (
                df['num_buys_bundle_mean'] * 
                df['last_buy_recency_weight']
            )
        
        # Install velocity
        if 'new_bundles_count' in df.columns and 'weeks_since_first_seen' in df.columns:
            df['install_velocity'] = (
                df['new_bundles_count'] / 
                (df['weeks_since_first_seen'] + 1)
            )
        
        # Category spending affinity
        if 'iap_revenue_usd_category_mean' in df.columns and 'num_buys_category_mean' in df.columns:
            df['category_avg_purchase_value'] = (
                df['iap_revenue_usd_category_mean'] / 
                (df['num_buys_category_count'] + 1)
            )
        
        # Device price × Purchase history (affordability proxy)
        if 'release_msrp' in df.columns and 'iap_revenue_usd_bundle_mean' in df.columns:
            df['device_spending_ratio'] = (
                df['iap_revenue_usd_bundle_mean'] / 
                (df['release_msrp'] + 1)
            )
        
        return df
    
    def create_local_distribution_features(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'iap_revenue_d7'
    ) -> pd.DataFrame:
        """
        Local percentile ranks (LDAO paper, 2025)
        Compute whale indicators within local contexts
        """
        logger.info("Creating local distribution features...")
        
        # Local contexts to consider
        contexts = ['country', 'dev_os', 'advertiser_category']
        
        for context in contexts:
            if context in df.columns and target_col in df.columns:
                # Local spending rank
                df[f'{context}_local_spend_rank'] = (
                    df.groupby(context)[target_col]
                    .rank(pct=True)
                    .fillna(0.5)
                )
                
                # Local average
                df[f'{context}_local_avg_spend'] = (
                    df.groupby(context)[target_col]
                    .transform('mean')
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
        target_col: str = 'iap_revenue_d7',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode high-cardinality categorical features
        Uses target encoding for better performance
        """
        logger.info("Encoding categorical features...")
        
        # High-cardinality categoricals
        high_card_cols = [
            'advertiser_bundle',
            'dev_make',
            'dev_model',
            'carrier'
        ]
        
        # Low-cardinality (label encoding)
        low_card_cols = [
            'advertiser_category',
            'advertiser_subcategory',
            'dev_os',
            'country',
            'region'
        ]
        
        # Label encoding for low-cardinality
        for col in low_card_cols:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(
                        df[col].astype(str).fillna('missing')
                    )
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        df[f'{col}_encoded'] = le.transform(
                            df[col].astype(str).fillna('missing')
                        )
        
        # Target encoding for high-cardinality (only if target available)
        if target_col in df.columns:
            for col in high_card_cols:
                if col in df.columns:
                    if fit:
                        te = TargetEncoder(smooth='auto')
                        df[f'{col}_target_encoded'] = te.fit_transform(
                            df[[col]].astype(str).fillna('missing'),
                            df[target_col]
                        )
                        self.target_encoders[col] = te
                    else:
                        te = self.target_encoders.get(col)
                        if te:
                            df[f'{col}_target_encoded'] = te.transform(
                                df[[col]].astype(str).fillna('missing')
                            )
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced temporal features"""
        logger.info("Creating temporal features...")
        
        if 'hour' in df.columns:
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # Time segments
            df['is_morning'] = (df['hour'] >= 6) & (df['hour'] < 12)
            df['is_afternoon'] = (df['hour'] >= 12) & (df['hour'] < 18)
            df['is_evening'] = (df['hour'] >= 18) & (df['hour'] < 22)
            df['is_night'] = (df['hour'] >= 22) | (df['hour'] < 6)
        
        if 'weekday' in df.columns:
            # Cyclical encoding
            df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
            df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
            
            # Weekend flag
            df['is_weekend'] = df['weekday'].isin([5, 6])
        
        return df
    
    def create_aggregated_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated behavioral patterns"""
        logger.info("Creating aggregated behavioral features...")
        
        # Session consistency
        if 'avg_daily_sessions' in df.columns and 'avg_act_days' in df.columns:
            df['session_consistency'] = (
                df['avg_daily_sessions'] * df['avg_act_days']
            )
        
        # Engagement score
        if 'avg_duration' in df.columns and 'avg_daily_sessions' in df.columns:
            df['engagement_score'] = (
                df['avg_duration'] * df['avg_daily_sessions']
            )
        
        # WiFi preference score
        if 'wifi_ratio' in df.columns and 'avg_daily_sessions' in df.columns:
            df['wifi_engagement'] = (
                df['wifi_ratio'] * df['avg_daily_sessions']
            )
        
        return df
    
    def engineer_all(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'iap_revenue_d7',
        fit: bool = True
    ) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        
        # 1. Interaction features
        df = self.create_interaction_features(df)
        
        # 2. Local distribution features (only for training with target)
        if target_col in df.columns:
            df = self.create_local_distribution_features(df, target_col)
        
        # 3. Categorical encoding
        df = self.encode_categorical_features(df, target_col, fit=fit)
        
        # 4. Temporal features
        df = self.create_temporal_features(df)
        
        # 5. Aggregated behavioral features
        df = self.create_aggregated_behavioral_features(df)
        
        return df