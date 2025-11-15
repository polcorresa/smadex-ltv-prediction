"""
Preprocessing for nested data structures (dicts, lists, tuples)
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class NestedFeatureParser:
    """Parse complex nested features in Smadex dataset"""
    
    @staticmethod
    def parse_dict_features(
        df: pd.DataFrame, 
        column: str, 
        aggregations: List[str] = ['mean', 'std', 'max', 'min']
    ) -> pd.DataFrame:
        """
        Parse dictionary/map columns (e.g., iap_revenue_usd_bundle)
        
        Example:
            {'bundle_a': 10.5, 'bundle_b': 3.2} -> 
            {
                'bundle_mean': 6.85,
                'bundle_std': 5.16,
                'bundle_max': 10.5,
                'bundle_min': 3.2,
                'bundle_count': 2
            }
        """
        if column not in df.columns:
            return df
        
        logger.info(f"Parsing dict feature: {column}")
        
        # Extract values from dictionaries
        values = df[column].apply(
            lambda x: list(x.values()) if isinstance(x, dict) and x else []
        )
        
        # Compute aggregations
        for agg in aggregations:
            if agg == 'mean':
                df[f'{column}_mean'] = values.apply(
                    lambda x: np.mean(x) if x else 0.0
                )
            elif agg == 'std':
                df[f'{column}_std'] = values.apply(
                    lambda x: np.std(x) if len(x) > 1 else 0.0
                )
            elif agg == 'max':
                df[f'{column}_max'] = values.apply(
                    lambda x: np.max(x) if x else 0.0
                )
            elif agg == 'min':
                df[f'{column}_min'] = values.apply(
                    lambda x: np.min(x) if x else 0.0
                )
        
        # Count non-zero entries
        df[f'{column}_count'] = values.apply(len)
        
        # Drop original column to save memory
        df = df.drop(columns=[column])
        
        return df
    
    @staticmethod
    def parse_list_features(
        df: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        """
        Parse list columns (e.g., bundles_ins)
        
        Example:
            ['bundle_a', 'bundle_b', 'bundle_c'] ->
            {
                'bundles_ins_count': 3,
                'bundles_ins_unique': 3
            }
        """
        if column not in df.columns:
            return df
        
        logger.info(f"Parsing list feature: {column}")
        
        df[f'{column}_count'] = df[column].apply(
            lambda x: len(x) if isinstance(x, (list, tuple)) else 0
        )
        
        df[f'{column}_unique'] = df[column].apply(
            lambda x: len(set(x)) if isinstance(x, (list, tuple)) else 0
        )
        
        df = df.drop(columns=[column])
        return df
    
    @staticmethod
    def parse_histogram_features(
        df: pd.DataFrame,
        column: str,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Parse histogram columns (e.g., country_hist)
        Extract top-K most frequent values + entropy
        
        Example:
            {'US': 100, 'UK': 50, 'DE': 30} ->
            {
                'country_hist_top1_freq': 100,
                'country_hist_top2_freq': 50,
                'country_hist_entropy': 1.23
            }
        """
        if column not in df.columns:
            return df
        
        logger.info(f"Parsing histogram feature: {column}")
        
        def extract_top_k_and_entropy(hist: Dict[str, int]) -> Dict[str, float]:
            if not isinstance(hist, dict) or not hist:
                return {f'top{i+1}_freq': 0.0 for i in range(top_k)} | {'entropy': 0.0}
            
            # Sort by frequency
            sorted_items = sorted(hist.items(), key=lambda x: x[1], reverse=True)
            
            # Top-K frequencies
            features = {}
            for i in range(top_k):
                if i < len(sorted_items):
                    features[f'top{i+1}_freq'] = sorted_items[i][1]
                else:
                    features[f'top{i+1}_freq'] = 0.0
            
            # Entropy (measure of diversity)
            total = sum(hist.values())
            if total > 0:
                probs = np.array(list(hist.values())) / total
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                features['entropy'] = entropy
            else:
                features['entropy'] = 0.0
            
            return features
        
        # Apply extraction
        hist_features = df[column].apply(extract_top_k_and_entropy)
        hist_df = pd.DataFrame(hist_features.tolist())
        
        # Add prefix
        hist_df.columns = [f'{column}_{col}' for col in hist_df.columns]
        
        # Concat and drop original
        df = pd.concat([df, hist_df], axis=1)
        df = df.drop(columns=[column])
        
        return df
    
    @staticmethod
    def parse_timestamp_features(
        df: pd.DataFrame,
        column: str,
        reference_date: str = "2025-10-08"
    ) -> pd.DataFrame:
        """
        Parse timestamp columns to recency features
        
        Example:
            last_buy_ts_bundle: '2025-09-15' ->
            {
                'last_buy_days_ago': 23,
                'last_buy_recency_weight': 0.48  # exp(-23/30)
            }
        """
        if column not in df.columns:
            return df
        
        logger.info(f"Parsing timestamp feature: {column}")
        
        ref_date = pd.to_datetime(reference_date)
        
        # Convert to datetime
        df[f'{column}_parsed'] = pd.to_datetime(df[column], errors='coerce')
        
        # Days ago
        df[f'{column}_days_ago'] = (
            ref_date - df[f'{column}_parsed']
        ).dt.days.fillna(9999)  # Large value for missing
        
        # Recency weight (exponential decay, 30-day half-life)
        df[f'{column}_recency_weight'] = np.exp(
            -df[f'{column}_days_ago'] / 30.0
        )
        
        # Drop intermediate columns
        df = df.drop(columns=[column, f'{column}_parsed'])
        
        return df
    
    def process_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all nested features in dataset"""
        
        # Dictionary features (purchase history)
        dict_cols = [
            'iap_revenue_usd_bundle',
            'iap_revenue_usd_category',
            'num_buys_bundle',
            'num_buys_category',
            'cpm',
            'ctr',
            'cpm_pct_rk',
            'ctr_pct_rk',
            'rev_by_adv',
            'whale_users_bundle_num_buys_prank',
            'whale_users_bundle_revenue_prank'
        ]
        
        for col in dict_cols:
            df = self.parse_dict_features(df, col)
        
        # List features (install history)
        list_cols = [
            'bundles_ins',
            'new_bundles'
        ]
        
        for col in list_cols:
            df = self.parse_list_features(df, col)
        
        # Histogram features
        hist_cols = [
            'country_hist',
            'region_hist',
            'city_hist',
            'dev_osv_hist',
            'dev_language_hist'
        ]
        
        for col in hist_cols:
            df = self.parse_histogram_features(df, col)
        
        # Timestamp features
        ts_cols = [
            'last_buy',
            'last_buy_ts_bundle',
            'first_request_ts',
            'last_ins',
            'last_install_ts_bundle'
        ]
        
        for col in ts_cols:
            df = self.parse_timestamp_features(df, col)
        
        return df