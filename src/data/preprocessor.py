"""
Preprocessing for nested data structures (dicts, lists, tuples) - OPTIMIZED FOR DASK.

Following ArjanCodes best practices:
- Complete type hints
- Precondition/postcondition assertions
- Clear parameter validation
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from .missing_handler import MissingValueHandler

logger = logging.getLogger(__name__)


class NestedFeatureParser:
    """Parse complex nested features in Smadex dataset - DASK OPTIMIZED."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize parser.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.missing_handler = MissingValueHandler(self.config)
    
    @staticmethod
    def parse_dict_features(
        df: pd.DataFrame, 
        column: str, 
        aggregations: list[str] = ['mean', 'std', 'max', 'min']
    ) -> pd.DataFrame:
        """
        Parse dictionary/map columns (e.g., iap_revenue_usd_bundle).
        
        OPTIMIZED: Vectorized operations, no Python loops, uses pd.concat to avoid fragmentation.
        PRESERVES REVENUE DATA: Creates aggregated features before dropping original.
        
        Args:
            df: Input dataframe
            column: Column name containing dictionary data
            aggregations: List of aggregation functions to apply
            
        Returns:
            DataFrame with dictionary column expanded to aggregated features
            
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
        assert df is not None, "DataFrame must not be None"
        assert column is not None, "Column name must not be None"
        assert len(aggregations) > 0, "Must have at least one aggregation"
        
        if column not in df.columns:
            return df
        
        # Extract values from dictionaries (vectorized with fillna)
        values = df[column].apply(
            lambda x: list(x.values()) if isinstance(x, dict) and x else []
        )
        
        # Collect all new columns in a dictionary to avoid fragmentation
        new_columns = {}
        
        # Compute aggregations (vectorized NumPy operations)
        if 'mean' in aggregations:
            new_columns[f'{column}_mean'] = values.apply(
                lambda x: np.mean(x) if x else 0.0
            )
        if 'std' in aggregations:
            new_columns[f'{column}_std'] = values.apply(
                lambda x: np.std(x) if len(x) > 1 else 0.0
            )
        if 'max' in aggregations:
            new_columns[f'{column}_max'] = values.apply(
                lambda x: np.max(x) if x else 0.0
            )
        if 'min' in aggregations:
            new_columns[f'{column}_min'] = values.apply(
                lambda x: np.min(x) if x else 0.0
            )
        
        # Count non-zero entries (important for sparse revenue data)
        new_columns[f'{column}_count'] = values.apply(len)
        
        # Concatenate all new columns at once to avoid fragmentation
        new_df = pd.DataFrame(new_columns, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        
        # Drop original column AFTER extracting features (saves memory)
        df = df.drop(columns=[column])
        
        return df
    
    @staticmethod
    def parse_list_features(
        df: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        """
        Parse list columns (e.g., bundles_ins) - OPTIMIZED.
        
        Args:
            df: Input dataframe
            column: Column name containing list data
            
        Returns:
            DataFrame with list column expanded to count and unique count features
            
        Example:
            ['bundle_a', 'bundle_b', 'bundle_c'] ->
            {
                'bundles_ins_count': 3,
                'bundles_ins_unique': 3
            }
        """
        assert df is not None, "DataFrame must not be None"
        assert column is not None, "Column name must not be None"
        
        if column not in df.columns:
            return df
        
        # Collect new columns to avoid fragmentation
        new_columns = {}
        
        # Vectorized operations
        new_columns[f'{column}_count'] = df[column].apply(
            lambda x: len(x) if isinstance(x, (list, tuple)) or (hasattr(x, '__len__') and hasattr(x, 'tolist')) else 0
        )
        
        # Handle unique counts safely (some lists contain unhashable items)
        def safe_unique_count(x):
            if not isinstance(x, (list, tuple)):
                # Handle numpy arrays
                if hasattr(x, '__len__') and hasattr(x, 'tolist'):
                    try:
                        x = x.tolist()
                    except:
                        return 0
                else:
                    return 0
            try:
                # Try to use set for hashable items
                return len(set(x))
            except TypeError:
                # Fallback: count unique by converting to strings or just return count
                try:
                    return len(set(str(item) for item in x))
                except:
                    return len(x)  # Just return total count if unique fails
        
        new_columns[f'{column}_unique'] = df[column].apply(safe_unique_count)
        
        # Concatenate all new columns at once
        new_df = pd.DataFrame(new_columns, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        
        df = df.drop(columns=[column])
        return df
    
    @staticmethod
    def parse_session_features(
        df: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        """
        Parse session columns (list of tuples with counts) - OPTIMIZED.
        
        Args:
            df: Input dataframe
            column: Column name containing session data (list of tuples)
            
        Returns:
            DataFrame with session statistics extracted
            
        Example:
            [(hash1, 5), (hash2, 3)] ->
            {
                'avg_daily_sessions_total': 8,
                'avg_daily_sessions_mean': 4.0,
                'avg_daily_sessions_max': 5,
                'avg_daily_sessions_count': 2
            }
        """
        assert df is not None, "DataFrame must not be None"
        assert column is not None, "Column name must not be None"
        
        if column not in df.columns:
            return df
        
        def extract_session_stats(session_list):
            """Extract statistics from list of (hash, count) tuples"""
            if not isinstance(session_list, (list, tuple)) or not session_list:
                return {'total': 0.0, 'mean': 0.0, 'max': 0.0, 'count': 0}
            
            try:
                # Extract counts from tuples
                counts = [item[1] for item in session_list if isinstance(item, (list, tuple)) and len(item) >= 2]
                
                if not counts:
                    return {'total': 0.0, 'mean': 0.0, 'max': 0.0, 'count': 0}
                
                return {
                    'total': float(sum(counts)),
                    'mean': float(np.mean(counts)),
                    'max': float(max(counts)),
                    'count': len(counts)
                }
            except (ValueError, TypeError, IndexError):
                return {'total': 0.0, 'mean': 0.0, 'max': 0.0, 'count': 0}
        
        # Apply extraction
        session_stats = df[column].apply(extract_session_stats)
        session_df = pd.DataFrame(session_stats.tolist())
        
        # Add prefix
        session_df.columns = [f'{column}_{col}' for col in session_df.columns]
        
        # Concat and drop original
        df = pd.concat([df, session_df], axis=1)
        df = df.drop(columns=[column])
        
        return df
    
    @staticmethod
    def parse_histogram_features(
        df: pd.DataFrame,
        column: str,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Parse histogram columns (e.g., country_hist) - OPTIMIZED.
        Extract top-K most frequent values + entropy.
        
        Args:
            df: Input dataframe
            column: Column name containing histogram data
            top_k: Number of top frequencies to extract
            
        Returns:
            DataFrame with histogram features (top-k frequencies and entropy)
            
        Example:
            {'US': 100, 'UK': 50, 'DE': 30} ->
            {
                'country_hist_top1_freq': 100,
                'country_hist_top2_freq': 50,
                'country_hist_entropy': 1.23
            }
        """
        assert df is not None, "DataFrame must not be None"
        assert column is not None, "Column name must not be None"
        assert top_k > 0, "top_k must be positive"
        
        if column not in df.columns:
            return df
        
        def extract_top_k_and_entropy(hist: Dict[str, int]) -> Dict[str, float]:
            if not isinstance(hist, dict) or not hist:
                return {f'top{i+1}_freq': 0.0 for i in range(top_k)} | {'entropy': 0.0}
            
            # Sort by frequency (use heapq for large dicts if needed)
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
                probs = np.array(list(hist.values()), dtype=np.float32) / total
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                features['entropy'] = float(entropy)
            else:
                features['entropy'] = 0.0
            
            return features
        
        # Apply extraction (optimized with list comprehension)
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
        Parse timestamp columns to recency features - OPTIMIZED.
        
        Args:
            df: Input dataframe
            column: Column name containing timestamp data
            reference_date: Reference date for calculating recency
            
        Returns:
            DataFrame with timestamp converted to days_ago and recency_weight
            
        Example:
            last_buy_ts_bundle: '2025-09-15' ->
            {
                'last_buy_days_ago': 23,
                'last_buy_recency_weight': 0.48  # exp(-23/30)
            }
        """
        assert df is not None, "DataFrame must not be None"
        assert column is not None, "Column name must not be None"
        assert reference_date is not None, "Reference date must not be None"
        
        if column not in df.columns:
            return df
        
        ref_date = pd.to_datetime(reference_date)
        
        # Convert to datetime (vectorized)
        df[f'{column}_parsed'] = pd.to_datetime(df[column], errors='coerce')
        
        # Days ago (vectorized operation)
        df[f'{column}_days_ago'] = (
            ref_date - df[f'{column}_parsed']
        ).dt.days.fillna(9999)  # Large value for missing
        
        # Recency weight (exponential decay, 30-day half-life) - vectorized
        df[f'{column}_recency_weight'] = np.exp(
            -df[f'{column}_days_ago'] / 30.0
        )
        
        # Drop intermediate columns
        df = df.drop(columns=[column, f'{column}_parsed'])
        
        return df
    
    def process_all(
        self, 
        df: pd.DataFrame, 
        handle_missing: bool = True,
        create_missing_indicators: bool = True,
        fit_missing_handler: bool = True
    ) -> pd.DataFrame:
        """
        Process all nested features in dataset - DASK OPTIMIZED.
        
        All operations are partition-safe for Dask map_partitions.
        PRESERVES REVENUE DATA: Creates aggregated features before dropping.
        
        Args:
            df: Input dataframe (can be a Dask partition)
            handle_missing: Whether to impute missing values
            create_missing_indicators: Whether to create missingness indicators
            fit_missing_handler: Whether to fit the missing handler (True for train, False for test)
            
        Returns:
            Processed dataframe with all nested features expanded
        """
        assert df is not None, "DataFrame must not be None"
        assert len(df) > 0, "DataFrame must not be empty"
        
        # Create missing indicators BEFORE processing (to capture raw missingness)
        if create_missing_indicators:
            df = self.missing_handler.create_missing_indicators(df, threshold=0.05)
        
        # Dictionary features (purchase history) - PRESERVES REVENUE INFO
        dict_cols = [
            'iap_revenue_usd_bundle',  # Revenue by bundle
            'iap_revenue_usd_category',  # Revenue by category
            'iap_revenue_usd_category_bottom_taxonomy',  # Revenue by bottom taxonomy
            'num_buys_bundle',
            'num_buys_category',
            'num_buys_category_bottom_taxonomy',
            'cpm',
            'ctr',
            'cpm_pct_rk',
            'ctr_pct_rk',
            'rev_by_adv',  # Revenue by advertiser
            'rwd_prank',  # Reward prank
            'whale_users_bundle_num_buys_prank',
            'whale_users_bundle_revenue_prank',
            'whale_users_bundle_total_num_buys',
            'whale_users_bundle_total_revenue',
            'hour_ratio',
        ]
        
        for col in dict_cols:
            df = self.parse_dict_features(df, col)
        
        # List features (install history, taxonomies, etc.)
        list_cols = [
            'bundles_ins',
            'new_bundles',
            'user_bundles',  # Array of bundle IDs
            'user_bundles_l28d',  # Array of bundle IDs (last 28 days)
            'bcat',
            'bcat_bottom_taxonomy',
            'bundles_cat',
            'bundles_cat_bottom_taxonomy',
            'first_request_ts_bundle',
            'first_request_ts_category_bottom_taxonomy',
            'last_buy_ts_bundle',
            'last_buy_ts_category',
            'last_install_ts_bundle',
            'last_install_ts_category',
            'user_actions_bundles_action_count',
            'user_actions_bundles_action_last_timestamp',
        ]
        
        for col in list_cols:
            df = self.parse_list_features(df, col)
        
        # Session features (list of tuples with counts)
        session_cols = [
            'avg_daily_sessions',
            'avg_duration',  # Also a list of tuples
        ]
        
        for col in session_cols:
            if col in df.columns:
                df = self.parse_session_features(df, col)
        
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
        
        # Handle missing values AFTER feature engineering
        if handle_missing:
            if fit_missing_handler:
                df = self.missing_handler.fit_transform(df, verbose=False)  # Disable verbose in partitions
            else:
                df = self.missing_handler.transform(df)
        
        return df
