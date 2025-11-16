"""
Preprocessing for nested data structures (dicts, lists, tuples) - OPTIMIZED FOR DASK.

Following ArjanCodes best practices:
- Complete type hints
- Precondition/postcondition assertions
- Clear parameter validation
"""
from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

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
    def _dict_numeric_values(value: Any) -> list[float]:
        """Extract numeric values from a dict-like object."""
        if not isinstance(value, dict) or not value:
            return []
        numeric_values: list[float] = []
        for item in value.values():
            try:
                numeric_values.append(float(item))
            except (TypeError, ValueError):
                continue
        return numeric_values
    
    @staticmethod
    def _normalize_sequence(value: Any) -> list[Any]:
        """Return value as list when possible; otherwise an empty list."""
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if hasattr(value, 'tolist'):
            try:
                converted = value.tolist()
                if isinstance(converted, list):
                    return converted
                if isinstance(converted, tuple):
                    return list(converted)
            except (AttributeError, TypeError, ValueError):
                return []
        return []
    
    @staticmethod
    def _safe_unique_count(sequence: Sequence[Any]) -> int:
        """Count unique elements safeguarding against unhashable items."""
        if not sequence:
            return 0
        try:
            return len(set(sequence))
        except TypeError:
            try:
                return len(set(str(item) for item in sequence))
            except Exception:
                return len(sequence)
    
    @staticmethod
    def _extract_session_stats(session_list: Any) -> dict[str, float | int]:
        """Compute total, mean, max, and count from session tuples."""
        if not isinstance(session_list, (list, tuple)) or not session_list:
            return {'total': 0.0, 'mean': 0.0, 'max': 0.0, 'count': 0}
        counts: list[float] = []
        for item in session_list:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    counts.append(float(item[1]))
                except (TypeError, ValueError):
                    continue
        if not counts:
            return {'total': 0.0, 'mean': 0.0, 'max': 0.0, 'count': 0}
        total = float(np.sum(counts))
        mean = float(np.mean(counts))
        max_value = float(np.max(counts))
        count = len(counts)
        return {'total': total, 'mean': mean, 'max': max_value, 'count': count}
    
    @staticmethod
    def _extract_top_k_and_entropy(hist: Any, top_k: int) -> dict[str, float]:
        """Extract top-k frequencies and entropy from histogram dict."""
        if not isinstance(hist, dict) or not hist:
            return {f'top{i+1}_freq': 0.0 for i in range(top_k)} | {'entropy': 0.0}
        sorted_items = sorted(hist.items(), key=lambda x: x[1], reverse=True)
        features: dict[str, float] = {}
        for i in range(top_k):
            features[f'top{i+1}_freq'] = float(sorted_items[i][1]) if i < len(sorted_items) else 0.0
        total = float(sum(hist.values()))
        if total > 0.0:
            probs = np.array(list(hist.values()), dtype=np.float32) / total
            entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
            features['entropy'] = entropy
        else:
            features['entropy'] = 0.0
        return features
    
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
        
        # Extract values from dictionaries using vectorized list comprehensions
        dict_values = [
            NestedFeatureParser._dict_numeric_values(value)
            for value in df[column].tolist()
        ]
        
        # Collect all new columns in a dictionary to avoid fragmentation
        new_columns: dict[str, list[float]] = {}
        
        # Compute aggregations with NumPy directly on the prepared lists
        if 'mean' in aggregations:
            new_columns[f'{column}_mean'] = [
                float(np.mean(values)) if values else 0.0
                for values in dict_values
            ]
        if 'std' in aggregations:
            new_columns[f'{column}_std'] = [
                float(np.std(values)) if len(values) > 1 else 0.0
                for values in dict_values
            ]
        if 'max' in aggregations:
            new_columns[f'{column}_max'] = [
                float(np.max(values)) if values else 0.0
                for values in dict_values
            ]
        if 'min' in aggregations:
            new_columns[f'{column}_min'] = [
                float(np.min(values)) if values else 0.0
                for values in dict_values
            ]
        
        # Count entries in each dict-derived list
        new_columns[f'{column}_count'] = [len(values) for values in dict_values]
        
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
        new_columns: dict[str, list[int]] = {}
        normalized_sequences = [
            NestedFeatureParser._normalize_sequence(value)
            for value in df[column].tolist()
        ]
        
        # Vectorized operations using list comprehensions
        new_columns[f'{column}_count'] = [len(seq) for seq in normalized_sequences]
        new_columns[f'{column}_unique'] = [
            NestedFeatureParser._safe_unique_count(seq)
            for seq in normalized_sequences
        ]
        
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
        
        session_stats = [
            NestedFeatureParser._extract_session_stats(session_list)
            for session_list in df[column].tolist()
        ]
        session_df = pd.DataFrame(session_stats, index=df.index)
        
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
        
        hist_features = [
            NestedFeatureParser._extract_top_k_and_entropy(hist, top_k)
            for hist in df[column].tolist()
        ]
        hist_df = pd.DataFrame(hist_features, index=df.index)
        
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

