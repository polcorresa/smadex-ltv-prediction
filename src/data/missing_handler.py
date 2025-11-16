"""
Comprehensive missing value handling for Smadex LTV dataset.

Following ArjanCodes best practices:
- Complete type hints
- Enum for strategies instead of magic strings
- Clear assertions for validation
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging

from src.types import ImputationStrategy

logger = logging.getLogger(__name__)


class MissingValueHandler:
    """Handle missing values (None, NaN, null) with domain-appropriate strategies."""
    
    def __init__(self, config: dict) -> None:
        """
        Initialize missing value handler.
        
        Args:
            config: Configuration dictionary
        """
        assert config is not None, "Config must not be None"
        
        self.config = config
        self.imputation_values: Dict[str, Any] = {}
        self.column_strategies: Dict[str, ImputationStrategy] = {}
        
    def analyze_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze missing value patterns
        
        Returns:
            DataFrame with missing value statistics
        """
        missing_stats = []
        
        for col in df.columns:
            n_missing = df[col].isnull().sum()
            # Handle 'None' strings safely
            n_none = 0
            if df[col].dtype == 'object':
                try:
                    n_none = (df[col].astype(str) == 'None').sum()
                except:
                    n_none = 0
            
            total_missing = n_missing + n_none
            pct_missing = (total_missing / len(df)) * 100
            
            if total_missing > 0:
                missing_stats.append({
                    'column': col,
                    'n_missing': total_missing,
                    'pct_missing': pct_missing,
                    'dtype': str(df[col].dtype)
                })
        
        missing_df = pd.DataFrame(missing_stats)
        if not missing_df.empty:
            missing_df = missing_df.sort_values('pct_missing', ascending=False)
        
        return missing_df
    
    def _get_imputation_strategy(self, col: str, dtype: str) -> ImputationStrategy:
        """
        Determine imputation strategy based on column name and type.
        
        Args:
            col: Column name
            dtype: Data type string
            
        Returns:
            ImputationStrategy enum value
        """
        col_lower = col.lower()
        
        # Revenue/spending features -> 0 (missing means no spend)
        if any(keyword in col_lower for keyword in [
            'revenue', 'iap', 'spend', 'buy', 'purchase', 'cpm', 'ctr'
        ]):
            return ImputationStrategy.ZERO
        
        # Count features -> 0 (missing means no events)
        if any(keyword in col_lower for keyword in [
            'count', 'num_', '_count', 'n_', 'total_'
        ]):
            return ImputationStrategy.ZERO
        
        # Ratio/percentage features -> median or 0.5
        if any(keyword in col_lower for keyword in [
            'ratio', 'pct', 'percentage', 'rate', '_prank'
        ]):
            return ImputationStrategy.MEDIAN
        
        # Timestamp/recency features -> large value (long time ago)
        if any(keyword in col_lower for keyword in [
            'days_ago', 'last_', 'first_', 'recency'
        ]):
            return ImputationStrategy.LARGE_VALUE
        
        # Behavioral aggregations -> median
        if any(keyword in col_lower for keyword in [
            'avg_', 'mean_', 'std_', 'max_', 'min_', 'median_'
        ]):
            return ImputationStrategy.MEDIAN
        
        # Categorical features -> mode or 'missing'
        if dtype == 'object' or 'category' in dtype:
            return ImputationStrategy.MODE
        
        # Device/network features -> mode
        if any(keyword in col_lower for keyword in [
            'dev_', 'carrier', 'country', 'region', 'os', 'make', 'model'
        ]):
            return ImputationStrategy.MODE
        
        # Histogram/entropy features -> 0
        if any(keyword in col_lower for keyword in [
            'hist_', 'entropy', 'top'
        ]):
            return ImputationStrategy.ZERO
        
        # Encoded features -> special value
        if 'encoded' in col_lower:
            return ImputationStrategy.SPECIAL_VALUE
        
        # Default: median for numeric, mode for categorical
        if 'float' in dtype or 'int' in dtype:
            return ImputationStrategy.MEDIAN
        else:
            return ImputationStrategy.MODE
    
    def fit(self, df: pd.DataFrame, verbose: bool = True) -> MissingValueHandler:
        """
        Learn imputation values from training data.
        
        Args:
            df: Training dataframe
            verbose: Print missing value statistics
            
        Returns:
            Self for method chaining
        """
        assert df is not None, "DataFrame must not be None"
        assert len(df) > 0, "DataFrame must not be empty"
        
        logger.info("Analyzing missing values...")
        
        # Analyze missing patterns
        if verbose:
            missing_stats = self.analyze_missing(df)
            if not missing_stats.empty:
                logger.info(f"\nMissing value statistics:\n{missing_stats.head(20).to_string()}")
                logger.info(f"\nTotal columns with missing values: {len(missing_stats)}")
            else:
                logger.info("No missing values found!")
        
        # Learn imputation values for each column
        for col in df.columns:
            if df[col].isnull().sum() > 0 or (df[col] == 'None').sum() > 0:
                dtype = str(df[col].dtype)
                strategy = self._get_imputation_strategy(col, dtype)
                
                self.column_strategies[col] = strategy
                
                if strategy == ImputationStrategy.ZERO:
                    self.imputation_values[col] = 0
                    
                elif strategy == ImputationStrategy.MEDIAN:
                    # Convert to numeric first
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    median_val = numeric_values.median()
                    self.imputation_values[col] = median_val if not pd.isna(median_val) else 0
                    
                elif strategy == ImputationStrategy.MODE:
                    # Get most frequent value
                    try:
                        # Filter out None/nan values first
                        valid_values = df[col].dropna()
                        if len(valid_values) > 0:
                            mode_values = valid_values.mode()
                            if len(mode_values) > 0:
                                mode_val = mode_values[0]
                                # Only use scalar values
                                if not isinstance(mode_val, (list, dict, tuple)):
                                    self.imputation_values[col] = mode_val
                                else:
                                    # For complex types, use 'missing' string
                                    self.imputation_values[col] = 'missing'
                            else:
                                self.imputation_values[col] = 'missing' if dtype == 'object' else 0
                        else:
                            self.imputation_values[col] = 'missing' if dtype == 'object' else 0
                    except Exception as e:
                        logger.warning(f"Could not compute mode for {col}: {e}")
                        self.imputation_values[col] = 'missing' if dtype == 'object' else 0
                        
                elif strategy == ImputationStrategy.LARGE_VALUE:
                    # For recency features, use a large number (e.g., 9999 days)
                    self.imputation_values[col] = 9999
                    
                elif strategy == ImputationStrategy.SPECIAL_VALUE:
                    # For encoded features, use -1 (outside normal range)
                    self.imputation_values[col] = -1
                    
                else:
                    # Default to 0
                    self.imputation_values[col] = 0
        
        logger.info(f"Learned imputation strategies for {len(self.imputation_values)} columns")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned imputation to data
        
        Args:
            df: Dataframe to impute
            
        Returns:
            Dataframe with missing values filled
        """
        logger.info("Imputing missing values...")
        
        df = df.copy()
        n_imputed = 0
        
        for col in df.columns:
            # Handle 'None' strings (common in object columns)
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].replace('None', np.nan)
                    df[col] = df[col].replace('', np.nan)
                except:
                    pass  # Skip if replacement fails
            
            # Check if column has missing values
            if df[col].isnull().sum() > 0:
                if col in self.imputation_values:
                    # Use learned value
                    fill_value = self.imputation_values[col]
                    
                    # Skip if fill_value is not scalar (e.g., list, dict)
                    if isinstance(fill_value, (list, dict, tuple)):
                        logger.warning(f"Skipping imputation for {col}: fill value is not scalar")
                        continue
                    
                    try:
                        df[col] = df[col].fillna(fill_value)
                        n_imputed += 1
                    except Exception as e:
                        logger.warning(f"Could not impute {col}: {e}")
                else:
                    # Fallback: use type-based default
                    try:
                        if df[col].dtype == 'object':
                            df[col] = df[col].fillna('missing')
                        else:
                            df[col] = df[col].fillna(0)
                        n_imputed += 1
                    except Exception as e:
                        logger.warning(f"Could not impute {col}: {e}")
        
        logger.info(f"Imputed missing values in {n_imputed} columns")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Fit and transform in one step (for training data)
        
        Args:
            df: Training dataframe
            verbose: Print statistics
            
        Returns:
            Dataframe with missing values filled
        """
        self.fit(df, verbose=verbose)
        return self.transform(df)
    
    def create_missing_indicators(
        self, 
        df: pd.DataFrame, 
        threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Create binary indicators for missingness (can be predictive)
        
        Args:
            df: Input dataframe
            threshold: Only create indicators for columns with >threshold missing
            
        Returns:
            Dataframe with additional missing indicator columns
        """
        logger.info("Creating missing value indicators...")
        
        n_indicators = 0
        
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            
            if missing_pct > threshold:
                indicator_col = f'{col}_is_missing'
                df[indicator_col] = df[col].isnull().astype(int)
                n_indicators += 1
        
        logger.info(f"Created {n_indicators} missing indicators")
        
        return df
    
    def get_summary(self) -> Dict:
        """
        Get summary of imputation strategy
        
        Returns:
            Dictionary with strategy counts
        """
        strategy_counts = {}
        for strategy in self.column_strategies.values():
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_columns_with_missing': len(self.imputation_values),
            'strategy_distribution': strategy_counts,
            'sample_imputations': {
                col: {'strategy': self.column_strategies[col], 'value': val}
                for col, val in list(self.imputation_values.items())[:10]
            }
        }


def summarize_missing_values(df: pd.DataFrame) -> None:
    """
    Quick utility to print missing value summary
    
    Args:
        df: Input dataframe
    """
    print("="*80)
    print("MISSING VALUE SUMMARY")
    print("="*80)
    
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    pct_missing = (missing_cells / total_cells) * 100
    
    print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Total cells: {total_cells:,}")
    print(f"Missing cells: {missing_cells:,} ({pct_missing:.2f}%)")
    
    cols_with_missing = (df.isnull().sum() > 0).sum()
    print(f"Columns with missing values: {cols_with_missing} / {df.shape[1]}")
    
    print("\nTop 10 columns by missing percentage:")
    missing_by_col = df.isnull().sum().sort_values(ascending=False)
    missing_pct_by_col = (missing_by_col / len(df) * 100)
    
    for col, count in missing_by_col.head(10).items():
        pct = missing_pct_by_col[col]
        print(f"  {col:50s}: {count:7,} ({pct:5.1f}%)")
    
    print("="*80)

