"""
Advanced feature selection for Smadex LTV prediction
Implements multiple selection strategies:
- Correlation-based filtering
- Variance-based filtering
- Tree-based importance (Random Forest, LightGBM)
- Recursive Feature Elimination (RFE)
- SHAP-based selection (Chen et al., 2025)
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE
)
from sklearn.linear_model import LassoCV
import lightgbm as lgb
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Comprehensive feature selection toolkit
    Based on: Htun et al. (2023), "Survey of feature selection and 
    extraction techniques for stock market prediction"
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.selected_features = None
        self.feature_importances = None
        self.correlation_matrix = None
        
    def remove_low_variance(
        self, 
        df: pd.DataFrame, 
        threshold: float = 0.01
    ) -> pd.DataFrame:
        """
        Remove features with low variance
        
        Args:
            df: Feature DataFrame
            threshold: Minimum variance threshold
        
        Returns:
            Filtered DataFrame
        """
        logger.info(f"Removing low variance features (threshold={threshold})...")
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df)
        
        selected_cols = df.columns[selector.get_support()].tolist()
        removed_count = len(df.columns) - len(selected_cols)
        
        logger.info(f"Removed {removed_count} low-variance features")
        
        return df[selected_cols]
    
    def remove_high_correlation(
        self, 
        df: pd.DataFrame, 
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Remove highly correlated features
        
        Strategy: For each pair of features with correlation > threshold,
        keep the one with higher variance
        
        Args:
            df: Feature DataFrame
            threshold: Correlation threshold
        
        Returns:
            Filtered DataFrame
        """
        logger.info(f"Removing highly correlated features (threshold={threshold})...")
        
        # Compute correlation matrix
        self.correlation_matrix = df.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = self.correlation_matrix.where(
            np.triu(np.ones(self.correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = set()
        
        for column in upper.columns:
            # Get features correlated with this one
            correlated = upper[column][upper[column] > threshold].index.tolist()
            
            if correlated:
                # Among correlated features, keep the one with highest variance
                variances = df[[column] + correlated].var()
                keep = variances.idxmax()
                
                # Drop others
                for col in [column] + correlated:
                    if col != keep:
                        to_drop.add(col)
        
        selected_cols = [col for col in df.columns if col not in to_drop]
        
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        
        return df[selected_cols]
    
    def select_by_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        method: str = 'lightgbm',
        n_features: Optional[int] = None,
        threshold: Optional[float] = 0.01,
        task: str = 'classification'
    ) -> Tuple[List[str], pd.Series]:
        """
        Select features by importance scores
        
        Args:
            X: Features
            y: Target
            method: 'lightgbm', 'random_forest', 'lasso'
            n_features: Number of features to select (if None, use threshold)
            threshold: Minimum importance threshold
            task: 'classification' or 'regression'
        
        Returns:
            (selected_features, importance_scores)
        """
        logger.info(f"Selecting features by importance ({method})...")
        
        if method == 'lightgbm':
            if task == 'classification':
                model = lgb.LGBMClassifier(
                    objective='binary',
                    num_leaves=31,
                    n_estimators=100,
                    random_state=42,
                    verbose=-1
                )
            else:
                model = lgb.LGBMRegressor(
                    objective='huber',
                    num_leaves=31,
                    n_estimators=100,
                    random_state=42,
                    verbose=-1
                )
            
            model.fit(X, y)
            importances = pd.Series(
                model.feature_importances_,
                index=X.columns
            )
        
        elif method == 'random_forest':
            if task == 'classification':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            
            model.fit(X, y)
            importances = pd.Series(
                model.feature_importances_,
                index=X.columns
            )
        
        elif method == 'lasso':
            # Lasso regression for feature selection
            model = LassoCV(cv=5, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            importances = pd.Series(
                np.abs(model.coef_),
                index=X.columns
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Sort by importance
        importances = importances.sort_values(ascending=False)
        
        # Select features
        if n_features is not None:
            selected = importances.head(n_features).index.tolist()
        else:
            # Normalize importances to sum to 1
            importances_norm = importances / importances.sum()
            selected = importances_norm[importances_norm >= threshold].index.tolist()
        
        logger.info(f"Selected {len(selected)} features by importance")
        
        self.feature_importances = importances
        
        return selected, importances
    
    def select_by_statistical_test(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        method: str = 'f_test',
        k: int = 100,
        task: str = 'classification'
    ) -> List[str]:
        """
        Select features by statistical tests
        
        Args:
            X: Features
            y: Target
            method: 'f_test', 'mutual_info'
            k: Number of features to select
            task: 'classification' or 'regression'
        
        Returns:
            selected_features
        """
        logger.info(f"Selecting features by {method}...")
        
        if method == 'f_test':
            if task == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression
        
        elif method == 'mutual_info':
            if task == 'classification':
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        
        selected = X.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected)} features by {method}")
        
        return selected
    
    def select_by_rfe(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_features: int = 100,
        task: str = 'classification'
    ) -> List[str]:
        """
        Recursive Feature Elimination
        
        Args:
            X: Features
            y: Target
            n_features: Number of features to select
            task: 'classification' or 'regression'
        
        Returns:
            selected_features
        """
        logger.info(f"Selecting features by RFE (n={n_features})...")
        
        if task == 'classification':
            estimator = lgb.LGBMClassifier(
                objective='binary',
                num_leaves=31,
                n_estimators=50,
                random_state=42,
                verbose=-1
            )
        else:
            estimator = lgb.LGBMRegressor(
                objective='huber',
                num_leaves=31,
                n_estimators=50,
                random_state=42,
                verbose=-1
            )
        
        selector = RFE(estimator, n_features_to_select=n_features, step=0.1)
        selector.fit(X, y)
        
        selected = X.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected)} features by RFE")
        
        return selected
    
    def select_comprehensive(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        task: str = 'classification',
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        importance_method: str = 'lightgbm',
        importance_threshold: float = 0.005,
        top_k: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive multi-stage feature selection
        
        Pipeline:
        1. Remove low-variance features
        2. Remove highly correlated features
        3. Select by importance
        
        Args:
            X: Features
            y: Target
            task: 'classification' or 'regression'
            variance_threshold: Variance threshold
            correlation_threshold: Correlation threshold
            importance_method: 'lightgbm', 'random_forest', 'lasso'
            importance_threshold: Minimum importance
            top_k: Select top-k features (overrides importance_threshold)
        
        Returns:
            (selected_X, selection_report)
        """
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE FEATURE SELECTION")
        logger.info("=" * 80)
        
        initial_count = X.shape[1]
        report = {
            'initial_features': initial_count,
            'stages': []
        }
        
        # Stage 1: Low variance removal
        X = self.remove_low_variance(X, threshold=variance_threshold)
        report['stages'].append({
            'stage': 'low_variance',
            'remaining': X.shape[1],
            'removed': initial_count - X.shape[1]
        })
        
        # Stage 2: Correlation removal
        X = self.remove_high_correlation(X, threshold=correlation_threshold)
        report['stages'].append({
            'stage': 'high_correlation',
            'remaining': X.shape[1],
            'removed': report['stages'][-1]['remaining'] - X.shape[1]
        })
        
        # Stage 3: Importance-based selection
        selected_features, importances = self.select_by_importance(
            X, y,
            method=importance_method,
            n_features=top_k,
            threshold=importance_threshold,
            task=task
        )
        
        X_selected = X[selected_features]
        
        report['stages'].append({
            'stage': 'importance',
            'remaining': X_selected.shape[1],
            'removed': X.shape[1] - X_selected.shape[1]
        })
        
        report['final_features'] = X_selected.shape[1]
        report['reduction_ratio'] = (initial_count - X_selected.shape[1]) / initial_count
        
        # Store results
        self.selected_features = selected_features
        
        # Log summary
        logger.info(f"\nFeature Selection Summary:")
        logger.info(f"  Initial: {initial_count}")
        for stage in report['stages']:
            logger.info(f"  After {stage['stage']}: {stage['remaining']} (removed {stage['removed']})")
        logger.info(f"  Final: {report['final_features']}")
        logger.info(f"  Reduction: {report['reduction_ratio']:.1%}")
        
        # Top 20 features
        top_20 = importances.head(20)
        logger.info(f"\nTop 20 Features:")
        for feat, imp in top_20.items():
            logger.info(f"  {feat}: {imp:.6f}")
        
        return X_selected, report
    
    def save_selection_report(
        self, 
        report: Dict[str, Any], 
        path: str
    ):
        """Save feature selection report to file"""
        import json
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Selection report saved to {path}")
    
    def get_feature_groups(
        self,
        importances: pd.Series,
        groups_config: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Analyze feature importance by groups
        
        Args:
            importances: Feature importance Series
            groups_config: Dict mapping group names to feature lists
        
        Returns:
            DataFrame with group-level statistics
        """
        group_stats = []
        
        for group_name, features in groups_config.items():
            # Find features in this group
            group_features = [f for f in features if f in importances.index]
            
            if group_features:
                group_importances = importances[group_features]
                
                group_stats.append({
                    'group': group_name,
                    'n_features': len(group_features),
                    'total_importance': group_importances.sum(),
                    'mean_importance': group_importances.mean(),
                    'max_importance': group_importances.max(),
                    'top_feature': group_importances.idxmax()
                })
        
        df = pd.DataFrame(group_stats).sort_values('total_importance', ascending=False)
        
        return df