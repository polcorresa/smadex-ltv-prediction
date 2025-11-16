"""
Threshold Optimizer for Buyer Classification.

Optimizes the decision threshold for buyer probability to maximize
revenue prediction accuracy (RMSLE), taking into account that
predicted_revenue = P(buyer) * E(revenue|buyer).

Following ArjanCodes best practices:
- Complete type hints
- Precondition/postcondition assertions
- Clear error messages
"""
from __future__ import annotations

import numpy as np
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """Results from threshold optimization."""
    optimal_threshold: float
    optimal_metric: float
    threshold_range: np.ndarray
    metric_values: np.ndarray


def _compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Logarithmic Error.
    
    Args:
        y_true: True revenue values
        y_pred: Predicted revenue values
        
    Returns:
        RMSLE score
    """
    assert len(y_true) == len(y_pred), "Arrays must have same length"
    assert len(y_true) > 0, "Arrays must not be empty"
    assert np.all(y_true >= 0), "y_true must be non-negative"
    assert np.all(y_pred >= 0), "y_pred must be non-negative"
    
    result = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
    
    assert result >= 0.0, "RMSLE must be non-negative"
    return result


class BuyerThresholdOptimizer:
    """
    Optimizes buyer probability threshold to maximize revenue prediction accuracy.
    
    The final revenue prediction is: predicted_revenue = P(buyer) * E(revenue|buyer)
    
    This optimizer searches for the threshold that produces the best RMSLE when
    we set P(buyer) = 1 if buyer_proba >= threshold else 0, then multiply by
    the revenue regression prediction.
    """
    
    def __init__(
        self,
        metric: str = "rmsle",
        n_thresholds: int = 100,
        threshold_range: tuple[float, float] = (0.01, 0.99)
    ) -> None:
        """
        Initialize threshold optimizer.
        
        Args:
            metric: Optimization metric ("rmsle", "rmse", or "mae")
            n_thresholds: Number of threshold values to test
            threshold_range: (min, max) threshold range to search
        """
        assert metric in {"rmsle", "rmse", "mae"}, \
            "metric must be 'rmsle', 'rmse', or 'mae'"
        assert n_thresholds > 0, "n_thresholds must be positive"
        assert 0.0 < threshold_range[0] < threshold_range[1] < 1.0, \
            "threshold_range must be in (0, 1)"
        
        self.metric = metric
        self.n_thresholds = n_thresholds
        self.threshold_range = threshold_range
        self.optimal_threshold: Optional[float] = None
    
    def optimize(
        self,
        buyer_proba: np.ndarray,
        revenue_pred: np.ndarray,
        y_true_revenue: np.ndarray
    ) -> ThresholdResult:
        """
        Find optimal buyer threshold by grid search.
        
        Args:
            buyer_proba: Buyer probabilities from Stage 1 (0 to 1)
            revenue_pred: Revenue predictions from Stage 2 (for buyers only)
            y_true_revenue: True revenue values (ground truth)
            
        Returns:
            ThresholdResult with optimal threshold and metric curve
        """
        # Preconditions
        assert len(buyer_proba) == len(revenue_pred) == len(y_true_revenue), \
            "All arrays must have same length"
        assert len(buyer_proba) > 0, "Arrays must not be empty"
        assert np.all((buyer_proba >= 0) & (buyer_proba <= 1)), \
            "buyer_proba must be in [0, 1]"
        assert np.all(revenue_pred >= 0), "revenue_pred must be non-negative"
        assert np.all(y_true_revenue >= 0), "y_true_revenue must be non-negative"
        
        logger.info(f"Optimizing buyer threshold using {self.metric}...")
        
        # Grid search over thresholds
        thresholds = np.linspace(
            self.threshold_range[0],
            self.threshold_range[1],
            self.n_thresholds
        )
        
        metric_values = np.zeros(self.n_thresholds)
        
        for i, threshold in enumerate(thresholds):
            # Apply threshold: set buyer_proba to 1 if >= threshold, else 0
            binary_buyer = (buyer_proba >= threshold).astype(float)
            
            # Final revenue prediction: P(buyer) * E(revenue|buyer)
            final_pred = binary_buyer * revenue_pred
            
            # Compute metric
            if self.metric == "rmsle":
                metric_values[i] = _compute_rmsle(y_true_revenue, final_pred)
            elif self.metric == "rmse":
                metric_values[i] = np.sqrt(np.mean((y_true_revenue - final_pred) ** 2))
            elif self.metric == "mae":
                metric_values[i] = np.mean(np.abs(y_true_revenue - final_pred))
        
        # Find best threshold (minimize metric)
        best_idx = np.argmin(metric_values)
        self.optimal_threshold = float(thresholds[best_idx])
        optimal_metric = float(metric_values[best_idx])
        
        logger.info(
            f"Optimal threshold: {self.optimal_threshold:.4f} "
            f"({self.metric}={optimal_metric:.6f})"
        )
        
        # Log some context
        baseline_idx = np.argmin(np.abs(thresholds - 0.5))
        baseline_metric = float(metric_values[baseline_idx])
        improvement = ((baseline_metric - optimal_metric) / baseline_metric) * 100
        
        logger.info(
            f"Baseline (threshold=0.5): {self.metric}={baseline_metric:.6f}"
        )
        logger.info(f"Improvement: {improvement:+.2f}%")
        
        # Postconditions
        assert 0.0 < self.optimal_threshold < 1.0, \
            "Optimal threshold must be in (0, 1)"
        assert optimal_metric >= 0.0, "Metric must be non-negative"
        
        return ThresholdResult(
            optimal_threshold=self.optimal_threshold,
            optimal_metric=optimal_metric,
            threshold_range=thresholds,
            metric_values=metric_values
        )
    
    def optimize_continuous(
        self,
        buyer_proba: np.ndarray,
        revenue_pred: np.ndarray,
        y_true_revenue: np.ndarray
    ) -> ThresholdResult:
        """
        Alternative optimization: keep continuous probabilities but scale them.
        
        Instead of hard threshold, this finds an optimal scaling factor alpha
        such that final_pred = (buyer_proba ** alpha) * revenue_pred
        
        This maintains smoothness while allowing adjustment of buyer confidence.
        
        Args:
            buyer_proba: Buyer probabilities from Stage 1
            revenue_pred: Revenue predictions from Stage 2
            y_true_revenue: True revenue values
            
        Returns:
            ThresholdResult with optimal scaling factor
        """
        # Preconditions
        assert len(buyer_proba) == len(revenue_pred) == len(y_true_revenue), \
            "All arrays must have same length"
        assert len(buyer_proba) > 0, "Arrays must not be empty"
        assert np.all((buyer_proba >= 0) & (buyer_proba <= 1)), \
            "buyer_proba must be in [0, 1]"
        assert np.all(revenue_pred >= 0), "revenue_pred must be non-negative"
        assert np.all(y_true_revenue >= 0), "y_true_revenue must be non-negative"
        
        logger.info("Optimizing continuous probability scaling (alpha)...")
        
        # Search over alpha values (exponent)
        alphas = np.linspace(0.1, 3.0, self.n_thresholds)
        metric_values = np.zeros(self.n_thresholds)
        
        for i, alpha in enumerate(alphas):
            # Scale probabilities: (P(buyer))^alpha
            scaled_proba = buyer_proba ** alpha
            
            # Final prediction
            final_pred = scaled_proba * revenue_pred
            
            # Compute metric
            if self.metric == "rmsle":
                metric_values[i] = _compute_rmsle(y_true_revenue, final_pred)
            elif self.metric == "rmse":
                metric_values[i] = np.sqrt(np.mean((y_true_revenue - final_pred) ** 2))
            elif self.metric == "mae":
                metric_values[i] = np.mean(np.abs(y_true_revenue - final_pred))
        
        # Find best alpha
        best_idx = np.argmin(metric_values)
        optimal_alpha = float(alphas[best_idx])
        optimal_metric = float(metric_values[best_idx])
        
        logger.info(
            f"Optimal alpha: {optimal_alpha:.4f} "
            f"({self.metric}={optimal_metric:.6f})"
        )
        
        # Store as "threshold" for compatibility
        self.optimal_threshold = optimal_alpha
        
        # Log baseline (alpha=1.0)
        baseline_idx = np.argmin(np.abs(alphas - 1.0))
        baseline_metric = float(metric_values[baseline_idx])
        improvement = ((baseline_metric - optimal_metric) / baseline_metric) * 100
        
        logger.info(f"Baseline (alpha=1.0): {self.metric}={baseline_metric:.6f}")
        logger.info(f"Improvement: {improvement:+.2f}%")
        
        return ThresholdResult(
            optimal_threshold=optimal_alpha,
            optimal_metric=optimal_metric,
            threshold_range=alphas,
            metric_values=metric_values
        )
    
    def get_optimal_threshold(self) -> float:
        """
        Get optimized threshold.
        
        Returns:
            Optimal threshold value
        """
        assert self.optimal_threshold is not None, \
            "Must call optimize() before getting threshold"
        return self.optimal_threshold
