"""
Histogram-based Oversampling/Undersampling (HistOS/HistUS).

Based on: Aminian et al., 2025, Machine Learning Journal.

Following ArjanCodes best practices:
- Complete type hints
- Precondition/postcondition assertions
- Clear parameter validation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HistogramOversampling:
    """
    HistOS: Adaptive oversampling for imbalanced regression.
    
    Builds incremental histograms to identify rare cases.
    """
    
    def __init__(
        self,
        n_bins: int = 30,
        window_size: int = 10000,
        target_percentile: float = 75.0
    ) -> None:
        """
        Initialize HistOS sampler.
        
        Args:
            n_bins: Number of histogram bins
            window_size: Sliding window for incremental updates
            target_percentile: Target count per bin (percentile of histogram)
        """
        assert n_bins > 0, "n_bins must be positive"
        assert window_size > 0, "window_size must be positive"
        assert 0.0 < target_percentile < 100.0, \
            "target_percentile must be in (0, 100)"
        
        self.n_bins = n_bins
        self.window_size = window_size
        self.target_percentile = target_percentile
        
        self.histogram: Optional[np.ndarray] = None
        self.bin_edges: Optional[np.ndarray] = None
        self.window: deque = deque(maxlen=window_size)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> HistogramOversampling:
        """
        Build histogram from training data.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Self for method chaining
        """
        assert len(X) > 0, "X must not be empty"
        assert len(X) == len(y), "X and y must have same length"
        assert np.all(np.isfinite(y)), "y must contain only finite values"
        
        # Use target values to create bins
        self.bin_edges = np.percentile(
            y, 
            np.linspace(0, 100, self.n_bins + 1)
        )
        
        # Ensure unique edges
        self.bin_edges = np.unique(self.bin_edges)
        
        # Initialize histogram
        self.histogram = np.zeros(len(self.bin_edges) - 1)
        
        # Count samples per bin
        bin_indices = np.digitize(y, self.bin_edges)
        for i in range(1, len(self.bin_edges)):
            self.histogram[i-1] = np.sum(bin_indices == i)
        
        logger.info(f"HistOS histogram: {self.histogram}")
        
        return self
    
    def fit_resample(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and resample data.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        assert len(X) > 0, "X must not be empty"
        assert len(X) == len(y), "X and y must have same length"
        
        self.fit(X, y)
        
        # Determine target count per bin
        target_count = np.percentile(self.histogram, self.target_percentile)
        
        logger.info(f"HistOS target count per bin: {target_count}")
        
        # Oversample rare bins
        X_resampled = []
        y_resampled = []
        
        bin_indices = np.digitize(y, self.bin_edges)
        
        for i in range(len(X)):
            bin_idx = bin_indices[i] - 1  # digitize returns 1-indexed
            
            if bin_idx < 0 or bin_idx >= len(self.histogram):
                n_copies = 1
            else:
                bin_count = self.histogram[bin_idx]
                if bin_count > 0:
                    # Oversample rare bins
                    n_copies = max(1, int(target_count / bin_count))
                else:
                    n_copies = 1
            
            # Add copies
            for _ in range(n_copies):
                X_resampled.append(X[i])
                y_resampled.append(y[i])
        
        X_resampled = np.array(X_resampled)
        y_resampled = np.array(y_resampled)
        
        # Postconditions
        assert len(X_resampled) >= len(X), \
            "Resampled data should be at least as large as original"
        assert len(X_resampled) == len(y_resampled), \
            "Resampled X and y must have same length"
        
        logger.info(f"HistOS: {len(X)} -> {len(X_resampled)} samples")
        
        return X_resampled, y_resampled


class HistogramUndersampling:
    """
    HistUS: Adaptive undersampling for dominant classes.
    
    Reduces zero-revenue dominance.
    """
    
    def __init__(
        self,
        n_bins: int = 30,
        target_percentile: float = 25.0
    ) -> None:
        """
        Initialize HistUS sampler.
        
        Args:
            n_bins: Number of histogram bins
            target_percentile: Target count per bin (percentile of histogram)
        """
        assert n_bins > 0, "n_bins must be positive"
        assert 0.0 < target_percentile < 100.0, \
            "target_percentile must be in (0, 100)"
        
        self.n_bins = n_bins
        self.target_percentile = target_percentile
        
        self.histogram: Optional[np.ndarray] = None
        self.bin_edges: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> HistogramUndersampling:
        """
        Build histogram from training data.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Self for method chaining
        """
        assert len(X) > 0, "X must not be empty"
        assert len(X) == len(y), "X and y must have same length"
        assert np.all(np.isfinite(y)), "y must contain only finite values"
        
        # Use target values to create bins
        self.bin_edges = np.percentile(
            y,
            np.linspace(0, 100, self.n_bins + 1)
        )
        
        self.bin_edges = np.unique(self.bin_edges)
        
        # Initialize histogram
        self.histogram = np.zeros(len(self.bin_edges) - 1)
        
        # Count samples per bin
        bin_indices = np.digitize(y, self.bin_edges)
        for i in range(1, len(self.bin_edges)):
            self.histogram[i-1] = np.sum(bin_indices == i)
        
        return self
    
    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and undersample data.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        assert len(X) > 0, "X must not be empty"
        assert len(X) == len(y), "X and y must have same length"
        
        self.fit(X, y)
        
        # Determine target count per bin
        target_count = np.percentile(self.histogram, self.target_percentile)
        
        logger.info(f"HistUS target count per bin: {target_count}")
        
        # Undersample over-represented bins
        X_resampled = []
        y_resampled = []
        
        bin_indices = np.digitize(y, self.bin_edges)
        
        for bin_idx in range(1, len(self.bin_edges)):
            # Get samples in this bin
            mask = (bin_indices == bin_idx)
            X_bin = X[mask]
            y_bin = y[mask]
            
            bin_count = len(X_bin)
            
            if bin_count > target_count:
                # Undersample
                indices = np.random.choice(
                    bin_count,
                    size=int(target_count),
                    replace=False
                )
                X_resampled.append(X_bin[indices])
                y_resampled.append(y_bin[indices])
            else:
                # Keep all
                X_resampled.append(X_bin)
                y_resampled.append(y_bin)
        
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.concatenate(y_resampled)
        
        # Postconditions
        assert len(X_resampled) <= len(X), \
            "Resampled data should be at most as large as original"
        assert len(X_resampled) == len(y_resampled), \
            "Resampled X and y must have same length"
        
        logger.info(f"HistUS: {len(X)} -> {len(X_resampled)} samples")
        
        return X_resampled, y_resampled
