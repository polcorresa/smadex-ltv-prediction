"""
Common types, enums, and dataclasses for the Smadex LTV prediction project.

This module provides type-safe alternatives to magic strings and plain dicts,
following ArjanCodes best practices.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Dict, Any
import numpy as np


class TimeHorizon(Enum):
    """Time horizons for revenue prediction."""
    D1 = "d1"
    D7 = "d7"
    D14 = "d14"
    D28 = "d28"
    
    def display_name(self) -> str:
        """Return human-readable name."""
        return self.name.replace('_', ' ').title()
    
    def to_days(self) -> int:
        """Convert horizon to number of days."""
        mapping = {
            TimeHorizon.D1: 1,
            TimeHorizon.D7: 7,
            TimeHorizon.D14: 14,
            TimeHorizon.D28: 28
        }
        return mapping[self]


class DataSplit(IntEnum):
    """Dataset split types."""
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class TaskType(Enum):
    """Machine learning task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ModelType(Enum):
    """Model types for feature selection and training."""
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    ELASTIC_NET = "elastic_net"
    LASSO = "lasso"


class ImputationStrategy(Enum):
    """Missing value imputation strategies."""
    ZERO = "zero"
    MEDIAN = "median"
    MODE = "mode"
    LARGE_VALUE = "large_value"
    SPECIAL_VALUE = "special_value"
    FORWARD_FILL = "forward_fill"
    MISSING_FLAG = "missing_flag"


@dataclass
class ModelMetrics:
    """Evaluation metrics for model performance."""
    rmsle: float
    rmse: float
    mae: float
    r2: float
    msle: float
    mape: float
    
    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        assert self.rmsle >= 0.0, "RMSLE must be non-negative"
        assert self.rmse >= 0.0, "RMSE must be non-negative"
        assert self.mae >= 0.0, "MAE must be non-negative"
        assert self.msle >= 0.0, "MSLE must be non-negative"
        assert self.mape >= 0.0, "MAPE must be non-negative"
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'msle': self.msle,
            'rmsle': self.rmsle,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'mape': self.mape
        }


@dataclass
class RevenuePredictions:
    """Multi-horizon revenue predictions from ODMN model."""
    d1: np.ndarray
    d7: np.ndarray
    d14: np.ndarray
    
    def __post_init__(self) -> None:
        """Validate predictions after initialization."""
        assert len(self.d1) == len(self.d7) == len(self.d14), \
            "All horizons must have same length"
        assert len(self.d1) > 0, "Predictions must not be empty"
        assert np.all(self.d1 >= 0), "D1 predictions must be non-negative"
        assert np.all(self.d7 >= 0), "D7 predictions must be non-negative"
        assert np.all(self.d14 >= 0), "D14 predictions must be non-negative"
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for backward compatibility."""
        return {
            TimeHorizon.D1.value: self.d1,
            TimeHorizon.D7.value: self.d7,
            TimeHorizon.D14.value: self.d14
        }
    
    def enforce_order_constraints(self) -> RevenuePredictions:
        """Ensure D1 ≤ D7 ≤ D14 ordering."""
        d1_corrected = np.minimum(self.d1, self.d7)
        d7_corrected = np.clip(self.d7, self.d1, self.d14)
        d14_corrected = np.maximum(self.d14, self.d7)
        
        return RevenuePredictions(
            d1=d1_corrected,
            d7=d7_corrected,
            d14=d14_corrected
        )


@dataclass
class FeatureSelectionReport:
    """Report from feature selection process."""
    initial_features: int
    final_features: int
    reduction_ratio: float
    stages: list[Dict[str, Any]]
    selected_features: list[str]
    
    def __post_init__(self) -> None:
        """Validate report after initialization."""
        assert self.initial_features >= self.final_features, \
            "Final features cannot exceed initial features"
        assert 0.0 <= self.reduction_ratio <= 1.0, \
            "Reduction ratio must be in [0, 1]"
        assert self.final_features == len(self.selected_features), \
            "Final feature count must match selected features list"


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    n_estimators: int
    learning_rate: float
    max_depth: int
    num_leaves: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    random_state: int
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        assert self.n_estimators > 0, "n_estimators must be positive"
        assert 0.0 < self.learning_rate <= 1.0, \
            "learning_rate must be in (0, 1]"
        assert self.max_depth > 0, "max_depth must be positive"
        assert self.num_leaves > 1, "num_leaves must be > 1"
        assert self.min_child_samples >= 1, \
            "min_child_samples must be >= 1"
        assert 0.0 < self.subsample <= 1.0, "subsample must be in (0, 1]"
        assert 0.0 < self.colsample_bytree <= 1.0, \
            "colsample_bytree must be in (0, 1]"


# Constants for commonly used thresholds
class Threshold:
    """Common threshold values used throughout the codebase."""
    CORRELATION_HIGH: float = 0.95
    CORRELATION_MEDIUM: float = 0.7
    VARIANCE_LOW: float = 0.01
    MISSING_VALUE_SIGNIFICANCE: float = 0.05
    IMPORTANCE_MINIMUM: float = 0.005
    PROBABILITY_EPSILON: float = 1e-9
