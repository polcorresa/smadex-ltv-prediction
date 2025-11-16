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
    values: dict[TimeHorizon, np.ndarray]

    def __post_init__(self) -> None:
        """Validate predictions after initialization."""
        assert self.values, "Revenue predictions must contain at least one horizon"
        lengths = {len(arr) for arr in self.values.values()}
        assert len(lengths) == 1, "All horizon arrays must have identical length"
        self._length = lengths.pop()
        assert self._length > 0, "Prediction arrays must not be empty"
        for horizon, arr in self.values.items():
            assert np.all(arr >= 0), f"{horizon.display_name()} predictions must be non-negative"

    def available_horizons(self) -> list[TimeHorizon]:
        """Return available horizons sorted by increasing day count."""
        return sorted(self.values.keys(), key=lambda horizon: horizon.to_days())

    def has(self, horizon: TimeHorizon) -> bool:
        """Return True if the requested horizon is present."""
        return horizon in self.values

    def get(self, horizon: TimeHorizon) -> np.ndarray | None:
        """Return predictions for a given horizon, if available."""
        return self.values.get(horizon)

    def primary_horizon(self) -> TimeHorizon:
        """Return preferred horizon for downstream features (defaults to D7)."""
        if self.has(TimeHorizon.D7):
            return TimeHorizon.D7
        return self.available_horizons()[0]

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary for backward compatibility."""
        return {horizon.value: self.values[horizon] for horizon in self.values}

    def enforce_order_constraints(self) -> RevenuePredictions:
        """Ensure predictions are monotonically non-decreasing over time."""
        if len(self.values) <= 1:
            return self

        corrected = {horizon: arr.copy() for horizon, arr in self.values.items()}
        horizons = self.available_horizons()

        # Forward pass: ensure later horizons are at least previous horizon
        for idx in range(1, len(horizons)):
            prev_horizon = horizons[idx - 1]
            curr_horizon = horizons[idx]
            corrected[curr_horizon] = np.maximum(corrected[curr_horizon], corrected[prev_horizon])

        # Backward pass: ensure earlier horizons are not larger than later horizons
        for idx in range(len(horizons) - 2, -1, -1):
            curr_horizon = horizons[idx]
            next_horizon = horizons[idx + 1]
            corrected[curr_horizon] = np.minimum(corrected[curr_horizon], corrected[next_horizon])

        return RevenuePredictions(corrected)

    def gate_with_probability(
        self,
        probability: np.ndarray,
        *,
        power: float = 1.0,
        floor: float = 0.0,
        probability_cap: float = 1.0,
        probability_cutoff: float = 0.0
    ) -> RevenuePredictions:
        """Blend revenue predictions with zero using buyer probabilities as gates."""
        assert len(probability) == self.length, "Probability vector must align with predictions"
        assert power > 0.0, "Power must be positive"
        assert floor >= 0.0, "Floor must be non-negative"

        assert 0.0 < probability_cap <= 1.0, "probability_cap must be in (0, 1]"
        assert 0.0 <= probability_cutoff < 1.0, "probability_cutoff must be in [0, 1)"
        assert probability_cutoff < probability_cap + 1e-9, "cutoff must be below cap"

        clipped = np.clip(probability, 0.0, probability_cap)
        gating = np.where(
            clipped <= probability_cutoff,
            0.0,
            clipped ** power
        )
        gated = {
            horizon: np.maximum(floor, values * gating)
            for horizon, values in self.values.items()
        }
        return RevenuePredictions(gated)

    @property
    def length(self) -> int:
        """Return number of samples represented by the predictions."""
        return self._length


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
