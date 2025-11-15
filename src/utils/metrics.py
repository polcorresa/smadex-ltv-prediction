"""
Evaluation metrics for LTV prediction.

Following ArjanCodes best practices:
- Complete type hints
- Precondition/postcondition assertions
- Dataclass for structured results
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.types import ModelMetrics


def msle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Logarithmic Error (competition metric).
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        MSLE score
    """
    assert len(y_true) == len(y_pred), "Arrays must have same length"
    assert len(y_true) > 0, "Arrays must not be empty"
    assert np.all(y_true >= 0), "y_true must be non-negative for MSLE"
    assert np.all(y_pred >= 0), "y_pred must be non-negative for MSLE"
    
    result = mean_squared_error(
        np.log1p(y_true),
        np.log1p(np.clip(y_pred, 0, None))
    )
    
    assert result >= 0.0, "MSLE must be non-negative"
    return float(result)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Logarithmic Error.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        RMSLE score
    """
    assert len(y_true) == len(y_pred), "Arrays must have same length"
    assert len(y_true) > 0, "Arrays must not be empty"
    assert np.all(y_true >= 0), "y_true must be non-negative for RMSLE"
    assert np.all(y_pred >= 0), "y_pred must be non-negative for RMSLE"
    
    result = np.sqrt(msle(y_true, y_pred))
    
    assert result >= 0.0, "RMSLE must be non-negative"
    return float(result)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        RMSE score
    """
    assert len(y_true) == len(y_pred), "Arrays must have same length"
    assert len(y_true) > 0, "Arrays must not be empty"
    
    result = np.sqrt(mean_squared_error(y_true, y_pred))
    
    assert result >= 0.0, "RMSE must be non-negative"
    return float(result)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        MAPE score (percentage)
    """
    assert len(y_true) == len(y_pred), "Arrays must have same length"
    assert len(y_true) > 0, "Arrays must not be empty"
    
    mask = y_true != 0
    if not mask.any():
        return 0.0
    
    result = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    assert result >= 0.0, "MAPE must be non-negative"
    return float(result)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    """
    Comprehensive evaluation using structured dataclass.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        ModelMetrics dataclass with all evaluation metrics
    """
    assert len(y_true) == len(y_pred), "Arrays must have same length"
    assert len(y_true) > 0, "Arrays must not be empty"
    assert np.all(np.isfinite(y_true)), "y_true must contain only finite values"
    assert np.all(np.isfinite(y_pred)), "y_pred must contain only finite values"
    
    return ModelMetrics(
        msle=msle(y_true, y_pred),
        rmsle=rmsle(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
        mape=mape(y_true, y_pred)
    )