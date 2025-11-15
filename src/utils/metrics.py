"""
Evaluation metrics for LTV prediction
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def msle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Logarithmic Error (competition metric)"""
    return mean_squared_error(
        np.log1p(y_true),
        np.log1p(np.clip(y_pred, 0, None))
    )


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error"""
    return np.sqrt(msle(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Comprehensive evaluation"""
    return {
        'msle': msle(y_true, y_pred),
        'rmsle': rmsle(y_true, y_pred),
        'rmse': mean_squared_error(y_true, y_pred, squared=False),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mape(y_true, y_pred)
    }