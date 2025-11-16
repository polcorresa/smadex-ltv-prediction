"""Probability calibration utilities for buyer classifiers.

Provides Platt scaling, isotonic regression, and beta calibration
following ArjanCodes-inspired safety/clarity guidelines.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CalibrationMethod = Literal["none", "platt", "isotonic", "beta"]


@dataclass(slots=True)
class CalibrationSettings:
    """Configuration for probability calibration."""

    method: CalibrationMethod = "none"
    regularization: float = 1e-3
    isotonic_out_of_bounds: str = "clip"
    epsilon: float = 1e-6
    max_iter: int = 200

    def __post_init__(self) -> None:
        allowed = {"none", "platt", "isotonic", "beta"}
        assert self.method in allowed, f"Unsupported calibration method: {self.method}"
        assert self.regularization >= 0.0, "regularization must be non-negative"
        assert self.epsilon > 0.0, "epsilon must be positive"
        assert self.max_iter > 0, "max_iter must be positive"
        if self.isotonic_out_of_bounds not in {"clip", "nan", "raise"}:
            raise ValueError("isotonic_out_of_bounds must be clip|nan|raise")


class ProbabilityCalibrator:
    """Flexible probability calibration helper."""

    def __init__(self, settings: CalibrationSettings | None = None) -> None:
        self.settings = settings or CalibrationSettings()
        self._model: Any = None

    @property
    def method(self) -> CalibrationMethod:
        return self.settings.method

    @property
    def is_fitted(self) -> bool:
        return self.settings.method == "none" or self._model is not None

    def fit(self, probabilities: np.ndarray, targets: np.ndarray) -> None:
        """Fit calibrator on raw probabilities and binary targets."""
        assert len(probabilities) == len(targets), "probabilities and targets must align"
        assert len(probabilities) > 0, "Cannot fit calibrator on empty arrays"
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0)), \
            "Probabilities must lie in [0, 1]"
        unique_targets = np.unique(targets)
        assert set(unique_targets).issubset({0, 1}), "Targets must be binary"
        if len(unique_targets) < 2:
            # Degenerate case: fallback to identity to avoid crashes.
            self._model = None
            self.settings = CalibrationSettings(method="none")
            return

        method = self.settings.method
        if method == "none":
            self._model = None
            return

        if method == "isotonic":
            isotonic = IsotonicRegression(out_of_bounds=self.settings.isotonic_out_of_bounds)
            isotonic.fit(probabilities, targets)
            self._model = isotonic
            return

        if method == "platt":
            logistic = LogisticRegression(
                penalty="l2",
                C=1.0 / max(self.settings.regularization, 1e-12),
                solver="lbfgs",
                max_iter=self.settings.max_iter,
                class_weight="balanced"
            )
            logistic.fit(probabilities.reshape(-1, 1), targets)
            self._model = logistic
            return

        if method == "beta":
            eps = self.settings.epsilon
            clipped = np.clip(probabilities, eps, 1.0 - eps)
            features = np.column_stack([
                np.log(clipped),
                np.log1p(-clipped)
            ])
            beta_model = LogisticRegression(
                penalty="l2",
                C=1.0 / max(self.settings.regularization, 1e-12),
                solver="lbfgs",
                max_iter=self.settings.max_iter,
                class_weight="balanced"
            )
            beta_model.fit(features, targets)
            self._model = beta_model
            return

        raise ValueError(f"Unsupported calibration method: {method}")

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibrated mapping to probability array."""
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0)), \
            "Probabilities must lie in [0, 1]"
        if self.settings.method == "none" or self._model is None:
            return np.clip(probabilities, 0.0, 1.0)

        if self.settings.method == "isotonic":
            calibrated = self._model.predict(probabilities)
        elif self.settings.method == "platt":
            calibrated = self._model.predict_proba(probabilities.reshape(-1, 1))[:, 1]
        elif self.settings.method == "beta":
            eps = self.settings.epsilon
            clipped = np.clip(probabilities, eps, 1.0 - eps)
            features = np.column_stack([
                np.log(clipped),
                np.log1p(-clipped)
            ])
            calibrated = self._model.predict_proba(features)[:, 1]
        else:
            raise ValueError(f"Unsupported calibration method: {self.settings.method}")

        return np.clip(calibrated, 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        """Persist calibrator to disk."""
        path_obj = Path(path)
        payload = {
            "settings": asdict(self.settings),
            "model": self._model
        }
        joblib.dump(payload, path_obj)

    @classmethod
    def load(cls, path: str | Path) -> ProbabilityCalibrator:
        """Load calibrator from disk."""
        path_obj = Path(path)
        payload = joblib.load(path_obj)
        settings = CalibrationSettings(**payload["settings"])
        calibrator = cls(settings)
        calibrator._model = payload.get("model")
        return calibrator

