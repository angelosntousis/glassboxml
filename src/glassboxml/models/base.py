"""Shared validation and prediction for single-target linear estimators."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Copy real numeric input into a finite float64 array."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite float64 values")
    return result


def _feature_matrix(X: ArrayLike) -> NDArray[np.float64]:
    """Require explicit sample and feature dimensions, including one feature."""
    features = _real_array(X, "X")
    if features.ndim != 2 or 0 in features.shape:
        raise ValueError("X must have shape (n_samples, n_features), both nonzero")
    return features


class _LinearModel:
    """Shared storage and prediction; subclasses provide their fitting equation."""

    def __init__(self, *, fit_intercept: bool = True) -> None:
        if not isinstance(fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be boolean")
        self.fit_intercept = bool(fit_intercept)
        self._coef: NDArray[np.float64] | None = None
        self._intercept: float | None = None

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Return an independent copy of fitted coefficients, shape (n_features,)."""
        if self._coef is None:
            raise RuntimeError("Call fit before accessing coefficients")
        return self._coef.copy()

    @property
    def intercept_(self) -> float:
        """Return the fitted intercept, or zero when fit_intercept=False."""
        if self._intercept is None:
            raise RuntimeError("Call fit before accessing the intercept")
        return self._intercept

    def _prepare_fit(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
        """Validate a single-target problem and optionally center both arrays."""
        features = _feature_matrix(X)
        targets = _real_array(y, "y")
        if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
            raise ValueError("y must have shape (n_samples,) matching X")
        x_offset = np.zeros(features.shape[1], dtype=np.float64)
        y_offset = 0.0
        if self.fit_intercept:
            with np.errstate(over="raise", invalid="raise"):
                x_offset = np.mean(features, axis=0)
                y_offset = float(np.mean(targets))
                features = features - x_offset
                targets = targets - y_offset
        return features, targets, x_offset, y_offset

    def _store_fit(
        self, coef: NDArray[np.float64], x_offset: NDArray[np.float64], y_offset: float
    ) -> None:
        """Commit a complete finite fit, leaving previous state intact on failure."""
        with np.errstate(over="raise", invalid="raise"):
            intercept = float(y_offset - x_offset @ coef)
        if not np.all(np.isfinite(coef)) or not np.isfinite(intercept):
            raise FloatingPointError("nonfinite fit; rescale the data")
        self._coef, self._intercept = coef.copy(), intercept

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict a 1D target vector for a nonempty 2D feature matrix.

        Requires the fitted number of features. Invalid input raises ValueError;
        an unfitted model raises RuntimeError. Inputs are never modified.
        """
        if self._coef is None:
            raise RuntimeError("Call fit before predict")
        features = _feature_matrix(X)
        if features.shape[1] != self._coef.size:
            raise ValueError(
                "X must have the same number of features as the fitted data"
            )
        with np.errstate(over="raise", invalid="raise"):
            predictions = features @ self._coef + self.intercept_
        if not np.all(np.isfinite(predictions)):
            raise FloatingPointError("nonfinite predictions; rescale the data")
        return np.asarray(predictions, dtype=np.float64)
