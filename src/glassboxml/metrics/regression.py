"""Single-target regression metrics with explicit shape and edge-case semantics."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _paired_targets(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Require matching nonempty 1D finite real vectors, without broadcasting."""
    arrays = []
    for name, value in (("y_true", y_true), ("y_pred", y_pred)):
        array = np.asarray(value)
        if array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numeric values")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(array, dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a nonempty 1D array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite float64 values")
        arrays.append(array)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("y_true and y_pred must have matching shapes")
    return arrays[0], arrays[1]


def _rms(values: NDArray[np.float64]) -> float:
    """Scale before squaring to avoid needless overflow/underflow."""
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    return float(scale * np.sqrt(np.mean((values / scale) ** 2)))


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return sqrt(mean((y_true - y_pred)**2)) for matching 1D vectors.

    Invalid inputs raise ValueError; unrepresentable residuals raise
    FloatingPointError. Scaling avoids overflow when squaring large residuals.
    """
    truth, prediction = _paired_targets(y_true, y_pred)
    with np.errstate(over="raise", invalid="raise"):
        return _rms(truth - prediction)


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean((y_true - y_pred)**2) for matching nonempty 1D vectors.

    Raise FloatingPointError if the mean squared error exceeds float64 range.
    """
    error = root_mean_squared_error(y_true, y_pred)
    with np.errstate(over="raise", invalid="raise"):
        return float(np.square(error))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return 1 - residual_sum_of_squares / total_sum_of_squares.

    At least two observations are required; otherwise raise ValueError.
    Constant targets score 1 for exact predictions and 0 otherwise, matching
    the common finite-score convention. Negative scores are valid and indicate
    performance worse than always predicting the target mean. No clipping is
    applied. Inputs must be matching finite 1D vectors.
    """
    truth, prediction = _paired_targets(y_true, y_pred)
    if truth.size < 2:
        raise ValueError("R^2 requires at least two observations")
    if np.all(truth == truth[0]):
        return 1.0 if np.array_equal(truth, prediction) else 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        # A common scale cancels in the ratio and keeps both the residuals and
        # the mean finite even when unscaled sums/differences would overflow.
        scale = max(float(np.max(np.abs(truth))), float(np.max(np.abs(prediction))))
        truth = truth / scale
        prediction = prediction / scale
        residual_rms = _rms(truth - prediction)
        target_rms = _rms(truth - np.mean(truth))
        return float(1.0 - np.square(np.float64(residual_rms) / target_rms))


# Short names are convenient in experiment scripts.
mse = mean_squared_error
rmse = root_mean_squared_error
