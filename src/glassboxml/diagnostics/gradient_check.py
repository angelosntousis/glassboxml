"""Centered finite differences for checking derivatives of real functions.

For smooth functions, truncation error is O(eps**2), while floating-point
cancellation grows as eps shrinks. The default absolute step suits moderately
scaled float64 inputs; rescale parameters or tune eps for other scales. Large
constant offsets in the objective can also hide small changes. These checks are
unreliable at discontinuities, kinks, or domain boundaries, and require a
deterministic objective (including fixed random samples for stochastic models).
Each gradient costs two objective evaluations per parameter, so use small
problems. A small error is evidence of agreement, not a proof of correctness.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert real numeric data to finite float64 values without aliasing it."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def numerical_gradient(
    function: Callable[[NDArray[np.float64]], float | np.floating[Any]],
    parameters: ArrayLike,
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """Approximate a scalar objective's gradient using centered differences.

    Args:
        function: Deterministic callable returning a finite real scalar. Each
            call receives a fresh float64 parameter array of the original shape.
        parameters: Finite real parameters of any shape, including a scalar
            (zero-dimensional array). Integer inputs are converted to float64.
        eps: Positive, finite absolute perturbation applied to each coordinate.

    Returns:
        A float64 gradient with the same shape as parameters. Empty inputs
        return an empty gradient without calling the objective.

    Raises:
        ValueError: Inputs or objective results are invalid, a perturbation is
            unrepresentable, or the resulting derivative is not finite.

    The caller's parameters are never modified, even if the objective mutates
    its argument or raises. Objective exceptions propagate unchanged. No
    automatic step scaling is performed; see the module's numerical caveats.
    """
    step = _finite_real_array(eps, "eps")
    if step.ndim != 0 or step <= 0:
        raise ValueError("eps must be a positive finite scalar")
    epsilon = float(step)
    values = _finite_real_array(parameters, "parameters")
    gradient = np.empty_like(values)

    for index in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            plus[index] += epsilon
            minus[index] -= epsilon
        if (
            not np.isfinite(plus[index])
            or not np.isfinite(minus[index])
            or plus[index] == values[index]
            or minus[index] == values[index]
        ):
            raise ValueError("eps cannot produce finite, distinct perturbations")

        high = _finite_real_array(function(plus), "function result")
        low = _finite_real_array(function(minus), "function result")
        if high.ndim != 0 or low.ndim != 0:
            raise ValueError("function must return a scalar")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            difference = float(high) - float(low)
            denominator = 2.0 * epsilon
            if not np.isfinite(difference):
                # Only halve first when necessary: subnormal values can vanish.
                derivative = (float(high) * 0.5 - float(low) * 0.5) / epsilon
            elif not np.isfinite(denominator):
                derivative = (difference / epsilon) * 0.5
            else:
                derivative = difference / denominator
        if not np.isfinite(derivative):
            raise ValueError("numerical gradient is not finite; rescale the objective")
        gradient[index] = derivative

    return gradient


def relative_error(numerical: ArrayLike, analytical: ArrayLike) -> float:
    """Return ||numerical - analytical||₂ / (||numerical||₂ + ||analytical||₂).

    Inputs must be finite real arrays with exactly matching shapes; broadcasting
    is not allowed. Arrays are flattened for the Euclidean norm. Two zero (or
    empty) gradients have error zero. If only one gradient is zero, error is one.
    Scaling both inputs before taking norms avoids overflow and underflow for
    very large or small gradients without adding an arbitrary denominator floor.

    This global score can hide mistakes in small components when other
    components dominate. Near-zero gradients can also have large relative error
    despite negligible absolute error; inspect absolute differences as well.
    """
    numeric = _finite_real_array(numerical, "numerical gradient")
    analytic = _finite_real_array(analytical, "analytical gradient")
    if numeric.shape != analytic.shape:
        raise ValueError("gradient shapes must match exactly")
    scale = max(
        float(np.max(np.abs(numeric), initial=0.0)),
        float(np.max(np.abs(analytic), initial=0.0)),
    )
    if scale == 0.0:
        return 0.0
    numeric = (numeric / scale).ravel()
    analytic = (analytic / scale).ravel()
    return float(
        np.linalg.norm(numeric - analytic)
        / (np.linalg.norm(numeric) + np.linalg.norm(analytic))
    )
