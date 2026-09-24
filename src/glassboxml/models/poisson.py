"""Analytical and numerical maximum likelihood for independent Poisson counts.

For theta = log(rate), the mean log likelihood, up to a constant, is
mean(x) * theta - exp(theta). Its derivative is mean(x) - exp(theta).
The numerical method follows this derivative without calling an optimizer.
"""

from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln  # type: ignore[import-untyped]


def _validate_counts(x: ArrayLike) -> NDArray[np.float64]:
    """Validate a nonempty vector of finite, nonnegative integer counts."""
    values = np.asarray(x)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a nonempty one-dimensional array of counts")
    if values.dtype.kind not in "iuf":
        raise ValueError("x must contain real numeric counts")
    counts = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(counts)):
        raise ValueError("x must contain only finite counts")
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("x must contain nonnegative integer counts")
    return counts


def _objective_and_gradient(theta: float, mean: float) -> tuple[float, float]:
    """Return mean log likelihood (without constants) and its theta derivative."""
    rate = float(np.exp(theta))
    return mean * theta - rate, mean - rate


class PoissonMLE:
    """Estimate a single Poisson rate from independent count observations.

    Args:
        method: ``closed_form`` uses the sample mean; ``gradient`` maximizes
            the log likelihood in unconstrained log-rate space.
        tol: Positive finite relative score tolerance for numerical fitting.
        max_iter: Positive maximum number of gradient updates.

    ``fit`` accepts a nonempty one-dimensional array (or list) of finite,
    nonnegative integer counts, including integer-valued floats. All-zero
    training samples are rejected: their MLE is zero, and no finite log-rate
    maximizer exists. Every successful fit therefore has a strictly positive
    ``rate_``. All-zero evaluation samples are allowed by ``log_likelihood``.

    Computation uses float64. Very large counts can lose integer precision;
    likelihood evaluation can suffer cancellation between large terms. These
    limitations also affect finite-difference checks of the full likelihood.
    """

    def __init__(
        self,
        method: Literal["closed_form", "gradient"] = "closed_form",
        *,
        tol: float = 1e-10,
        max_iter: int = 1000,
    ) -> None:
        if method not in ("closed_form", "gradient"):
            raise ValueError("method must be 'closed_form' or 'gradient'")
        if isinstance(tol, bool) or not np.isscalar(tol):
            raise ValueError("tol must be a positive finite number")
        if not isinstance(tol, (int, float, np.integer, np.floating)):
            raise ValueError("tol must be a positive finite number")
        try:
            tolerance = float(tol)
        except (OverflowError, ValueError) as error:
            raise ValueError("tol must be a positive finite number") from error
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tol must be a positive finite number")
        if (
            isinstance(max_iter, bool)
            or not isinstance(max_iter, (int, np.integer))
            or max_iter <= 0
        ):
            raise ValueError("max_iter must be a positive integer")
        self.method = method
        self.tol = tolerance
        self.max_iter = int(max_iter)
        self._rate: float | None = None

    @property
    def rate_(self) -> float:
        """Fitted rate; raise RuntimeError if no fit has succeeded."""
        if self._rate is None:
            raise RuntimeError("Call fit before accessing the fitted rate")
        return self._rate

    def fit(self, x: ArrayLike) -> Self:
        """Fit the rate and return self, preserving an earlier fit on failure.

        The numerical method starts at theta=0 (rate=1) and takes gradient
        ascent steps with step size 1 / max(mean(x), exp(theta)). The resulting
        theta update has magnitude at most one and approaches the optimum
        monotonically. Scaling the mean likelihood makes updates independent
        of sample size. Convergence requires abs(mean-rate) / mean <= tol.

        Raises ValueError for invalid or all-zero data, and RuntimeError when
        the numerical fit does not converge within max_iter updates.
        """
        counts = _validate_counts(x)
        with np.errstate(over="ignore"):
            mean = float(np.mean(counts))
        if not np.isfinite(mean):
            raise ValueError("counts are too large to compute a finite sample mean")
        if mean == 0:
            raise ValueError("all-zero samples have no strictly positive Poisson MLE")

        if self.method == "closed_form":
            rate = mean
        else:
            theta = 0.0
            for iteration in range(self.max_iter + 1):
                rate = float(np.exp(theta))
                _, gradient = _objective_and_gradient(theta, mean)
                if abs(gradient) / mean <= self.tol:
                    break
                if iteration == self.max_iter:
                    raise RuntimeError("Poisson gradient fit did not converge")
                theta += gradient / max(mean, rate)

        self._rate = rate
        return self

    def log_likelihood(self, x: ArrayLike) -> float:
        """Return the summed log likelihood, including log-factorial terms.

        Uses gammaln(x + 1) instead of explicitly computing factorials. The
        observations may differ from the training data. Requires a fitted model
        and the same count validation as fit, but permits all-zero samples.
        Raises ValueError if float64 arithmetic cannot produce a finite result.
        """
        rate = self.rate_
        counts = _validate_counts(x)
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(np.sum(counts * np.log(rate) - rate - gammaln(counts + 1)))
        if not np.isfinite(result):
            raise ValueError("log likelihood is not finite within float64 limits")
        return result
