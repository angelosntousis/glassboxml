"""Analytical Gaussian linear regression using only NumPy linear algebra."""

from typing import Literal, Self, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from glassboxml.models.base import _feature_matrix, _real_array


def _positive_precision(value: float, name: str) -> float:
    """Require a finite, strictly positive real scalar precision."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a positive finite precision")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a positive finite precision") from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite precision")
    return result


class BayesianLinearRegression:
    r"""Fit a Gaussian posterior with fixed prior and observation precisions.

    The model and conjugate posterior are

    .. math::
        y = \Phi w + \epsilon, \quad \epsilon \sim N(0, \beta^{-1}I),
        \quad w \sim N(0, \alpha^{-1}I),

        S_N^{-1} = \alpha I + \beta \Phi^T\Phi,
        \quad m_N = \beta S_N\Phi^T y,
        \quad p(w\mid\Phi,y) = N(m_N,S_N).

    For a new design vector phi, the posterior predictive distribution is

    .. math::
        p(y_*\mid\phi,\Phi,y)
        = N(\phi^T m_N,\; \underbrace{\phi^T S_N\phi}_{epistemic}
          + \underbrace{\beta^{-1}}_{observation\ noise}).

    Args:
        alpha: Positive prior precision, fixed rather than estimated (default 1).
        beta: Positive observation-noise precision, fixed rather than estimated
            (default 1). Observation variance is 1/beta.

    X is the explicit design matrix Phi, shape (n_samples, n_features); y is a
    matching 1D vector. No intercept is silently added or centered out. Include
    a column of ones for a bias weight; its prior is the same as every other
    weight's. PolynomialFeatures(include_bias=True) provides such a design.

    Calculations use float64. An augmented QR factorization avoids forming a
    Gram matrix or calling matrix inversion. A positive prior handles collinear
    and underdetermined designs, but sensible feature scaling still matters.
    Unrepresentable results raise FloatingPointError. Inputs are not modified;
    posterior properties return copies. Parameter changes affect the next fit,
    not predictions from the already fitted posterior.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = _positive_precision(alpha, "alpha")
        self.beta = _positive_precision(beta, "beta")
        self._mean: NDArray[np.float64] | None = None
        self._covariance: NDArray[np.float64] | None = None
        self._covariance_factor: NDArray[np.float64] | None = None
        self._noise_std: float | None = None

    @property
    def posterior_mean_(self) -> NDArray[np.float64]:
        """Return m_N, shape (n_features,), as an independent array."""
        if self._mean is None:
            raise RuntimeError("Call fit before accessing the posterior")
        return self._mean.copy()

    @property
    def posterior_covariance_(self) -> NDArray[np.float64]:
        """Return S_N, shape (n_features, n_features), as an independent array."""
        if self._covariance is None:
            raise RuntimeError("Call fit before accessing the posterior")
        return self._covariance.copy()

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Return the posterior mean weights, an alias for posterior_mean_."""
        return self.posterior_mean_

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        r"""Compute the exact posterior and return self, preserving state on failure.

        Factor A = [sqrt(beta)*Phi; sqrt(alpha)*I] = Q R. Then
        R.T R = S_N^{-1}. Solving R m_N = Q.T [sqrt(beta)*y; 0]
        gives the posterior mean. Solve R C = I and use S_N = C C.T;
        the same factor provides nonnegative predictive variances.
        """
        alpha = _positive_precision(self.alpha, "alpha")
        beta = _positive_precision(self.beta, "beta")
        features = _feature_matrix(X)
        targets = _real_array(y, "y")
        if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
            raise ValueError("y must have shape (n_samples,) matching X")
        n_features = features.shape[1]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            identity = np.eye(n_features)
            design = np.vstack((np.sqrt(beta) * features, np.sqrt(alpha) * identity))
            response = np.concatenate((np.sqrt(beta) * targets, np.zeros(n_features)))
            q, r = np.linalg.qr(design, mode="reduced")
            mean = np.linalg.solve(r, q.T @ response)
            factor = np.linalg.solve(r, identity)
            covariance = factor @ factor.T
            covariance = 0.5 * covariance + 0.5 * covariance.T
            noise_std = float(1.0 / np.sqrt(beta))
        if not all(np.all(np.isfinite(a)) for a in (mean, factor, covariance)):
            raise FloatingPointError(
                "nonfinite posterior; rescale the data or precisions"
            )
        self._mean, self._covariance = mean, covariance
        self._covariance_factor, self._noise_std = factor, noise_std
        return self

    @overload
    def predict(
        self, X: ArrayLike, return_std: Literal[False] = False
    ) -> NDArray[np.float64]: ...

    @overload
    def predict(
        self, X: ArrayLike, return_std: Literal[True]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    @overload
    def predict(
        self, X: ArrayLike, return_std: bool
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]: ...

    def predict(
        self, X: ArrayLike, return_std: bool = False
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Return predictive means, optionally with total standard deviations.

        Both outputs have shape (n_samples,). The returned standard deviation is
        sqrt(phi.T S_N phi + 1/beta), including future observation noise. It is
        not a standard error of the mean. All precisions are conditional on the
        fitted settings; hyperparameter uncertainty is not included.
        """
        if (
            self._mean is None
            or self._covariance_factor is None
            or self._noise_std is None
        ):
            raise RuntimeError("Call fit before predict")
        if not isinstance(return_std, (bool, np.bool_)):
            raise ValueError("return_std must be boolean")
        features = _feature_matrix(X)
        if features.shape[1] != self._mean.size:
            raise ValueError(
                "X must have the same number of features as the fitted data"
            )
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean = np.asarray(features @ self._mean, dtype=np.float64)
            if not np.all(np.isfinite(mean)):
                raise FloatingPointError("nonfinite predictions; rescale X")
            if not return_std:
                return mean
            projected = features @ self._covariance_factor
            if not np.all(np.isfinite(projected)):
                raise FloatingPointError("nonfinite predictive uncertainty; rescale X")
            # Rowwise norms with scaling avoid overflowing squared projections.
            scale = np.max(np.abs(projected), axis=1)
            scaled = projected / np.where(scale > 0, scale, 1.0)[:, None]
            epistemic_std = scale * np.sqrt(np.sum(scaled**2, axis=1))
            std = np.asarray(np.hypot(epistemic_std, self._noise_std), dtype=np.float64)
            if not np.all(np.isfinite(std)):
                raise FloatingPointError("nonfinite predictive uncertainty; rescale X")
        return mean, std
