"""L2-regularized least squares with an unpenalized intercept."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from glassboxml.models.base import _LinearModel


class RidgeRegression(_LinearModel):
    """Minimize sum((y - X @ coef - intercept)**2) + alpha * sum(coef**2).

    alpha is a finite, nonnegative scalar (default 1.0). The loss is a sum,
    not a mean: this convention matches the usual ridge penalty definition.
    alpha=0 reduces to ordinary least squares. ``fit_intercept=True`` centers
    X and y and fits an unregularized intercept; False fixes it at zero.

    Solve the augmented system [X_centered; sqrt(alpha)*I] against
    [y_centered; 0] using np.linalg.lstsq. This avoids the loss of conditioning
    from forming X.T @ X and handles collinear features. It allocates an
    additional n_features-by-n_features block, appropriate for small dense
    problems. Features are not automatically standardized, so their scales
    affect the penalty. Inputs follow LinearRegression's single-target API.
    """

    def __init__(self, alpha: float = 1.0, *, fit_intercept: bool = True) -> None:
        super().__init__(fit_intercept=fit_intercept)
        if isinstance(alpha, (bool, np.bool_)) or not isinstance(
            alpha, (int, float, np.integer, np.floating)
        ):
            raise ValueError("alpha must be a finite nonnegative number")
        try:
            value = float(alpha)
        except OverflowError as error:
            raise ValueError("alpha must be a finite nonnegative number") from error
        if not np.isfinite(value) or value < 0:
            raise ValueError("alpha must be a finite nonnegative number")
        self.alpha = value

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit coefficients/intercept and return self; failed fits preserve state."""
        features, targets, x_offset, y_offset = self._prepare_fit(X, y)
        if self.alpha > 0:
            n_features = features.shape[1]
            features = np.vstack((features, np.sqrt(self.alpha) * np.eye(n_features)))
            targets = np.concatenate((targets, np.zeros(n_features)))
        coef, _, _, _ = np.linalg.lstsq(features, targets, rcond=None)
        self._store_fit(coef, x_offset, y_offset)
        return self
