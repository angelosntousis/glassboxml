"""Ordinary least squares with a stable least-squares solve."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from glassboxml.models.base import _LinearModel


class LinearRegression(_LinearModel):
    """Minimize sum((y - X @ coef - intercept)**2) in float64.

    ``fit_intercept=True`` (default) centers X and y before solving and recovers
    the intercept afterward. With False, the intercept is fixed at zero.
    Inputs are a nonempty 2D feature matrix and a matching 1D target vector.
    Multi-output regression is not supported. Input arrays are never modified.

    np.linalg.lstsq avoids explicit inversion and forming X.T @ X. For a
    rank-deficient or underdetermined design it returns the minimum-norm
    coefficient solution, using NumPy's default numerical rank threshold.
    Poorly scaled features still benefit from scaling before fitting.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit coefficients/intercept and return self; failed fits preserve state."""
        features, targets, x_offset, y_offset = self._prepare_fit(X, y)
        coef, _, _, _ = np.linalg.lstsq(features, targets, rcond=None)
        self._store_fit(coef, x_offset, y_offset)
        return self
