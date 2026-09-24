"""Monomial features for a single real-valued input variable."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


class PolynomialFeatures:
    """Map x to [x, x**2, ..., x**degree], optionally prepending ones.

    Args:
        degree: Nonnegative integer maximum power (default 2).
        include_bias: Include a constant column. Defaults to False because
            GlassBoxML regression models fit their own intercept. Degree zero
            is supported only when include_bias=True.

    Accepts shapes (n_samples,) and (n_samples, 1), returning a fresh float64
    matrix. This is a stateless, one-dimensional feature map: fit only validates
    the input, and transform may be used without fitting. No centering or
    scaling is learned from the data, so there is no preprocessing leakage.

    High powers can overflow and monomial columns become ill-conditioned at
    high degree even on [-1, 1]. Scale the input domain sensibly and consider
    ridge regularization. This transformer does not silently rescale features;
    their scale determines the meaning of a ridge coefficient penalty.
    """

    def __init__(self, degree: int = 2, *, include_bias: bool = False) -> None:
        if (
            isinstance(degree, (bool, np.bool_))
            or not isinstance(degree, (int, np.integer))
            or degree < 0
        ):
            raise ValueError("degree must be a nonnegative integer")
        if not isinstance(include_bias, (bool, np.bool_)):
            raise ValueError("include_bias must be boolean")
        if degree == 0 and not include_bias:
            raise ValueError("degree zero requires include_bias=True")
        self.degree = int(degree)
        self.include_bias = bool(include_bias)

    @staticmethod
    def _validate_input(X: ArrayLike) -> NDArray[np.float64]:
        """Validate one nonempty real-valued feature and return a copied vector."""
        values = np.asarray(X)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                "X must have nonempty shape (n_samples,) or (n_samples, 1)"
            )
        if values.dtype.kind not in "iuf":
            raise ValueError("X must contain real numeric values")
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.array(values, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(result)):
            raise ValueError("X must contain finite float64 values")
        return result

    def fit(self, X: ArrayLike) -> Self:
        """Validate the single input feature and return self; no state is learned."""
        self._validate_input(X)
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return ascending powers without changing X; raise on numeric overflow."""
        values = self._validate_input(X)
        start = 0 if self.include_bias else 1
        powers = np.arange(start, self.degree + 1, dtype=np.int64)
        with np.errstate(over="raise", invalid="raise"):
            return np.asarray(values[:, None] ** powers, dtype=np.float64)

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Validate and transform in one pass (equivalent to fit then transform)."""
        return self.transform(X)
