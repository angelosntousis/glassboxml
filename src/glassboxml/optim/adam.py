"""Adam with bias-corrected first and second gradient moments."""

import numpy as np

from glassboxml.optim.base import (
    ParameterDict,
    Parameters,
    _check_finite,
    _decay,
    _positive_float,
    _prepare_step,
)


class Adam:
    """Apply Adam updates, starting with zero moments and step count zero.

    At successful step t (starting at 1), for each parameter::

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        parameter_new = parameter - learning_rate * m_hat / (sqrt(v_hat) + eps)

    Both beta coefficients must be in [0, 1); eps is positive and sits outside
    the square root. Very large gradients can overflow when squared; such a
    step raises FloatingPointError without changing history. Rescale the loss
    in that case. See Optimizer for input and mutation guarantees.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.learning_rate = _positive_float(learning_rate, "learning_rate")
        self.beta1 = _decay(beta1, "beta1")
        self.beta2 = _decay(beta2, "beta2")
        self.eps = _positive_float(eps, "eps")
        self._first: ParameterDict | None = None
        self._second: ParameterDict = {}
        self._steps = 0

    def step(self, params: Parameters, grads: Parameters) -> ParameterDict:
        """Return updated parameters, committing moments only on success."""
        learning_rate = _positive_float(self.learning_rate, "learning_rate")
        beta1 = _decay(self.beta1, "beta1")
        beta2 = _decay(self.beta2, "beta2")
        eps = _positive_float(self.eps, "eps")
        parameters, gradients = _prepare_step(params, grads, self._first)
        first: ParameterDict = {}
        second: ParameterDict = {}
        updated: ParameterDict = {}
        step = self._steps + 1
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for name, parameter in parameters.items():
                old_first = (
                    np.zeros_like(parameter)
                    if self._first is None
                    else self._first[name]
                )
                old_second = (
                    np.zeros_like(parameter)
                    if self._first is None
                    else self._second[name]
                )
                gradient = gradients[name]
                first[name] = np.asarray(
                    beta1 * old_first + (1 - beta1) * gradient,
                    dtype=np.float64,
                )
                second[name] = np.asarray(
                    beta2 * old_second + (1 - beta2) * gradient**2,
                    dtype=np.float64,
                )
                first_corrected = first[name] / (1 - beta1**step)
                second_corrected = second[name] / (1 - beta2**step)
                updated[name] = np.asarray(
                    parameter
                    - learning_rate
                    * first_corrected
                    / (np.sqrt(second_corrected) + eps),
                    dtype=np.float64,
                )
        _check_finite(first, second, updated)
        self._first, self._second, self._steps = first, second, step
        return updated

    def reset(self) -> None:
        """Clear moments, step count, and names/shapes for a new training run."""
        self._first = None
        self._second = {}
        self._steps = 0
