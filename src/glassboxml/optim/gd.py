"""Fixed-step gradient descent."""

import numpy as np

from glassboxml.optim.base import (
    ParameterDict,
    Parameters,
    _check_finite,
    _positive_float,
    _prepare_step,
)


class GradientDescent:
    """Apply ``parameter_new = parameter - learning_rate * gradient``.

    The caller supplies full-data gradients for batch gradient descent. This
    optimizer has no history. See Optimizer for input and mutation guarantees.
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = _positive_float(learning_rate, "learning_rate")

    def step(self, params: Parameters, grads: Parameters) -> ParameterDict:
        """Return a fresh dictionary of parameters after one descent step."""
        learning_rate = _positive_float(self.learning_rate, "learning_rate")
        parameters, gradients = _prepare_step(params, grads)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            updated = {
                name: np.asarray(
                    parameter - learning_rate * gradients[name], dtype=np.float64
                )
                for name, parameter in parameters.items()
            }
        _check_finite(updated)
        return updated

    def reset(self) -> None:
        """No-op: fixed-step gradient descent has no history."""
