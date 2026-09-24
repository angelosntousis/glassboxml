"""Gradient descent with classical (non-Nesterov) momentum."""

import numpy as np

from glassboxml.optim.base import (
    ParameterDict,
    Parameters,
    _check_finite,
    _decay,
    _positive_float,
    _prepare_step,
)


class Momentum:
    """Accumulate ``velocity = momentum * velocity + gradient``.

    Update parameters with ``parameter_new = parameter - learning_rate *
    velocity``. Velocity starts at zero; this convention does not multiply
    the gradient by (1 - momentum). Momentum must be in [0, 1). See Optimizer
    for input, state lifetime, and mutation guarantees.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        self.learning_rate = _positive_float(learning_rate, "learning_rate")
        self.momentum = _decay(momentum, "momentum")
        self._velocity: ParameterDict | None = None

    def step(self, params: Parameters, grads: Parameters) -> ParameterDict:
        """Return updated parameters and retain independent velocity history."""
        learning_rate = _positive_float(self.learning_rate, "learning_rate")
        momentum = _decay(self.momentum, "momentum")
        parameters, gradients = _prepare_step(params, grads, self._velocity)
        velocity: ParameterDict = {}
        updated: ParameterDict = {}
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for name, parameter in parameters.items():
                previous = (
                    np.zeros_like(parameter)
                    if self._velocity is None
                    else self._velocity[name]
                )
                velocity[name] = np.asarray(
                    momentum * previous + gradients[name], dtype=np.float64
                )
                updated[name] = np.asarray(
                    parameter - learning_rate * velocity[name], dtype=np.float64
                )
        _check_finite(velocity, updated)
        self._velocity = velocity
        return updated

    def reset(self) -> None:
        """Forget velocity and parameter names/shapes for a new training run."""
        self._velocity = None
