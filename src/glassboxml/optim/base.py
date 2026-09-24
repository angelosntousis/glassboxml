"""Small, explicit update contract for named NumPy parameter arrays.

Optimizers minimize a loss given gradients computed by the caller. They neither
differentiate functions nor select batches. All updates are deterministic for a
given sequence of gradients. Inputs are floating-point arrays; calculations and
outputs use float64, matching the project's numerical diagnostics.
"""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

type Parameters = Mapping[str, NDArray[np.floating[Any]]]
type ParameterDict = dict[str, NDArray[np.float64]]


@runtime_checkable
class Optimizer(Protocol):
    """Interface consumed by model training loops.

    Example::

        optimizer = Adam(learning_rate=0.01)
        params = {"weights": np.zeros((4, 3)), "bias": np.zeros(3)}
        # Compute matching analytical gradients in the model, then:
        params = optimizer.step(params, grads)

    Each step returns fresh arrays without mutating or aliasing either input.
    Stateful optimizers retain history by parameter name, not array identity.
    Their names and shapes must stay fixed until reset; dictionary order may
    change. Use a separate instance per training run, or reset before reuse.
    Validation failures and nonfinite updates do not advance optimizer history.
    """

    def step(self, params: Parameters, grads: Parameters) -> ParameterDict:
        """Return updated float64 parameters with matching names and shapes.

        Inputs must be nonempty mappings of string names to finite floating
        NumPy arrays. Scalar and empty arrays are supported. Shapes must match
        exactly; broadcasting is prohibited. Invalid inputs raise ValueError;
        arithmetic overflow raises FloatingPointError (rescale or reduce the
        learning rate). Returned arrays may safely be modified by the caller.
        """
        ...

    def reset(self) -> None:
        """Clear update history and parameter schema, keeping hyperparameters."""
        ...


def _finite_float(value: float, name: str) -> float:
    """Validate a real finite scalar hyperparameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _positive_float(value: float, name: str) -> float:
    """Validate a strictly positive scalar hyperparameter."""
    result = _finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _decay(value: float, name: str) -> float:
    """Validate a decay coefficient in [0, 1)."""
    result = _finite_float(value, name)
    if not 0 <= result < 1:
        raise ValueError(f"{name} must be in [0, 1)")
    return result


def _prepare_step(
    params: Parameters, grads: Parameters, state: ParameterDict | None = None
) -> tuple[ParameterDict, ParameterDict]:
    """Validate an entire update and return independent float64 input copies."""
    if not isinstance(params, Mapping) or not isinstance(grads, Mapping):
        raise ValueError("params and grads must be mappings of named arrays")
    if not params or any(not isinstance(name, str) for name in params):
        raise ValueError("params must be nonempty with string names")
    if params.keys() != grads.keys():
        raise ValueError("params and grads must have exactly matching names")
    if state is not None and state.keys() != params.keys():
        raise ValueError("parameter names changed; call reset before a new run")

    parameters: ParameterDict = {}
    gradients: ParameterDict = {}
    for name in params:
        for source, destination in ((params, parameters), (grads, gradients)):
            array = source[name]
            if not isinstance(array, np.ndarray) or array.dtype.kind != "f":
                raise ValueError(f"{name!r} must be a floating-point NumPy array")
            with np.errstate(over="ignore", invalid="ignore"):
                value = np.array(array, dtype=np.float64, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name!r} must contain finite float64 values")
            destination[name] = value
        if parameters[name].shape != gradients[name].shape:
            raise ValueError(f"parameter and gradient shapes must match for {name!r}")
        if state is not None and state[name].shape != parameters[name].shape:
            raise ValueError("parameter shapes changed; call reset before a new run")
    return parameters, gradients


def _check_finite(*mappings: ParameterDict) -> None:
    """Reject nonfinite results before committing any optimizer state."""
    if any(
        not np.all(np.isfinite(value))
        for mapping in mappings
        for value in mapping.values()
    ):
        raise FloatingPointError("nonfinite optimizer update; rescale the problem")
