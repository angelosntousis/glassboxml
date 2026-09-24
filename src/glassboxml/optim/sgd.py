"""Stochastic gradient descent with caller-provided batch gradients."""

from glassboxml.optim.base import ParameterDict, Parameters
from glassboxml.optim.gd import GradientDescent


class SGD(GradientDescent):
    """Apply the gradient descent equation to a sample or mini-batch gradient.

    SGD and batch gradient descent have the same parameter update equation.
    Their difference is which observations the caller uses to compute grads.
    The training loop selects batches, averages their gradients, and owns its
    NumPy random generator/seed. This class draws no random numbers and keeps
    no state, so a fixed sequence of batch gradients gives repeatable updates.
    ``learning_rate`` defaults to 0.01, as in GradientDescent.
    """

    def step(self, params: Parameters, grads: Parameters) -> ParameterDict:
        """Apply ``params - learning_rate * grads`` to a supplied batch gradient.

        The implementation is shared with GradientDescent because the update
        equation is identical. The caller computes grads from one sample or a
        mini-batch; this method does not receive or sample training data.
        """
        return super().step(params, grads)
