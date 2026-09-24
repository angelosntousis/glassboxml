"""Multiclass linear softmax regression with analytical NumPy gradients."""

from copy import deepcopy
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from glassboxml.models.base import _feature_matrix, _real_array
from glassboxml.optim import Adam, Optimizer
from glassboxml.optim.base import ParameterDict, _finite_float


def _log_probabilities(logits: ArrayLike) -> NDArray[np.float64]:
    """Compute log softmax after subtracting each row's maximum logit.

    Very negative finite logits may produce -inf after subtraction if their
    difference is unrepresentable. Their probabilities are correctly zero;
    an unrepresentable cross-entropy is rejected by the loss calculation.
    """
    values = _real_array(logits, "logits")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("logits must have shape (n_samples, n_classes >= 2)")
    with np.errstate(over="ignore", under="ignore"):
        shifted = values - np.max(values, axis=1, keepdims=True)
        log_normalizer = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return np.asarray(shifted - log_normalizer, dtype=np.float64)


def _softmax(logits: ArrayLike) -> NDArray[np.float64]:
    """Return stable rowwise probabilities; extreme tails may underflow to zero."""
    with np.errstate(under="ignore"):
        return np.exp(_log_probabilities(logits))


def _cross_entropy(
    X: NDArray[np.float64], y: NDArray[np.int64], params: ParameterDict
) -> tuple[float, NDArray[np.float64]]:
    """Return mean cross-entropy and probabilities for validated, encoded data."""
    with np.errstate(over="raise", invalid="raise"):
        logits = X @ params["weights"] + params["bias"]
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("nonfinite logits; rescale X or the learning rate")
        log_probs = _log_probabilities(logits)
        # Normalize before summing to avoid unnecessary large-batch overflow.
        loss = float(-np.sum(log_probs[np.arange(y.size), y] / y.size))
    if not np.isfinite(loss):
        raise FloatingPointError("nonfinite cross-entropy; rescale logits")
    with np.errstate(under="ignore"):
        probabilities = np.exp(log_probs)
    return loss, probabilities


def _penalty(weights: NDArray[np.float64], l2: float) -> float:
    """Return (l2/2) * ||W||_F**2; the bias is never penalized."""
    if l2 == 0:
        return 0.0
    with np.errstate(over="raise", invalid="raise"):
        # Scale before squaring so tiny l2 can temper large finite weights.
        scaled = np.sqrt(l2) * weights / np.sqrt(2.0)
        value = float(np.sum(scaled**2))
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite L2 penalty; rescale weights")
    return value


def _loss_and_gradients(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    params: ParameterDict,
    l2: float = 0.0,
) -> tuple[float, ParameterDict]:
    r"""Return the regularized objective and vectorized gradients.

    For encoded class indices y and one-hot matrix Y, with P=softmax(XW+b):
    L = -mean(log(P[i,y[i]])) + (l2/2)||W||_F^2,
    dW = X.T @ (P-Y)/n + l2*W, db = sum(P-Y, axis=0)/n.
    Cross-entropy uses log-softmax directly, never log of clipped probabilities.
    This internal helper expects validated data and parameters.
    """
    data_loss, residual = _cross_entropy(X, y, params)
    with np.errstate(over="raise", invalid="raise"):
        loss = data_loss + _penalty(params["weights"], l2)
        residual[np.arange(y.size), y] -= 1.0
        residual /= y.size
        grads = {
            "weights": X.T @ residual + l2 * params["weights"],
            "bias": residual.sum(axis=0),
        }
    if not np.isfinite(loss) or any(not np.all(np.isfinite(g)) for g in grads.values()):
        raise FloatingPointError("nonfinite loss or gradient; rescale the problem")
    return loss, grads


def _labels(y: ArrayLike, n_samples: int) -> NDArray[Any]:
    """Preserve discrete labels without coercing arbitrary classes into indices."""
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.size != n_samples:
        raise ValueError("y must have shape (n_samples,) matching X")
    if labels.dtype.kind not in "biufUS":
        raise ValueError("y must contain integer, boolean, or string class labels")
    if labels.dtype.kind in "iuf":
        if not np.all(np.isfinite(labels)):
            raise ValueError("y must contain finite class labels")
        if labels.dtype.kind == "f" and np.any(labels != np.floor(labels)):
            raise ValueError("floating class labels must be integer-valued")
    return labels.copy()


def _encode_known(labels: NDArray[Any], classes: NDArray[Any]) -> NDArray[np.int64]:
    """Encode validation labels using the training classes, rejecting unseen labels."""
    # A dictionary avoids unsafe string-to-number coercion or searchsorted casting.
    mapping = {label: index for index, label in enumerate(classes.tolist())}
    try:
        return np.array([mapping[label] for label in labels.tolist()], dtype=np.int64)
    except KeyError as error:
        raise ValueError(
            "validation data contains a class not present in training"
        ) from error


def _positive_integer(value: int, name: str) -> int:
    """Validate positive integer counts, excluding booleans."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class SoftmaxRegression:
    r"""Optimize multiclass cross-entropy for logits = X W + b.

    p(y=k|x) = exp(z_k - max(z)) / sum_j exp(z_j - max(z)).
    The training objective is mean cross-entropy + (l2/2) * ||W||_F^2.
    Biases are not regularized. Features are not automatically standardized.

    Args:
        optimizer: An object implementing step(params, grads) and reset(). Each
            fit deep-copies and resets it, leaving the supplied object untouched.
            None uses Adam(learning_rate=0.01).
        l2: Nonnegative coefficient of the squared-weight penalty.
        max_epochs: Maximum complete passes over training data (default 200).
        batch_size: Positive mini-batch size, or None for full-batch updates.
        shuffle: Permute observations at the start of each epoch (default True).
        random_state: Nonnegative seed for a local NumPy generator, or None.
        early_stopping: Require validation_data and stop on validation-loss
            stagnation, restoring the absolute best validation epoch.
        patience: Number of epochs without a min_delta improvement before stopping.
        min_delta: Nonnegative absolute validation-loss decrease needed to reset
            patience. Smaller improvements can still become the restored best epoch.

    X is a finite 2D matrix; y is a matching vector with at least two classes.
    Integer, boolean, string, and integer-valued floating labels are supported;
    classes_ gives the sorted probability-column order. coef_ has shape
    (n_features, n_classes), matching X @ W, and intercept_ has shape (n_classes,).
    W and b start at zero, which is valid for linear softmax. A seed controls
    batching, not initialization. Refit starts a fresh run, not a warm start.

    After each epoch, history_ records full-data train_loss (plain cross-entropy),
    train_objective (including L2), train_accuracy, validation_loss, and
    validation_accuracy. Validation lists are empty without validation data.
    n_iter_ counts completed epochs; best_epoch_ is the 1-based lowest validation
    loss epoch, or None. History retains epochs after best_epoch_ even when the
    fitted parameters are restored. Learned properties return independent copies.
    Failed fits leave the previous fitted estimator intact.
    """

    def __init__(
        self,
        *,
        optimizer: Optimizer | None = None,
        l2: float = 0.0,
        max_epochs: int = 200,
        batch_size: int | None = 32,
        shuffle: bool = True,
        random_state: int | None = None,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 1e-4,
    ) -> None:
        self.optimizer = optimizer
        self.l2, self.max_epochs, self.batch_size = l2, max_epochs, batch_size
        self.shuffle, self.random_state = shuffle, random_state
        self.early_stopping, self.patience, self.min_delta = (
            early_stopping,
            patience,
            min_delta,
        )
        self._validate_hyperparameters()
        self._params: ParameterDict | None = None
        self._classes: NDArray[Any] | None = None
        self._history: dict[str, list[float]] | None = None
        self._n_iter = 0
        self._best_epoch: int | None = None

    def _validate_hyperparameters(self) -> None:
        """Validate settings at construction and before every fresh fit."""
        if self.optimizer is not None and (
            isinstance(self.optimizer, type)
            or not isinstance(self.optimizer, Optimizer)
            or not callable(self.optimizer.step)
            or not callable(self.optimizer.reset)
        ):
            raise ValueError(
                "optimizer must be an instance with callable step and reset"
            )
        self.l2 = _finite_float(self.l2, "l2")
        self.min_delta = _finite_float(self.min_delta, "min_delta")
        if self.l2 < 0 or self.min_delta < 0:
            raise ValueError("l2 and min_delta must be nonnegative")
        self.max_epochs = _positive_integer(self.max_epochs, "max_epochs")
        self.patience = _positive_integer(self.patience, "patience")
        if self.batch_size is not None:
            self.batch_size = _positive_integer(self.batch_size, "batch_size")
        for name in ("shuffle", "early_stopping"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        if self.random_state is not None:
            if (
                isinstance(self.random_state, (bool, np.bool_))
                or not isinstance(self.random_state, (int, np.integer))
                or self.random_state < 0
            ):
                raise ValueError("random_state must be a nonnegative integer or None")
            self.random_state = int(self.random_state)

    def _require_fitted(self) -> ParameterDict:
        """Return internal parameters only for implementation use."""
        if self._params is None:
            raise RuntimeError("Call fit before accessing the fitted classifier")
        return self._params

    @classmethod
    def from_parameters(
        cls, coef: ArrayLike, intercept: ArrayLike, classes: ArrayLike
    ) -> Self:
        """Restore an inference model from copied, validated NumPy parameters.

        coef has shape (n_features, n_classes), intercept has shape (n_classes,),
        and classes contains one unique label per column, in that same order.
        No training history or optimizer state is reconstructed: history_ remains
        unavailable, n_iter_ is zero, and best_epoch_ is None. A later fit starts
        afresh with the usual constructor defaults. No pickle is required.
        """
        weights = _feature_matrix(coef)
        if weights.shape[1] < 2:
            raise ValueError("coef must have at least two class columns")
        bias = _real_array(intercept, "intercept")
        if bias.shape != (weights.shape[1],):
            raise ValueError("intercept must have one value per class column")
        labels = _labels(classes, weights.shape[1])
        if np.unique(labels).size != labels.size:
            raise ValueError("classes must contain unique labels")
        model = cls()
        model._params = {"weights": weights.copy(), "bias": bias.copy()}
        model._classes = labels.copy()
        return model

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Weights W of shape (n_features, n_classes), copied for the caller."""
        return self._require_fitted()["weights"].copy()

    @property
    def intercept_(self) -> NDArray[np.float64]:
        """Biases b of shape (n_classes,), copied for the caller."""
        return self._require_fitted()["bias"].copy()

    @property
    def classes_(self) -> NDArray[Any]:
        """Original labels in probability-column order; sorted when learned by fit."""
        if self._classes is None:
            raise RuntimeError("Call fit before accessing classes")
        return self._classes.copy()

    @property
    def history_(self) -> dict[str, list[float]]:
        """Independent epoch histories; losses are measured after all epoch updates."""
        if self._history is None:
            raise RuntimeError("Call fit before accessing training history")
        return {key: values.copy() for key, values in self._history.items()}

    @property
    def n_iter_(self) -> int:
        """Number of completed training epochs."""
        self._require_fitted()
        return self._n_iter

    @property
    def best_epoch_(self) -> int | None:
        """Lowest-validation-loss epoch (1-based), or None without validation."""
        self._require_fitted()
        return self._best_epoch

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        validation_data: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> Self:
        """Fit from scratch using only training gradients, returning self.

        Each observation appears once per epoch, including the final partial
        batch, whose gradient uses its actual size. Validation data must have
        matching features and known classes. Early stopping monitors plain
        validation cross-entropy, never validation gradients or training loss.
        """
        self._validate_hyperparameters()
        features = _feature_matrix(X)
        labels = _labels(y, features.shape[0])
        classes, encoded = np.unique(labels, return_inverse=True)
        encoded = np.asarray(encoded, dtype=np.int64)
        if classes.size < 2:
            raise ValueError("training data must contain at least two classes")
        validation: tuple[NDArray[np.float64], NDArray[np.int64]] | None = None
        if validation_data is not None:
            if (
                not isinstance(validation_data, (tuple, list))
                or len(validation_data) != 2
            ):
                raise ValueError("validation_data must be an (X, y) pair")
            validation_X = _feature_matrix(validation_data[0])
            if validation_X.shape[1] != features.shape[1]:
                raise ValueError(
                    "validation data must have the same number of features"
                )
            validation_y = _labels(validation_data[1], validation_X.shape[0])
            validation = (validation_X, _encode_known(validation_y, classes))
        if self.early_stopping and validation is None:
            raise ValueError("early_stopping requires validation_data")

        optimizer = (
            Adam(learning_rate=0.01)
            if self.optimizer is None
            else deepcopy(self.optimizer)
        )
        optimizer.reset()
        rng = np.random.default_rng(self.random_state)
        params = {
            "weights": np.zeros((features.shape[1], classes.size), dtype=np.float64),
            "bias": np.zeros(classes.size, dtype=np.float64),
        }
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_objective": [],
            "train_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
        }
        batch_size = features.shape[0] if self.batch_size is None else self.batch_size
        best_loss, significant_loss = np.inf, np.inf
        best_epoch, stale_epochs = None, 0
        best_params: ParameterDict | None = None
        for epoch in range(1, self.max_epochs + 1):
            order = (
                rng.permutation(features.shape[0])
                if self.shuffle
                else np.arange(features.shape[0])
            )
            for start in range(0, features.shape[0], batch_size):
                batch = order[start : start + batch_size]
                _, gradients = _loss_and_gradients(
                    features[batch], encoded[batch], params, self.l2
                )
                params = optimizer.step(params, gradients)

            train_loss, train_probs = _cross_entropy(features, encoded, params)
            train_objective = train_loss + _penalty(params["weights"], self.l2)
            if not np.isfinite(train_objective):
                raise FloatingPointError("nonfinite training objective")
            history["train_loss"].append(train_loss)
            history["train_objective"].append(train_objective)
            history["train_accuracy"].append(
                float(np.mean(train_probs.argmax(axis=1) == encoded))
            )
            if validation is not None:
                validation_loss, validation_probs = _cross_entropy(*validation, params)
                history["validation_loss"].append(validation_loss)
                history["validation_accuracy"].append(
                    float(np.mean(validation_probs.argmax(axis=1) == validation[1]))
                )
                if validation_loss < best_loss:
                    best_loss, best_epoch = validation_loss, epoch
                    if self.early_stopping:
                        best_params = {
                            key: value.copy() for key, value in params.items()
                        }
                if validation_loss < significant_loss - self.min_delta:
                    significant_loss, stale_epochs = validation_loss, 0
                else:
                    stale_epochs += 1
                if self.early_stopping and stale_epochs >= self.patience:
                    break

        if self.early_stopping and best_params is not None:
            params = best_params
        self._params = {key: value.copy() for key, value in params.items()}
        self._classes, self._history = classes.copy(), history
        self._n_iter, self._best_epoch = len(history["train_loss"]), best_epoch
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return shape (n_samples, n_classes) probabilities in classes_ order."""
        params = self._require_fitted()
        features = _feature_matrix(X)
        if features.shape[1] != params["weights"].shape[0]:
            raise ValueError(
                "X must have the same number of features as the fitted data"
            )
        with np.errstate(over="raise", invalid="raise"):
            logits = features @ params["weights"] + params["bias"]
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("nonfinite logits; rescale X")
        return _softmax(logits)

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Return original labels; ties choose the first probability column."""
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return mean classification accuracy; unseen target labels count as errors."""
        predicted = self.predict(X)
        labels = _labels(y, predicted.size)
        return float(np.mean(predicted == labels))
