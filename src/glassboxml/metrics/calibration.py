"""Top-label confidence and calibration diagnostics implemented with NumPy.

These diagnose the winning class probability, not calibration of every class.
ECE is a finite-sample, bin-dependent estimate: changing the number of bins can
change the result, sparse bins are noisy, and errors can cancel within a bin.
A small ECE does not guarantee good accuracy or classwise calibration. Evaluate
on held-out observations; these functions do not fit or recalibrate a model.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _probabilities(probabilities: ArrayLike) -> NDArray[np.float64]:
    """Validate an (N, K) probability matrix without clipping or renormalizing."""
    values = np.asarray(probabilities)
    if values.dtype.kind not in "iuf":
        raise ValueError("probabilities must contain real numeric values")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape (N, K), with N > 0 and K >= 2")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1, rtol=1e-6, atol=1e-8):
        raise ValueError("each probability row must sum to 1")
    return values


def confidence(probabilities: ArrayLike) -> NDArray[np.float64]:
    """Return max_k p(y=k|x) for each row of an (N, K) probability matrix.

    Require N > 0, K >= 2, finite entries in [0, 1], and row sums close to one
    (rtol=1e-6, atol=1e-8, allowing float32 rounding). Inputs are never mutated,
    clipped, or renormalized. Binary probabilities must have two columns.
    """
    return np.max(_probabilities(probabilities), axis=1)


@dataclass(frozen=True)
class ReliabilityDiagramData:
    """Equal-width bin statistics for a top-label reliability diagram.

    bin_edges has length n_bins + 1; all other arrays have length n_bins.
    Empty bins have count zero and NaN mean_confidence/accuracy; omit them when
    plotting rather than displaying zero accuracy. The returned arrays are fresh.
    """

    bin_edges: NDArray[np.float64]
    counts: NDArray[np.int64]
    mean_confidence: NDArray[np.float64]
    accuracy: NDArray[np.float64]

    @property
    def ece(self) -> float:
        """Return sum_b (n_b/N) * |accuracy_b - mean_confidence_b|."""
        occupied = self.counts > 0
        weights = self.counts[occupied] / self.counts.sum()
        gaps = np.abs(self.accuracy[occupied] - self.mean_confidence[occupied])
        return float(weights @ gaps)


def _labels(values: ArrayLike, name: str) -> NDArray[Any]:
    """Require a finite numeric or Unicode label vector; reject object arrays."""
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.dtype.kind not in "biufU":
        raise ValueError(f"{name} must be a 1D numeric or Unicode label vector")
    if labels.dtype.kind != "U" and not np.all(np.isfinite(labels)):
        raise ValueError(f"{name} must contain finite labels")
    return labels


def reliability_diagram_data(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    n_bins: int = 10,
    classes: ArrayLike | None = None,
) -> ReliabilityDiagramData:
    """Bin top-label accuracy and mean confidence on equal-width intervals.

    Args:
        y_true: Matching 1D ground-truth labels. Without classes, require integer
            column indices in [0, K); integer-valued floats are accepted.
        probabilities: Valid (N, K) probability matrix, as in confidence().
        n_bins: Positive integer, default 10.
        classes: Optional unique numeric or Unicode labels in probability-column
            order, for example model.classes_. Unknown labels raise ValueError.

    Bins cover [0, 1] and are [left, right), except the final bin includes 1.
    A value exactly on an interior edge goes in the bin to its right. Argmax
    ties choose the first column, matching NumPy and SoftmaxRegression. Every
    observation belongs to exactly one bin. No inputs are mutated.

    Plot occupied-bin mean_confidence on x and accuracy on y, with y=x as the
    calibration reference. counts provides a confidence histogram or bin sizes.
    """
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(
        n_bins, (int, np.integer)
    ):
        raise ValueError("n_bins must be a positive integer")
    if n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    values = _probabilities(probabilities)
    truth = _labels(y_true, "y_true")
    if truth.shape != (values.shape[0],):
        raise ValueError("y_true must match the number of probability rows")
    if classes is None:
        if (
            truth.dtype.kind == "U"
            or np.any(truth < 0)
            or np.any(truth >= values.shape[1])
            or np.any(truth != np.floor(truth))
        ):
            raise ValueError("y_true must contain integer column indices in [0, K)")
        correct = values.argmax(axis=1) == truth
    else:
        labels = _labels(classes, "classes")
        if labels.shape != (values.shape[1],) or np.unique(labels).size != labels.size:
            raise ValueError("classes must contain one unique label per column")
        if (truth.dtype.kind == "U") != (labels.dtype.kind == "U"):
            raise ValueError("y_true and classes must use compatible label types")
        if not np.all(np.isin(truth, labels)):
            raise ValueError("y_true contains labels absent from classes")
        correct = labels[values.argmax(axis=1)] == truth
    scores = values.max(axis=1)
    # Divide each integer edge directly, avoiding accumulated step rounding
    # (e.g. an intended 0.6 boundary represented as 0.6000000000000001).
    edges = np.arange(int(n_bins) + 1, dtype=np.float64) / int(n_bins)
    bins = np.searchsorted(edges[1:-1], scores, side="right")
    counts = np.bincount(bins, minlength=int(n_bins)).astype(np.int64)
    mean_confidence = np.full(int(n_bins), np.nan)
    accuracy = np.full(int(n_bins), np.nan)
    np.divide(
        np.bincount(bins, weights=scores, minlength=int(n_bins)),
        counts,
        out=mean_confidence,
        where=counts > 0,
    )
    np.divide(
        np.bincount(bins, weights=correct, minlength=int(n_bins)),
        counts,
        out=accuracy,
        where=counts > 0,
    )
    return ReliabilityDiagramData(edges, counts, mean_confidence, accuracy)


def expected_calibration_error(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    n_bins: int = 10,
    classes: ArrayLike | None = None,
) -> float:
    """Return top-label ECE = sum_b (n_b/N) * |accuracy_b - confidence_b|.

    Uses reliability_diagram_data bin and label conventions. Empty bins contribute
    zero. The result lies in [0, 1]; it is a diagnostic, not a significance test.
    """
    return reliability_diagram_data(
        y_true, probabilities, n_bins=n_bins, classes=classes
    ).ece


ece = expected_calibration_error
