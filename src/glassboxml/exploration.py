"""Reusable explorer workflows; UI frameworks are deliberately not imported here.

Synthetic data generation, artifact validation, model calls, and diagnostics live
here so presentation layers need not implement ML equations. Models, feature
maps, and metrics remain the source of the numerical implementations.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from glassboxml.metrics import (
    ReliabilityDiagramData,
    confidence,
    mean_squared_error,
    reliability_diagram_data,
)
from glassboxml.models import BayesianLinearRegression, SoftmaxRegression
from glassboxml.preprocessing import PolynomialFeatures

SIGNALS = ("Cubic", "Sine", "Linear")
SAMPLING = ("Random", "Evenly spaced", "Gap in the middle")


def _npz_archive(path: Path) -> np.lib.npyio.NpzFile:
    """Reject plain NPY files as well as pickled objects before artifact access."""
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Expected an NPZ archive containing named arrays")
    return archive


@dataclass(frozen=True)
class RegressionSettings:
    """Deterministic synthetic data and fixed Gaussian-model hyperparameters."""

    n_samples: int = 20
    degree: int = 3
    alpha: float = 1.0
    noise_std: float = 0.2
    generation_noise_std: float = 0.2
    seed: int = 42
    signal: str = "Cubic"
    sampling: str = "Random"

    def __post_init__(self) -> None:
        for name, minimum in (("n_samples", 2), ("degree", 0), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("alpha", "noise_std", "generation_noise_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.signal not in SIGNALS or self.sampling not in SAMPLING:
            raise ValueError("unknown signal or observation sampling scheme")


@dataclass(frozen=True)
class RegressionExploration:
    """Plot-ready posterior predictions and the observations used to fit them."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    grid: NDArray[np.float64]
    truth: NDArray[np.float64]
    mean: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    weights: NDArray[np.float64]
    training_mse: float
    mean_interval_width: float
    epistemic_std: NDArray[np.float64]
    latent_lower: NDArray[np.float64]
    latent_upper: NDArray[np.float64]
    posterior_draws: NDArray[np.float64]
    test_mse: float
    test_coverage: float


def explore_regression(settings: RegressionSettings) -> RegressionExploration:
    """Fit a polynomial Gaussian model and form pointwise 95% predictive intervals.

    The intervals include observation noise, conditional on fixed alpha/beta.
    All bias and polynomial weights share the same zero-mean Gaussian prior.
    Model-control changes preserve the observations when generation settings match.
    """
    rng = np.random.default_rng(settings.seed)
    if settings.sampling == "Evenly spaced":
        x = np.linspace(-1.0, 1.0, settings.n_samples)
    elif settings.sampling == "Gap in the middle":
        x = rng.uniform(0.35, 1.0, settings.n_samples)
        x[: settings.n_samples // 2] *= -1
        x.sort()
    else:
        x = np.sort(rng.uniform(-1, 1, settings.n_samples))

    def signal(values: NDArray[np.float64]) -> NDArray[np.float64]:
        if settings.signal == "Sine":
            return np.sin(np.pi * values)
        if settings.signal == "Linear":
            return 0.5 + 1.2 * values
        return 0.5 + 1.2 * values - 0.8 * values**2 - 0.6 * values**3

    y = signal(x) + rng.normal(0, settings.generation_noise_std, settings.n_samples)
    grid = np.linspace(-1.3, 1.3, 401)
    features = PolynomialFeatures(settings.degree, include_bias=True)
    model = BayesianLinearRegression(settings.alpha, beta=1 / settings.noise_std**2)
    design = features.transform(x)
    model.fit(design, y)
    mean, std = model.predict(features.transform(grid), return_std=True)
    half_width = 1.959963984540054 * std
    grid_design = features.transform(grid)
    covariance = model.posterior_covariance_
    epistemic_std = np.sqrt(
        np.maximum(np.einsum("ij,jk,ik->i", grid_design, covariance, grid_design), 0)
    )
    draw_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, 891]))
    sampled_weights = draw_rng.multivariate_normal(
        model.posterior_mean_, covariance, size=6
    )
    test_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, 275]))
    test_x = test_rng.uniform(-1, 1, 1000)
    test_y = signal(test_x) + test_rng.normal(0, settings.generation_noise_std, 1000)
    test_mean, test_std = model.predict(features.transform(test_x), return_std=True)
    return RegressionExploration(
        x,
        y,
        grid,
        signal(grid),
        mean,
        mean - half_width,
        mean + half_width,
        model.posterior_mean_,
        mean_squared_error(y, model.predict(design)),
        float(np.mean(2 * half_width)),
        epistemic_std,
        mean - 1.959963984540054 * epistemic_std,
        mean + 1.959963984540054 * epistemic_std,
        grid_design @ sampled_weights.T,
        mean_squared_error(test_y, test_mean),
        float(np.mean(np.abs(test_y - test_mean) <= 1.959963984540054 * test_std)),
    )


def available_softmax_models(path: Path) -> tuple[str, ...]:
    """List complete named models in the experiment's NPZ artifact, without pickle."""
    with _npz_archive(path) as archive:
        names = tuple(
            sorted(
                key.removesuffix("_weights")
                for key in archive.files
                if key.endswith("_weights")
                and key.removesuffix("_weights") + "_bias" in archive.files
                and key.removesuffix("_weights") + "_classes" in archive.files
            )
        )
    if not names:
        raise ValueError(
            "No models found; expected NAME_weights, NAME_bias, NAME_classes"
        )
    return names


def load_softmax_artifact(path: Path, name: str) -> SoftmaxRegression:
    """Restore a named softmax model from the MNIST experiment's numeric NPZ format."""
    with _npz_archive(path) as archive:
        try:
            return SoftmaxRegression.from_parameters(
                archive[f"{name}_weights"],
                archive[f"{name}_bias"],
                archive[f"{name}_classes"],
            )
        except KeyError as error:
            raise ValueError(f"Incomplete or missing model: {name}") from error


@dataclass(frozen=True)
class MnistExploration:
    """Inference and diagnostics for exactly the recorded test split, when supplied."""

    images: NDArray[np.uint8]
    labels: NDArray[np.int64]
    indices: NDArray[np.int64]
    classes: NDArray[np.int64]
    weights: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    predictions: NDArray[np.int64]
    confidence: NDArray[np.float64]
    uncertainty: NDArray[np.float64]
    accuracy: float
    mean_confidence: float
    calibration: ReliabilityDiagramData
    confusion: NDArray[np.int64]
    common_errors: list[dict[str, int]]
    groups: dict[str, NDArray[np.int64]]
    bias: NDArray[np.float64]


def explore_mnist(
    weights_path: Path,
    data_path: Path,
    name: str,
    metadata_path: Path | None = None,
) -> MnistExploration:
    """Load a trained artifact and evaluate its held-out MNIST images using the package.

    Optional experiment JSON restores exact test indices and bin count, verifying
    the dataset checksum before use. Without it, evaluate the full official test
    pool with ten bins. Metrics are recomputed for these weights, never trusted
    from a potentially unrelated JSON. No downloads or training occur here.
    """
    model = load_softmax_artifact(weights_path, name)
    classes = model.classes_
    if (
        model.coef_.shape != (784, 10)
        or classes.dtype.kind not in "iu"
        or not np.array_equal(np.sort(classes), np.arange(10))
    ):
        raise ValueError("MNIST requires 784 features and integer classes 0 through 9")
    with _npz_archive(data_path) as data:
        try:
            images, labels = data["x_test"], data["y_test"]
        except KeyError as error:
            raise ValueError("Dataset must contain x_test and y_test") from error
        if (
            images.ndim != 3
            or images.shape[1:] != (28, 28)
            or images.shape[0] == 0
            or images.dtype != np.uint8
        ):
            raise ValueError("MNIST images must be nonempty uint8 (N, 28, 28)")
        if (
            labels.shape != (len(images),)
            or labels.dtype.kind not in "iu"
            or np.any((labels < 0) | (labels > 9))
        ):
            raise ValueError("MNIST labels must match images and be integers in [0, 9]")
    indices = np.arange(len(images), dtype=np.int64)
    n_bins = 10
    if metadata_path is not None:
        try:
            metadata: Any = json.loads(metadata_path.read_text())
            expected_hash = metadata["dataset"]["sha256"]
            with data_path.open("rb") as handle:
                actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            if expected_hash != actual_hash:
                raise ValueError(
                    "Dataset checksum differs from the experiment metadata"
                )
            indices = np.asarray(metadata["splits"]["official_test_indices"])
            if (
                indices.ndim != 1
                or indices.size == 0
                or indices.dtype.kind not in "iu"
                or np.any(indices < 0)
                or np.any(indices >= len(images))
                or np.unique(indices).size != indices.size
            ):
                raise ValueError("Metadata test indices must be unique and in range")
            indices = indices.astype(np.int64)
            n_bins = metadata["config"].get("calibration_bins", 10)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Invalid MNIST experiment metadata structure") from error
    images, labels = images[indices].copy(), labels[indices].astype(np.int64)
    probabilities = model.predict_proba(images.reshape(len(images), -1) / 255.0)
    predictions = classes[probabilities.argmax(axis=1)].astype(np.int64)
    scores = confidence(probabilities)
    calibration = reliability_diagram_data(
        labels, probabilities, classes=classes, n_bins=n_bins
    )
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    wrong_counts = confusion.copy()
    np.fill_diagonal(wrong_counts, 0)
    pairs = np.argsort(-wrong_counts.ravel(), kind="stable")
    common = [
        {
            "True digit": int(i // 10),
            "Predicted digit": int(i % 10),
            "Count": int(wrong_counts.ravel()[i]),
        }
        for i in pairs[:10]
        if wrong_counts.ravel()[i] > 0
    ]
    correct = np.flatnonzero(predictions == labels)
    wrong = np.flatnonzero(predictions != labels)
    groups = {
        "All test images": np.arange(len(images), dtype=np.int64),
        "Most confident correct": correct[np.argsort(-scores[correct], kind="stable")],
        "Most confident incorrect": wrong[np.argsort(-scores[wrong], kind="stable")],
        "Least confident": np.argsort(scores, kind="stable"),
    }
    return MnistExploration(
        images,
        labels,
        indices,
        classes.astype(np.int64),
        model.coef_,
        probabilities,
        predictions,
        scores,
        1 - scores,
        float(np.mean(predictions == labels)),
        float(scores.mean()),
        calibration,
        confusion,
        common,
        groups,
        model.intercept_,
    )
