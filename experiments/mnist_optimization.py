"""Compare NumPy softmax optimizers on MNIST, without notebooks or sklearn models.

Run ``python experiments/mnist_optimization.py`` with the experiments extra.
The first run downloads MNIST; later runs use the local NPZ cache. Defaults use
50,000 training, 10,000 validation, and 10,000 official test images. Pixels are
divided by 255, with no fitted preprocessing. All optimizers start from zero and
see identical epoch permutations. Learning rates are fixed before testing.

Early stopping is disabled for an equal epoch budget. Convergence is the first
completed epoch reaching a common validation NLL threshold, distinct from total
fit time (which includes full-data epoch diagnostics). Timings vary by hardware,
BLAS settings, and load; seeded results also depend on software versions. These
fixed-rate comparisons do not establish a universal optimizer ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from glassboxml.metrics import (
    ReliabilityDiagramData,
    confidence,
    reliability_diagram_data,
)
from glassboxml.models import SoftmaxRegression
from glassboxml.optim import SGD, Adam, Momentum, Optimizer

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
OPTIMIZER_SETTINGS: dict[str, dict[str, float]] = {
    "SGD": {"learning_rate": 0.1},
    "Momentum": {"learning_rate": 0.02, "momentum": 0.9},
    "Adam": {"learning_rate": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
}
FIGURE_NAMES = (
    "training_loss",
    "validation_loss",
    "confusion_matrix",
    "class_weights",
    "confident_errors",
    "calibration",
    "confident_correct",
    "least_confident",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Shared data, optimization, and convergence settings for all three runs."""

    seed: int = 42
    train_size: int = 50_000
    validation_size: int = 10_000
    test_size: int = 10_000
    epochs: int = 20
    batch_size: int = 128
    l2: float = 1e-4
    target_nll: float = 0.35
    calibration_bins: int = 10

    def __post_init__(self) -> None:
        for name in (
            "seed",
            "train_size",
            "validation_size",
            "test_size",
            "epochs",
            "batch_size",
            "calibration_bins",
        ):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.train_size < 10:
            raise ValueError("train_size must be at least 10 for ten MNIST classes")
        for name in ("l2", "target_nll"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class Dataset:
    """Raw uint8 images and labels from the official training and test pools."""

    train_images: NDArray[np.uint8]
    train_labels: NDArray[np.int64]
    test_images: NDArray[np.uint8]
    test_labels: NDArray[np.int64]


@dataclass(frozen=True)
class Splits:
    """Normalized design matrices and original-pool indices, shared by all fits."""

    X_train: NDArray[np.float64]
    y_train: NDArray[np.int64]
    X_validation: NDArray[np.float64]
    y_validation: NDArray[np.int64]
    X_test: NDArray[np.float64]
    y_test: NDArray[np.int64]
    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]


@dataclass(frozen=True)
class RunResult:
    """A fitted estimator plus held-out diagnostics and measured elapsed time."""

    name: str
    model: SoftmaxRegression
    fit_seconds: float
    test_accuracy: float
    test_nll: float
    confusion: NDArray[np.int64]
    predictions: NDArray[np.int64]
    confidence: NDArray[np.float64]
    calibration: ReliabilityDiagramData


def load_dataset(path: Path) -> Dataset:
    """Read a local NPZ without pickle and validate MNIST image/label conventions."""
    with np.load(path, allow_pickle=False) as archive:
        arrays: list[NDArray[Any]] = []
        for pool in ("train", "test"):
            images, labels = archive[f"x_{pool}"], archive[f"y_{pool}"]
            if (
                images.ndim != 3
                or images.shape[1:] != (28, 28)
                or images.shape[0] == 0
                or images.dtype != np.uint8
            ):
                raise ValueError(f"{pool} images must be nonempty uint8 (N, 28, 28)")
            if (
                labels.shape != (images.shape[0],)
                or not np.issubdtype(labels.dtype, np.integer)
                or np.any((labels < 0) | (labels > 9))
            ):
                raise ValueError(f"{pool} labels must be an integer vector in [0, 9]")
            arrays.extend((images, labels.astype(np.int64)))
    return Dataset(*arrays)


def ensure_dataset(path: Path) -> None:
    """Download once, replacing the cache only after a complete, validated download.

    Network failures propagate rather than silently substituting another dataset.
    Existing caches are untouched; load_dataset validates them before fitting.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as f:
            temporary = Path(f.name)
            with urllib.request.urlopen(DATA_URL, timeout=60) as response:
                shutil.copyfileobj(response, f)
        load_dataset(temporary)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def make_splits(data: Dataset, config: ExperimentConfig) -> Splits:
    """Split the official train pool once; never fit on the official test pool.

    Validation is a prefix of a seeded permutation, followed by training. A
    separate RNG stream selects test subsets; full test sets retain official
    order. Sampling is random, not stratified. All training classes are required,
    so very small subsets can raise a clear error.
    """
    if config.train_size + config.validation_size > len(data.train_labels):
        raise ValueError("train_size + validation_size exceeds the training pool")
    if config.test_size > len(data.test_labels):
        raise ValueError("test_size exceeds the official test pool")
    train_seed, test_seed = np.random.SeedSequence(config.seed).spawn(2)
    permutation = np.random.default_rng(train_seed).permutation(len(data.train_labels))
    validation = permutation[: config.validation_size]
    train = permutation[
        config.validation_size : config.validation_size + config.train_size
    ]
    test = np.arange(len(data.test_labels), dtype=np.int64)
    if config.test_size < len(test):
        test = np.random.default_rng(test_seed).permutation(test)[: config.test_size]
    if not np.array_equal(np.unique(data.train_labels[train]), np.arange(10)):
        raise ValueError(
            "training subset must contain all ten classes; increase train_size"
        )

    def design(images: NDArray[np.uint8]) -> NDArray[np.float64]:
        return images.reshape(len(images), -1).astype(np.float64) / 255.0

    return Splits(
        design(data.train_images[train]),
        data.train_labels[train].copy(),
        design(data.train_images[validation]),
        data.train_labels[validation].copy(),
        design(data.test_images[test]),
        data.test_labels[test].copy(),
        train,
        validation,
        test,
    )


def negative_log_likelihood(
    logits: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> float:
    """Mean NLL from shifted logits, avoiding log of underflowed probabilities."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(log_normalizer - shifted[np.arange(len(labels)), labels]))


def run_experiment(splits: Splits, config: ExperimentConfig) -> list[RunResult]:
    """Fit identical zero-initialized models with fixed, declared optimizer settings."""
    optimizers: dict[str, Optimizer] = {
        "SGD": SGD(**OPTIMIZER_SETTINGS["SGD"]),
        "Momentum": Momentum(**OPTIMIZER_SETTINGS["Momentum"]),
        "Adam": Adam(**OPTIMIZER_SETTINGS["Adam"]),
    }
    results = []
    for name, optimizer in optimizers.items():
        print(f"Training {name}: {config.epochs} epochs ...", flush=True)
        model = SoftmaxRegression(
            optimizer=optimizer,
            max_epochs=config.epochs,
            batch_size=config.batch_size,
            l2=config.l2,
            random_state=config.seed,
            shuffle=True,
            early_stopping=False,
        )
        started = time.perf_counter()
        model.fit(
            splits.X_train,
            splits.y_train,
            validation_data=(splits.X_validation, splits.y_validation),
        )
        elapsed = time.perf_counter() - started
        probabilities = model.predict_proba(splits.X_test)
        predictions = np.argmax(probabilities, axis=1).astype(np.int64)
        logits = splits.X_test @ model.coef_ + model.intercept_
        confusion = np.zeros((10, 10), dtype=np.int64)
        np.add.at(confusion, (splits.y_test, predictions), 1)
        result = RunResult(
            name,
            model,
            elapsed,
            float(np.mean(predictions == splits.y_test)),
            negative_log_likelihood(logits, splits.y_test),
            confusion,
            predictions,
            confidence(probabilities),
            reliability_diagram_data(
                splits.y_test,
                probabilities,
                n_bins=config.calibration_bins,
                classes=model.classes_,
            ),
        )
        print(
            f"  accuracy={result.test_accuracy:.4f}, NLL={result.test_nll:.4f}, "
            f"ECE={result.calibration.ece:.4f}, fit={elapsed:.2f}s",
            flush=True,
        )
        results.append(result)
    return results


def convergence_metrics(
    history: list[float], config: ExperimentConfig
) -> dict[str, Any]:
    """Return completed epochs/updates to a common target, or null if unreached."""
    epoch = next(
        (i + 1 for i, loss in enumerate(history) if loss <= config.target_nll), None
    )
    return {
        "target_validation_nll": config.target_nll,
        "epoch_to_target": epoch,
        "updates_to_target": (
            epoch * math.ceil(config.train_size / config.batch_size)
            if epoch is not None
            else None
        ),
        "best_validation_epoch": int(np.argmin(history)) + 1,
        "best_validation_nll": min(history),
    }


def confident_error_indices(
    result: RunResult, labels: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Rank errors by decreasing predicted-class probability, breaking ties stably."""
    wrong = np.flatnonzero(result.predictions != labels)
    return wrong[np.argsort(-result.confidence[wrong], kind="stable")]


def prediction_groups(
    result: RunResult, labels: NDArray[np.int64]
) -> dict[str, NDArray[np.int64]]:
    """Rank correct/errors descending and all examples ascending by confidence.

    Return indices into the supplied test split, breaking ties by split order.
    The least-confident group includes both correct and incorrect predictions.
    Empty correct/error groups are valid; callers choose how many to display.
    """
    correct = np.flatnonzero(result.predictions == labels)
    return {
        "highest_confidence_correct": correct[
            np.argsort(-result.confidence[correct], kind="stable")
        ],
        "highest_confidence_errors": confident_error_indices(result, labels),
        "least_confident": np.argsort(result.confidence, kind="stable"),
    }


def save_results(
    results: list[RunResult],
    splits: Splits,
    config: ExperimentConfig,
    dataset_path: Path,
    output: Path,
) -> None:
    """Persist metrics, histories, exact splits, dataset digest, and fitted weights."""
    output.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata: dict[str, Any] = {
        "dataset": {
            "name": "MNIST",
            "source_url": DATA_URL,
            "sha256": digest,
            "archive_path": str(dataset_path.resolve()),
        },
        "config": asdict(config),
        "protocol": {
            "scaling": "float64 pixels / 255; 784 row-major features",
            "initialization": "zero weights and biases for every optimizer",
            "split": "SeedSequence(seed).spawn(2); validation then training",
            "batching": "default_rng(seed); identical epoch permutations for all runs",
            "selection": "fixed final epoch, no early stopping or test-set tuning",
            "loss": "mean negative log likelihood; train_objective adds l2/2 * ||W||^2",
            "convergence": "first completed epoch with validation NLL <= target",
            "timing": "perf_counter around fit, including epoch metrics, excluding I/O",
            "calibration": "test top-label ECE; equal-width [left, right), last closed",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": version("matplotlib"),
            "platform": platform.platform(),
            "threads": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
        "splits": {
            "train_pool_indices": splits.train_indices.tolist(),
            "validation_pool_indices": splits.validation_indices.tolist(),
            "official_test_indices": splits.test_indices.tolist(),
        },
        "runs": {},
    }
    weights = {}
    for result in results:
        history = result.model.history_
        groups = prediction_groups(result, splits.y_test)
        calibration = result.calibration
        metadata["runs"][result.name] = {
            "optimizer": OPTIMIZER_SETTINGS[result.name],
            "history": history,
            "test_accuracy": result.test_accuracy,
            "test_nll": result.test_nll,
            "test_ece": calibration.ece,
            "calibration": {
                "bin_edges": calibration.bin_edges.tolist(),
                "counts": calibration.counts.tolist(),
                "mean_confidence": [
                    float(v) if np.isfinite(v) else None
                    for v in calibration.mean_confidence
                ],
                "accuracy": [
                    float(v) if np.isfinite(v) else None for v in calibration.accuracy
                ],
            },
            "fit_seconds": result.fit_seconds,
            "mean_seconds_per_epoch": result.fit_seconds / result.model.n_iter_,
            "total_updates": result.model.n_iter_
            * math.ceil(config.train_size / config.batch_size),
            "convergence": convergence_metrics(history["validation_loss"], config),
            "confusion_matrix": result.confusion.tolist(),
        }
        for group, indices in groups.items():
            metadata["runs"][result.name][group] = [
                {
                    "official_test_index": int(splits.test_indices[i]),
                    "true_label": int(splits.y_test[i]),
                    "prediction": int(result.predictions[i]),
                    "confidence": float(result.confidence[i]),
                }
                for i in indices[:6]
            ]
        weights[f"{result.name}_weights"] = result.model.coef_
        weights[f"{result.name}_bias"] = result.model.intercept_
        weights[f"{result.name}_classes"] = result.model.classes_
    (output / "mnist_optimization_metrics.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        output / "mnist_optimization_weights.npz", allow_pickle=False, **weights
    )


def plot_results(
    results: list[RunResult],
    splits: Splits,
    config: ExperimentConfig,
    output: Path,
) -> None:
    """Save optimizer and calibration comparisons as 300-dpi PNG and vector PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def save(figure: Any, name: str) -> None:
        for extension in ("png", "pdf"):
            figure.savefig(
                figures / f"mnist_{name}.{extension}", dpi=300, bbox_inches="tight"
            )
        plt.close(figure)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        colors = ("#0072B2", "#D55E00", "#009E73")
        for key, name, title in (
            ("train_loss", "training_loss", "Training negative log likelihood"),
            (
                "validation_loss",
                "validation_loss",
                "Validation negative log likelihood",
            ),
        ):
            figure, axis = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
            for result, color in zip(results, colors, strict=True):
                loss = result.model.history_[key]
                rate = OPTIMIZER_SETTINGS[result.name]["learning_rate"]
                axis.plot(
                    np.arange(1, len(loss) + 1),
                    loss,
                    color=color,
                    linewidth=2,
                    label=f"{result.name} (lr={rate:g})",
                )
            if key == "validation_loss":
                axis.axhline(
                    config.target_nll,
                    color="0.4",
                    linestyle=":",
                    label=f"Shared target: {config.target_nll:g}",
                )
            axis.set(
                title=f"MNIST · {title}",
                xlabel="Completed epoch",
                ylabel="Mean NLL (nats)",
            )
            axis.grid(alpha=0.2)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.legend(frameon=False)
            save(figure, name)

        figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
        maximum = max(int(result.confusion.max()) for result in results)
        for axis, result in zip(axes, results, strict=True):
            picture = axis.imshow(result.confusion, cmap="Blues", vmin=0, vmax=maximum)
            for row, col in np.ndindex(10, 10):
                count = result.confusion[row, col]
                axis.text(
                    col,
                    row,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if count > maximum * 0.55 else "black",
                )
            axis.set(
                xticks=range(10),
                yticks=range(10),
                xlabel="Predicted digit",
                ylabel="True digit",
                title=f"{result.name} · accuracy {result.test_accuracy:.1%}",
            )
        figure.colorbar(picture, ax=list(axes), shrink=0.8, label="Test images")
        figure.suptitle("MNIST · Held-out confusion matrices")
        save(figure, "confusion_matrix")

        figure, axes = plt.subplots(3, 10, figsize=(14, 5), layout="constrained")
        limit = max(float(np.abs(result.model.coef_).max()) for result in results)
        limit = max(limit, np.finfo(float).eps)
        for row, result in enumerate(results):
            weights = result.model.coef_
            for digit in range(10):
                axis = axes[row, digit]
                picture = axis.imshow(
                    weights[:, digit].reshape(28, 28),
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    interpolation="nearest",
                )
                axis.set(xticks=[], yticks=[])
                if row == 0:
                    axis.set_title(str(digit))
                if digit == 0:
                    axis.set_ylabel(result.name)
        figure.colorbar(
            picture, ax=list(axes.flat), shrink=0.8, label="Class weight (shared scale)"
        )
        figure.suptitle("MNIST · Learned pixel weights, with zero initialization")
        save(figure, "class_weights")

        figure, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
        for col, (result, color) in enumerate(zip(results, colors, strict=True)):
            data = result.calibration
            occupied = data.counts > 0
            axis = axes[0, col]
            axis.plot(
                [0, 1], [0, 1], linestyle="--", color="0.5", label="Perfect calibration"
            )
            axis.scatter(
                data.mean_confidence[occupied],
                data.accuracy[occupied],
                color=color,
                s=35,
                label="Nonempty bins",
                zorder=3,
            )
            axis.vlines(
                data.mean_confidence[occupied],
                data.mean_confidence[occupied],
                data.accuracy[occupied],
                color=color,
                alpha=0.5,
            )
            axis.set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="Mean confidence in bin",
                ylabel="Accuracy in bin",
                title=f"{result.name} · ECE {data.ece:.4f}",
            )
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, frameon=False, loc="upper left")
            axis = axes[1, col]
            axis.bar(
                data.bin_edges[:-1],
                data.counts / data.counts.sum(),
                width=np.diff(data.bin_edges),
                align="edge",
                color=color,
                alpha=0.8,
                edgecolor="white",
            )
            axis.set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="Predicted confidence",
                ylabel="Fraction of test images",
            )
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle(
            "MNIST · Test-set top-label calibration · "
            f"{config.calibration_bins} equal-width bins"
        )
        save(figure, "calibration")

        for group, name, title in (
            (
                "highest_confidence_errors",
                "confident_errors",
                "Most confident incorrect",
            ),
            (
                "highest_confidence_correct",
                "confident_correct",
                "Most confident correct",
            ),
            ("least_confident", "least_confident", "Least confident"),
        ):
            figure, axes = plt.subplots(3, 6, figsize=(11, 6.5), layout="constrained")
            for row, result in enumerate(results):
                indices = prediction_groups(result, splits.y_test)[group][:6]
                for col in range(6):
                    axis = axes[row, col]
                    axis.set(xticks=[], yticks=[])
                    if col == 0:
                        axis.set_ylabel(result.name)
                    if col < len(indices):
                        i = indices[col]
                        axis.imshow(
                            splits.X_test[i].reshape(28, 28),
                            cmap="gray",
                            vmin=0,
                            vmax=1,
                        )
                        axis.set_title(
                            f"True {splits.y_test[i]} → pred {result.predictions[i]}\n"
                            f"p={result.confidence[i]:.5f} · #{splits.test_indices[i]}",
                            fontsize=9,
                        )
                    else:
                        axis.text(
                            0.5,
                            0.5,
                            "No further examples",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )
            figure.suptitle(f"MNIST · {title} test predictions")
            save(figure, name)


def main() -> None:
    """Run a reproducible command-line benchmark and write figures and metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = ExperimentConfig()
    for name in (
        "seed",
        "train_size",
        "validation_size",
        "test_size",
        "epochs",
        "batch_size",
        "calibration_bins",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name)
        )
    for name in ("l2", "target_nll"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name)
        )
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "mnist.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    try:
        config = ExperimentConfig(
            **{name: getattr(args, name) for name in asdict(defaults)}
        )
        ensure_dataset(args.data_path)
        data = load_dataset(args.data_path)
        splits = make_splits(data, config)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    results = run_experiment(splits, config)
    save_results(results, splits, config, args.data_path, args.output_dir)
    plot_results(results, splits, config, args.output_dir)
    print(f"Saved metrics, weights, and figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
