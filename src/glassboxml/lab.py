"""Experiment comparisons and local prediction explanations for the interactive lab.

These workflows use GlassBoxML models and metrics. Nothing here depends on a UI
framework. Pixel contributions explain a linear logit difference exactly; they
are not causal attributions. Confidence thresholds are descriptive diagnostics.
"""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import center_of_mass, shift, zoom  # type: ignore[import-untyped]

from glassboxml.exploration import (
    MnistExploration,
    RegressionExploration,
    RegressionSettings,
    explore_regression,
)
from glassboxml.models import SoftmaxRegression


def compare_regressions(
    settings: RegressionSettings,
    degrees: tuple[int, ...],
) -> list[tuple[RegressionSettings, RegressionExploration]]:
    """Compare degrees on identical training observations and held-out draws."""
    return [
        (candidate, explore_regression(candidate))
        for degree in degrees
        for candidate in [replace(settings, degree=degree)]
    ]


def regression_report(
    settings: RegressionSettings, result: RegressionExploration
) -> str:
    """Export settings, actual observations, and independently evaluated diagnostics."""
    return json.dumps(
        {
            "settings": asdict(settings),
            "observations": {"x": result.x.tolist(), "y": result.y.tolist()},
            "posterior_mean_weights": result.weights.tolist(),
            "metrics": {
                "training_mse": result.training_mse,
                "test_mse": result.test_mse,
                "test_coverage_95": result.test_coverage,
            },
            "test_protocol": "1000 noisy draws in [-1,1]; SeedSequence([seed,275])",
            "interval": "pointwise 95% future observation; fixed alpha and beta",
        },
        indent=2,
        allow_nan=False,
    )


def _image(values: ArrayLike) -> NDArray[np.float64]:
    image = np.asarray(values)
    if image.shape != (28, 28) or image.dtype.kind not in "iuf":
        raise ValueError("image must be a real (28, 28) array scaled to [0, 1]")
    result = np.asarray(image, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError("image values must be finite and in [0, 1]")
    return result.copy()


def normalized_digit(data: MnistExploration, index: int) -> NDArray[np.float64]:
    """Return a fresh model-ready test image, with the artifact's pixel scaling."""
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(data.images)
    ):
        raise ValueError("index must identify an image in the test split")
    return np.asarray(data.images[index], dtype=np.float64) / 255.0


def prepare_drawing(pixels: ArrayLike) -> NDArray[np.float64] | None:
    """Crop white ink, fit into a 20-pixel box, and center mass on a 28×28 canvas.

    Accept square grayscale drawings with 16..512 pixels per side in [0,255].
    Blank drawings, including ink lost during resampling, return None.
    Bilinear resampling uses SciPy image primitives;
    this is deterministic preprocessing, not a learned transformation. Sketches
    can still differ substantially from the MNIST training distribution.
    """
    image = np.asarray(pixels)
    if (
        image.ndim != 2
        or image.shape[0] != image.shape[1]
        or not 16 <= image.shape[0] <= 512
        or image.dtype.kind not in "iuf"
    ):
        raise ValueError(
            "drawing must be a square grayscale array, 16..512 pixels wide"
        )
    image = np.asarray(image, dtype=np.float64)
    if not np.all(np.isfinite(image)) or np.any((image < 0) | (image > 255)):
        raise ValueError("drawing pixels must be finite and in [0, 255]")
    foreground = np.argwhere(image > 20)
    if len(foreground) == 0:
        return None
    low, high = foreground.min(axis=0), foreground.max(axis=0) + 1
    crop = image[low[0] : high[0], low[1] : high[1]] / 255
    target = np.maximum(1, np.rint(np.array(crop.shape) * 20 / max(crop.shape))).astype(
        int
    )
    resized = zoom(crop, target / np.array(crop.shape), order=1, prefilter=False)
    if not np.any(resized):
        return None
    result = np.zeros((28, 28))
    top, left = (28 - target) // 2
    result[top : top + target[0], left : left + target[1]] = resized
    cy, cx = center_of_mass(result)
    return np.asarray(
        shift(
            result,
            (13.5 - cy, 13.5 - cx),
            order=1,
            mode="constant",
            cval=0,
            prefilter=False,
        ),
        dtype=np.float64,
    )


def perturb_digit(
    image: ArrayLike,
    *,
    shift_x: int = 0,
    shift_y: int = 0,
    noise: float = 0,
    hide_rows: tuple[int, int] = (0, 0),
    seed: int = 42,
) -> NDArray[np.float64]:
    """Shift with zero padding, add seeded noise, then occlude a half-open row range."""
    values = _image(image)
    for offset in (shift_x, shift_y):
        if isinstance(offset, bool) or not isinstance(offset, int) or abs(offset) > 28:
            raise ValueError("shifts must be integer pixels between -28 and 28")
    if not np.isfinite(noise) or not 0 <= noise <= 1:
        raise ValueError("noise must lie in [0, 1]")
    if (
        len(hide_rows) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in hide_rows)
        or not 0 <= hide_rows[0] <= hide_rows[1] <= 28
    ):
        raise ValueError("hide_rows must be an ordered pair in [0, 28]")
    shifted = shift(values, (shift_y, shift_x), order=0, mode="constant", cval=0)
    result = np.clip(
        shifted + np.random.default_rng(seed).normal(0, noise, (28, 28)), 0, 1
    )
    result[hide_rows[0] : hide_rows[1]] = 0
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True)
class DigitInspection:
    """Prediction and additive evidence for the winner versus the runner-up."""

    image: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    prediction: int
    runner_up: int
    confidence: float
    contributions: NDArray[np.float64]
    bias_difference: float
    logit_margin: float


def inspect_digit(data: MnistExploration, image: ArrayLike) -> DigitInspection:
    """Predict via SoftmaxRegression and expose exact per-pixel logit contributions.

    For winner a and runner-up b, z_a-z_b = sum_j x_j(W_ja-W_jb)+(b_a-b_b).
    The heatmap is signed evidence in this specific linear comparison.
    """
    values = _image(image)
    model = SoftmaxRegression.from_parameters(data.weights, data.bias, data.classes)
    flat = values.reshape(1, -1)
    probabilities = model.predict_proba(flat)[0]
    winner, runner = np.argsort(-probabilities, kind="stable")[:2]
    contributions = values * (
        data.weights[:, winner] - data.weights[:, runner]
    ).reshape(28, 28)
    bias = float(data.bias[winner] - data.bias[runner])
    return DigitInspection(
        values,
        probabilities,
        int(data.classes[winner]),
        int(data.classes[runner]),
        float(probabilities[winner]),
        contributions,
        bias,
        float(contributions.sum() + bias),
    )


def confidence_tradeoff(
    data: MnistExploration, threshold: float
) -> dict[str, float | int | None]:
    """Measure coverage and accuracy after abstaining below a confidence threshold."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    retained = data.confidence >= threshold
    count = int(retained.sum())
    return {
        "retained": count,
        "coverage": count / len(retained),
        "accuracy": float(np.mean(data.predictions[retained] == data.labels[retained]))
        if count
        else None,
    }


def training_traces(path: Path) -> list[dict[str, Any]]:
    """Read saved epoch traces from JSON, distinct from recomputed diagnostics."""
    document = json.loads(path.read_text())
    rows = []
    for name, run in document.get("runs", {}).items():
        for key, label in (
            ("train_loss", "Training"),
            ("validation_loss", "Validation"),
        ):
            for epoch, loss in enumerate(run.get("history", {}).get(key, []), start=1):
                if (
                    not isinstance(loss, (int, float))
                    or not np.isfinite(loss)
                    or loss < 0
                ):
                    raise ValueError(
                        "training histories must contain finite nonnegative losses"
                    )
                rows.append(
                    {"Optimizer": name, "Split": label, "Epoch": epoch, "NLL": loss}
                )
    return rows
