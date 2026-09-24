"""Independent checks for model comparison, drawing preparation, and explanations."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from glassboxml.exploration import RegressionSettings, explore_mnist, explore_regression
from glassboxml.lab import (
    compare_regressions,
    confidence_tradeoff,
    inspect_digit,
    normalized_digit,
    perturb_digit,
    prepare_drawing,
    regression_report,
    training_traces,
)


@pytest.fixture
def digit_data(tmp_path):
    rng = np.random.default_rng(40)
    images = rng.integers(0, 256, (30, 28, 28), dtype=np.uint8)
    dataset = tmp_path / "mnist.npz"
    artifact = tmp_path / "weights.npz"
    np.savez(dataset, x_test=images, y_test=np.arange(30) % 10)
    # Deliberately permute labels: explanations must respect class-column order.
    np.savez(
        artifact,
        Adam_weights=rng.normal(0, 0.1, (784, 10)),
        Adam_bias=rng.normal(size=10),
        Adam_classes=np.arange(10)[::-1],
    )
    return explore_mnist(artifact, dataset, "Adam")


def test_comparisons_share_data_and_holdout_and_export_real_observations():
    settings = RegressionSettings()
    comparison = compare_regressions(settings, (1, 3, 9))
    assert [s.degree for s, _ in comparison] == [1, 3, 9]
    baseline = comparison[0][1]
    for setting, result in comparison:
        assert_array_equal(result.x, baseline.x)
        assert_array_equal(result.y, baseline.y)
        repeated = explore_regression(setting)
        assert result.test_mse == repeated.test_mse
        assert result.test_coverage == repeated.test_coverage
        assert_array_equal(result.posterior_draws, repeated.posterior_draws)
        assert result.posterior_draws.shape == (401, 6)
        assert np.all(result.lower <= result.latent_lower)
        assert np.all(result.upper >= result.latent_upper)
        assert_allclose(
            ((result.upper - result.lower) / (2 * 1.959963984540054)) ** 2,
            result.epistemic_std**2 + settings.noise_std**2,
        )
        report = json.loads(regression_report(setting, result))
        assert report["settings"]["degree"] == setting.degree
        assert report["observations"]["y"] == result.y.tolist()
        assert report["metrics"]["test_mse"] == result.test_mse


def test_drawing_is_cropped_centered_and_bounded_without_mutation():
    image = np.zeros((280, 280), dtype=np.uint8)
    image[15:135, 20:45] = 255
    before = image.copy()
    prepared = prepare_drawing(image)
    assert prepared is not None
    assert prepared.shape == (28, 28)
    assert prepared.dtype == np.float64
    assert prepared.min() >= 0 and prepared.max() <= 1
    ys, xs = np.indices(prepared.shape)
    assert np.sum(ys * prepared) / prepared.sum() == pytest.approx(13.5)
    assert np.sum(xs * prepared) / prepared.sum() == pytest.approx(13.5)
    assert_array_equal(image, before)
    assert prepare_drawing(np.zeros((280, 280))) is None
    assert_allclose(prepare_drawing(image), prepared)


def test_sparse_ink_lost_in_resampling_is_treated_as_blank():
    image = np.zeros((280, 280))
    image[[0, 279, 130, 130], [130, 130, 0, 279]] = 255
    assert prepare_drawing(image) is None


@pytest.mark.parametrize(
    "image",
    [
        np.zeros(28),
        np.zeros((28, 20)),
        np.zeros((8, 8)),
        np.full((28, 28), -1),
        np.full((28, 28), 256),
        np.full((28, 28), np.nan),
    ],
)
def test_invalid_drawings(image):
    with pytest.raises(ValueError):
        prepare_drawing(image)


def test_perturbations_are_reproducible_do_not_wrap_and_do_not_mutate():
    image = np.zeros((28, 28))
    image[10, 27] = 1
    original = image.copy()
    assert_array_equal(perturb_digit(image), image)
    assert perturb_digit(image, shift_x=1).sum() == 0
    assert perturb_digit(image, shift_y=1)[11, 27] == 1
    assert_array_equal(perturb_digit(image, noise=0.2), perturb_digit(image, noise=0.2))
    hidden = perturb_digit(image, noise=0.2, hide_rows=(9, 12))
    assert np.all(hidden[9:12] == 0)
    assert hidden.min() >= 0 and hidden.max() <= 1
    assert_array_equal(image, original)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"noise": -1},
        {"noise": np.nan},
        {"shift_x": 29},
        {"shift_y": 0.5},
        {"hide_rows": (20, 10)},
    ],
)
def test_invalid_perturbations(kwargs):
    with pytest.raises(ValueError):
        perturb_digit(np.zeros((28, 28)), **kwargs)


def test_local_evidence_exactly_decomposes_logit_margin(digit_data):
    image = normalized_digit(digit_data, 0)
    inspection = inspect_digit(digit_data, image)
    assert_allclose(inspection.probabilities, digit_data.probabilities[0])
    winner = int(np.flatnonzero(digit_data.classes == inspection.prediction)[0])
    runner = int(np.flatnonzero(digit_data.classes == inspection.runner_up)[0])
    logits = image.reshape(-1) @ digit_data.weights + digit_data.bias
    assert inspection.logit_margin == pytest.approx(logits[winner] - logits[runner])
    assert inspection.contributions.sum() + inspection.bias_difference == pytest.approx(
        inspection.logit_margin
    )
    assert inspection.confidence == max(inspection.probabilities)
    assert_array_equal(image, normalized_digit(digit_data, 0))


def test_abstention_uses_inclusive_threshold_and_handles_empty_selection(digit_data):
    all_images = confidence_tradeoff(digit_data, 0)
    assert all_images["coverage"] == 1
    assert all_images["accuracy"] == digit_data.accuracy
    none = confidence_tradeoff(digit_data, 1)
    assert none == {"retained": 0, "coverage": 0, "accuracy": None}
    threshold = float(digit_data.confidence[0])
    selection = digit_data.confidence >= threshold
    filtered = confidence_tradeoff(digit_data, threshold)
    assert filtered["retained"] == selection.sum()
    assert filtered["accuracy"] == np.mean(
        digit_data.predictions[selection] == digit_data.labels[selection]
    )
    with pytest.raises(ValueError):
        confidence_tradeoff(digit_data, 1.1)


def test_training_traces_are_labeled_and_validate_loss(tmp_path):
    path = tmp_path / "metrics.json"
    document = {
        "runs": {
            "SGD": {
                "history": {"train_loss": [1.2, 0.5], "validation_loss": [1.3, 0.6]}
            }
        }
    }
    path.write_text(json.dumps(document))
    rows = training_traces(path)
    assert len(rows) == 4
    assert rows[-1] == {
        "Optimizer": "SGD",
        "Epoch": 2,
        "Split": "Validation",
        "NLL": 0.6,
    }
    document["runs"]["SGD"]["history"]["train_loss"] = [-1]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        training_traces(path)
