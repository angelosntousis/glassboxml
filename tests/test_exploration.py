"""Framework-independent explorer workflows, validation, and artifact inference."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from glassboxml.exploration import (
    RegressionSettings,
    available_softmax_models,
    explore_mnist,
    explore_regression,
    load_softmax_artifact,
)
from glassboxml.models import BayesianLinearRegression, SoftmaxRegression
from glassboxml.preprocessing import PolynomialFeatures


def test_regression_workflow_matches_model_and_preserves_observations():
    settings = RegressionSettings()
    result = explore_regression(settings)
    repeated = explore_regression(settings)
    assert_array_equal(result.x, repeated.x)
    assert_array_equal(result.y, repeated.y)
    features = PolynomialFeatures(3, include_bias=True)
    model = BayesianLinearRegression(alpha=1, beta=25).fit(
        features.transform(result.x), result.y
    )
    mean, std = model.predict(features.transform(result.grid), return_std=True)
    assert_allclose(result.mean, mean)
    assert_allclose(result.upper, mean + 1.959963984540054 * std)
    assert_allclose(result.lower, mean - 1.959963984540054 * std)
    assert np.all(result.lower < result.upper)
    assert result.training_mse >= 0
    assert result.grid[0] < -1 and result.grid[-1] > 1
    changed = explore_regression(replace(settings, alpha=10, noise_std=0.4, degree=5))
    assert_array_equal(result.x, changed.x)
    assert_array_equal(result.y, changed.y)
    assert not np.array_equal(result.mean, changed.mean)
    new_seed = explore_regression(replace(settings, seed=43))
    assert not np.array_equal(result.y, new_seed.y)


@pytest.mark.parametrize("signal", ["Cubic", "Sine", "Linear"])
@pytest.mark.parametrize("sampling", ["Random", "Evenly spaced", "Gap in the middle"])
def test_regression_presets_and_uncertainty(signal, sampling):
    result = explore_regression(RegressionSettings(signal=signal, sampling=sampling))
    assert result.x.size == 20
    assert np.all(np.diff(result.x) >= 0)
    assert np.all(np.abs(result.x) <= 1)
    assert np.all(result.upper - result.lower >= 2 * 1.959963984540054 * 0.2)
    if sampling == "Gap in the middle":
        assert np.all(np.abs(result.x) >= 0.35)
    if signal == "Sine":
        assert_allclose(result.truth, np.sin(np.pi * result.grid))
    if signal == "Linear":
        assert_allclose(result.truth, 0.5 + 1.2 * result.grid)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 1},
        {"seed": -1},
        {"seed": True},
        {"degree": -1},
        {"alpha": 0},
        {"noise_std": 0},
        {"generation_noise_std": np.nan},
        {"signal": "unknown"},
        {"sampling": "unknown"},
    ],
)
def test_invalid_regression_settings(kwargs):
    with pytest.raises(ValueError):
        RegressionSettings(**kwargs)


@pytest.fixture
def artifacts(tmp_path):
    images = np.zeros((30, 28, 28), dtype=np.uint8)
    images[:, 10:20, 10:20] = 255
    labels = np.arange(30, dtype=np.uint8) % 10
    dataset = tmp_path / "mnist.npz"
    np.savez(dataset, x_test=images, y_test=labels)
    weights = tmp_path / "weights.npz"
    np.savez(
        weights,
        Adam_weights=np.zeros((784, 10)),
        Adam_bias=np.arange(10.0),
        Adam_classes=np.arange(10),
        SGD_weights=np.zeros((784, 10)),
        SGD_bias=-np.arange(10.0),
        SGD_classes=np.arange(10),
    )
    metadata = tmp_path / "metrics.json"
    metadata.write_text(
        json.dumps(
            {
                "dataset": {"sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()},
                "splits": {"official_test_indices": [29, 3, 8, 0, 9]},
                "config": {"calibration_bins": 5},
            }
        )
    )
    return weights, dataset, metadata


def test_mnist_artifact_split_inference_and_diagnostics(artifacts):
    weights, dataset, metadata = artifacts
    assert available_softmax_models(weights) == ("Adam", "SGD")
    restored = load_softmax_artifact(weights, "Adam")
    result = explore_mnist(weights, dataset, "Adam", metadata)
    assert_array_equal(result.indices, [29, 3, 8, 0, 9])
    assert_array_equal(result.labels, [9, 3, 8, 0, 9])
    assert_array_equal(result.predictions, [9] * 5)
    assert result.accuracy == 0.4
    assert result.confusion.sum() == 5
    assert result.calibration.counts.sum() == 5
    assert len(result.calibration.counts) == 5
    assert_array_equal(result.groups["Most confident correct"], [0, 4])
    assert_array_equal(result.groups["Most confident incorrect"], [1, 2, 3])
    assert_array_equal(result.groups["Least confident"], np.arange(5))
    assert all(
        row["True digit"] != row["Predicted digit"] for row in result.common_errors
    )
    assert sum(row["Count"] for row in result.common_errors) == 3
    assert_allclose(result.uncertainty, 1 - result.confidence)
    assert_allclose(
        result.probabilities, restored.predict_proba(result.images.reshape(5, -1) / 255)
    )
    assert explore_mnist(weights, dataset, "SGD").accuracy == 0.1


@pytest.mark.parametrize("indices", [[0, 0], [-1], [30], [], [1.5], [[0, 1]]])
def test_invalid_metadata_indices(artifacts, indices):
    weights, dataset, metadata = artifacts
    contents = json.loads(metadata.read_text())
    contents["splits"]["official_test_indices"] = indices
    metadata.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="indices"):
        explore_mnist(weights, dataset, "Adam", metadata)


def test_corrupt_or_mismatched_artifacts(artifacts):
    weights, dataset, metadata = artifacts
    contents = json.loads(metadata.read_text())
    contents["dataset"]["sha256"] = "incorrect"
    metadata.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="checksum"):
        explore_mnist(weights, dataset, "Adam", metadata)
    with pytest.raises(ValueError, match="missing model"):
        load_softmax_artifact(weights, "missing")
    np.savez(
        weights,
        Adam_weights=np.zeros((4, 2)),
        Adam_bias=np.zeros(2),
        Adam_classes=[0, 1],
    )
    with pytest.raises(ValueError, match="784"):
        explore_mnist(weights, dataset, "Adam")
    np.savez(weights, other=np.zeros(2))
    with pytest.raises(ValueError, match="No models found"):
        available_softmax_models(weights)


def test_plain_npy_file_has_a_clear_archive_error(tmp_path):
    path = tmp_path / "array.npy"
    np.save(path, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="NPZ archive"):
        available_softmax_models(path)


def test_restored_softmax_matches_training_and_owns_parameters():
    X = np.array([[-2.0, -1], [-1.0, -2], [2.0, -1], [1.0, -2], [0.0, 2], [0.5, 3]])
    fitted = SoftmaxRegression(max_epochs=30, random_state=42).fit(
        X, [2, 2, 4, 4, 8, 8]
    )
    weights, bias, classes = fitted.coef_, fitted.intercept_, fitted.classes_
    restored = SoftmaxRegression.from_parameters(weights, bias, classes)
    assert_array_equal(restored.predict_proba(X), fitted.predict_proba(X))
    weights[:], bias[:], classes[:] = 0, 0, 0
    assert_array_equal(restored.predict(X), fitted.predict(X))
    assert restored.n_iter_ == 0
    assert restored.best_epoch_ is None
    with pytest.raises(RuntimeError, match="history"):
        _ = restored.history_
    reversed_model = SoftmaxRegression.from_parameters(
        fitted.coef_[:, ::-1], fitted.intercept_[::-1], fitted.classes_[::-1]
    )
    assert_array_equal(reversed_model.predict(X), fitted.predict(X))


@pytest.mark.parametrize(
    "weights,bias,labels",
    [
        (np.zeros((2, 1)), [0], [0]),
        (np.zeros((2, 2)), [0], [0, 1]),
        (np.zeros((2, 2)), [0, 0], [0, 0]),
        (np.zeros((2, 2)), [0, 0], [0]),
        (np.zeros((2, 2)), [np.nan, 0], [0, 1]),
        (np.full((2, 2), np.inf), [0, 0], [0, 1]),
    ],
)
def test_invalid_restored_softmax_parameters(weights, bias, labels):
    with pytest.raises(ValueError):
        SoftmaxRegression.from_parameters(weights, bias, labels)
