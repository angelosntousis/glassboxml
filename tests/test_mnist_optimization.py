"""Offline tests for fair splitting, optimizer runs, metrics, and figure artifacts."""

import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from glassboxml.metrics import expected_calibration_error

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "mnist_optimization.py"


@pytest.fixture(scope="module")
def experiment():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture
def archive(tmp_path):
    """Small local image fixture, not a substitute dataset for real experiments."""
    rng = np.random.default_rng(8)
    arrays = {}
    for pool, size in (("train", 300), ("test", 50)):
        labels = np.arange(size, dtype=np.uint8) % 10
        images = rng.integers(0, 15, size=(size, 28, 28), dtype=np.uint8)
        images.reshape(size, -1)[np.arange(size), labels] = 255
        arrays[f"x_{pool}"], arrays[f"y_{pool}"] = images, labels
    path = tmp_path / "fixture.npz"
    np.savez(path, **arrays)
    return path


@pytest.fixture
def config(experiment):
    return experiment["ExperimentConfig"](
        train_size=200,
        validation_size=60,
        test_size=40,
        epochs=3,
        batch_size=64,
    )


@pytest.fixture
def splits(experiment, archive, config):
    return experiment["make_splits"](experiment["load_dataset"](archive), config)


def test_disjoint_seeded_splits_and_fixed_scaling(experiment, archive, config, splits):
    data = experiment["load_dataset"](archive)
    repeated = experiment["make_splits"](data, config)
    assert np.intersect1d(splits.train_indices, splits.validation_indices).size == 0
    for name in ("train_indices", "validation_indices", "test_indices"):
        indices = getattr(splits, name)
        assert_array_equal(indices, getattr(repeated, name))
        assert np.unique(indices).size == indices.size
    assert splits.X_train.shape == (200, 784)
    assert splits.X_train.dtype == np.float64
    assert_array_equal(splits.y_train, data.train_labels[splits.train_indices])
    assert_array_equal(
        splits.y_validation, data.train_labels[splits.validation_indices]
    )
    assert_array_equal(splits.y_test, data.test_labels[splits.test_indices])
    assert_allclose(
        splits.X_train, data.train_images[splits.train_indices].reshape(200, -1) / 255
    )
    changed = experiment["make_splits"](data, replace(config, seed=43))
    assert not np.array_equal(splits.train_indices, changed.train_indices)
    assert not np.array_equal(splits.test_indices, changed.test_indices)
    # Test selection is independent of how the training pool is partitioned.
    resized = experiment["make_splits"](data, replace(config, train_size=150))
    assert_array_equal(splits.test_indices, resized.test_indices)
    full_test = experiment["make_splits"](data, replace(config, test_size=50))
    assert_array_equal(full_test.test_indices, np.arange(50))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": True},
        {"epochs": 0},
        {"batch_size": 0},
        {"train_size": 9},
        {"validation_size": 0},
        {"test_size": 1.5},
        {"l2": -1},
        {"l2": np.nan},
        {"target_nll": np.inf},
        {"target_nll": -1},
    ],
)
def test_invalid_config(experiment, kwargs):
    with pytest.raises(ValueError):
        experiment["ExperimentConfig"](**kwargs)


def test_invalid_split_sizes_and_missing_training_class(experiment, archive, config):
    data = experiment["load_dataset"](archive)
    for invalid in (replace(config, train_size=300), replace(config, test_size=51)):
        with pytest.raises(ValueError, match="pool"):
            experiment["make_splits"](data, invalid)
    data.train_labels[data.train_labels == 9] = 8
    with pytest.raises(ValueError, match="all ten classes"):
        experiment["make_splits"](data, config)


@pytest.mark.parametrize("change", ["shape", "dtype", "labels", "label_shape"])
def test_invalid_archive(experiment, archive, change):
    with np.load(archive) as data:
        arrays = dict(data)
    if change == "shape":
        arrays["x_train"] = arrays["x_train"][:, :27]
    elif change == "dtype":
        arrays["x_test"] = arrays["x_test"].astype(float)
    elif change == "labels":
        arrays["y_test"][0] = 10
    else:
        arrays["y_train"] = arrays["y_train"][:, None]
    np.savez(archive, **arrays)
    with pytest.raises(ValueError):
        experiment["load_dataset"](archive)


def test_download_cache_and_failed_download_cleanup(
    experiment, archive, tmp_path, monkeypatch
):
    request = experiment["urllib"].request
    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(archive.read_bytes())

    monkeypatch.setattr(request, "urlopen", download)
    cached = tmp_path / "cache" / "mnist.npz"
    experiment["ensure_dataset"](cached)
    experiment["ensure_dataset"](cached)
    assert cached.read_bytes() == archive.read_bytes()
    assert calls == [(experiment["DATA_URL"], 60)]
    monkeypatch.setattr(request, "urlopen", lambda *a, **k: io.BytesIO(b"broken"))
    missing = tmp_path / "broken" / "mnist.npz"
    with pytest.raises(ValueError):
        experiment["ensure_dataset"](missing)
    assert not missing.exists()
    assert list(missing.parent.iterdir()) == []


def test_nll_handles_probability_underflow_and_logit_shift(experiment):
    nll = experiment["negative_log_likelihood"]
    logits = np.array([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 0.0]])
    labels = np.array([1, 2])
    assert nll(logits, labels) == pytest.approx(1500)
    assert nll(logits + 1e6, labels) == pytest.approx(1500)
    assert nll(np.zeros((5, 10)), np.arange(5)) == pytest.approx(np.log(10))


def test_convergence_has_exact_epochs_and_partial_batch_updates(experiment, config):
    convergence = experiment["convergence_metrics"]([0.8, 0.35, 0.36, 0.3], config)
    assert convergence["epoch_to_target"] == 2
    assert convergence["updates_to_target"] == 8  # ceil(200 / 64) * 2
    assert convergence["best_validation_epoch"] == 4
    missed = experiment["convergence_metrics"]([0.8, 0.4], config)
    assert missed["epoch_to_target"] is None
    assert missed["updates_to_target"] is None


def test_runs_are_reproducible_and_metrics_match_predictions(
    experiment, splits, config
):
    first = experiment["run_experiment"](splits, config)
    second = experiment["run_experiment"](splits, config)
    assert [r.name for r in first] == ["SGD", "Momentum", "Adam"]
    for result, repeated in zip(first, second, strict=True):
        assert_array_equal(result.model.coef_, repeated.model.coef_)
        assert result.model.history_ == repeated.model.history_
        assert result.fit_seconds > 0
        assert result.model.n_iter_ == config.epochs
        assert result.model.random_state == config.seed
        assert result.model.early_stopping is False
        assert result.model.shuffle is True
        assert result.model.history_["train_loss"][-1] < np.log(10)
        assert result.confusion.sum() == config.test_size
        assert_allclose(
            result.confusion.sum(axis=1), np.bincount(splits.y_test, minlength=10)
        )
        assert result.test_accuracy == pytest.approx(
            np.trace(result.confusion) / config.test_size
        )
        probabilities = result.model.predict_proba(splits.X_test)
        assert result.test_nll == pytest.approx(
            -np.log(probabilities[np.arange(config.test_size), splits.y_test]).mean()
        )
        assert result.calibration.ece == pytest.approx(
            expected_calibration_error(
                splits.y_test, probabilities, n_bins=config.calibration_bins
            )
        )
        assert result.calibration.counts.sum() == config.test_size
        errors = experiment["confident_error_indices"](result, splits.y_test)
        assert np.all(result.predictions[errors] != splits.y_test[errors])
        assert np.all(np.diff(result.confidence[errors]) <= 0)
        # Exercise the no-mistakes case used by the figure layout.
        assert (
            experiment["confident_error_indices"](result, result.predictions).size == 0
        )


def test_prediction_groups_rank_correct_errors_and_uncertain_with_stable_ties(
    experiment,
):
    result = SimpleNamespace(
        predictions=np.array([0, 1, 2, 1, 0]),
        confidence=np.array([0.9, 0.95, 0.9, 0.6, 0.5]),
    )
    groups = experiment["prediction_groups"](result, np.array([0, 0, 2, 2, 0]))
    assert_array_equal(groups["highest_confidence_correct"], [0, 2, 4])
    assert_array_equal(groups["highest_confidence_errors"], [1, 3])
    assert_array_equal(groups["least_confident"], [4, 3, 0, 2, 1])
    all_correct = experiment["prediction_groups"](result, result.predictions)
    assert all_correct["highest_confidence_errors"].size == 0
    all_wrong = experiment["prediction_groups"](result, result.predictions + 1)
    assert all_wrong["highest_confidence_correct"].size == 0


def test_cli_writes_metrics_weights_and_all_figures(experiment, archive, tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "artifacts"
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "mpl"))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-path",
            str(archive),
            "--output-dir",
            str(output),
            "--train-size",
            "200",
            "--validation-size",
            "60",
            "--test-size",
            "40",
            "--epochs",
            "2",
            "--batch-size",
            "64",
            "--calibration-bins",
            "7",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert "Saved metrics" in result.stdout
    metrics = json.loads((output / "mnist_optimization_metrics.json").read_text())
    assert (
        metrics["dataset"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert metrics["config"]["seed"] == 42
    assert metrics["config"]["calibration_bins"] == 7
    assert len(metrics["splits"]["train_pool_indices"]) == 200
    assert len(metrics["splits"]["validation_pool_indices"]) == 60
    assert len(metrics["splits"]["official_test_indices"]) == 40
    assert "matplotlib" in metrics["environment"]
    with np.load(
        output / "mnist_optimization_weights.npz", allow_pickle=False
    ) as weights:
        for name, run in metrics["runs"].items():
            assert weights[f"{name}_weights"].shape == (784, 10)
            assert weights[f"{name}_bias"].shape == (10,)
            assert_array_equal(weights[f"{name}_classes"], np.arange(10))
            assert len(run["history"]["train_loss"]) == 2
            assert len(run["history"]["validation_loss"]) == 2
            assert 0 <= run["test_accuracy"] <= 1
            assert run["test_nll"] >= 0
            assert run["total_updates"] == 8
            assert np.array(run["confusion_matrix"]).sum() == 40
            calibration = run["calibration"]
            assert len(calibration["bin_edges"]) == 8
            assert sum(calibration["counts"]) == 40
            manual_ece = 0.0
            for count, score, accuracy in zip(
                calibration["counts"],
                calibration["mean_confidence"],
                calibration["accuracy"],
                strict=True,
            ):
                if count == 0:
                    assert score is None and accuracy is None
                else:
                    manual_ece += count / 40 * abs(score - accuracy)
            assert run["test_ece"] == pytest.approx(manual_ece)
            for group in (
                "highest_confidence_correct",
                "highest_confidence_errors",
                "least_confident",
            ):
                examples = run[group]
                assert len(examples) <= 6
                for example in examples:
                    assert (
                        example["official_test_index"]
                        in metrics["splits"]["official_test_indices"]
                    )
                    if group != "least_confident":
                        assert (example["true_label"] == example["prediction"]) == (
                            group == "highest_confidence_correct"
                        )
                scores = [example["confidence"] for example in examples]
                assert scores == sorted(scores, reverse=group != "least_confident")
    for name in experiment["FIGURE_NAMES"]:
        png = output / "figures" / f"mnist_{name}.png"
        pdf = output / "figures" / f"mnist_{name}.pdf"
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert pdf.read_bytes().startswith(b"%PDF-")
        assert png.stat().st_size > 10000
        assert pdf.stat().st_size > 10000
