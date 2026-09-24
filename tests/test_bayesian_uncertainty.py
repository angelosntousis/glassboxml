"""Nested-data experiment statistics and saved artifacts."""

import csv
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "bayesian_uncertainty.py"


@pytest.fixture(scope="module")
def experiment():
    return runpy.run_path(str(SCRIPT))


def test_nested_experiment_is_reproducible_and_uncertainty_contracts(experiment):
    config = experiment["ExperimentConfig"]()
    first = experiment["run_experiment"](config)
    second = experiment["run_experiment"](config)
    x, y, grid, predictions = first
    for a, b in zip(first[:3], second[:3], strict=True):
        assert_array_equal(a, b)
    assert x.shape == y.shape == (100,)
    assert np.all((-1 <= x) & (x <= 1))
    assert grid[0] < -1 and grid[-1] > 1
    assert [p.n for p in predictions] == [5, 20, 100]
    for prediction, repeated in zip(predictions, second[3], strict=True):
        assert_array_equal(prediction.mean, repeated.mean)
        assert_array_equal(prediction.epistemic_std, repeated.epistemic_std)
        assert_array_equal(prediction.predictive_std, repeated.predictive_std)
        assert prediction.mean.shape == grid.shape
        assert np.all(prediction.predictive_std >= config.noise_std)
        assert_allclose(
            prediction.predictive_std**2,
            prediction.epistemic_std**2 + config.noise_std**2,
            rtol=1e-12,
        )
    for smaller, larger in zip(predictions, predictions[1:], strict=False):
        assert larger.posterior_trace < smaller.posterior_trace
        assert np.all(larger.epistemic_std <= smaller.epistemic_std + 1e-12)
        assert np.all(larger.predictive_std <= smaller.predictive_std + 1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": True},
        {"seed": 1.5},
        {"alpha": 0},
        {"alpha": np.inf},
        {"alpha": np.nan},
        {"noise_std": 0},
        {"noise_std": -1},
        {"noise_std": np.nan},
        {"noise_std": 1e-300},
        {"noise_std": 1e300},
    ],
)
def test_invalid_experiment_configuration(experiment, kwargs):
    with pytest.raises(ValueError):
        experiment["ExperimentConfig"](**kwargs)


def test_cli_saves_uncertainty_curves_summaries_and_figures(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "artifacts"
    environment = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "matplotlib"))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(output)],
        env=environment,
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "Mean epistemic SD" in result.stdout
    with (output / "bayesian_uncertainty_training.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 100
    with (output / "bayesian_uncertainty_predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 601
    assert {int(row["n"]) for row in rows} == {5, 20, 100}
    with (output / "bayesian_uncertainty_summary.csv").open() as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == 3
    assert all(float(row["mean_predictive_std"]) > 0.25 for row in summaries)
    metadata = json.loads((output / "bayesian_uncertainty_metadata.json").read_text())
    assert metadata["sample_sizes"] == [5, 20, 100]
    assert metadata["beta"] == 16
    assert "matplotlib" in metadata
    for name in ("bayesian_uncertainty", "bayesian_uncertainty_components"):
        png = output / "figures" / f"{name}.png"
        pdf = output / "figures" / f"{name}.pdf"
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert pdf.read_bytes().startswith(b"%PDF-")
        assert png.stat().st_size > 10000
        assert pdf.stat().st_size > 10000
