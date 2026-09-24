"""Experiment reproducibility, honest aggregation, and end-to-end artifacts."""

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

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "bias_variance.py"


@pytest.fixture(scope="module")
def experiment():
    return runpy.run_path(str(SCRIPT))


def test_reproducible_paired_experiment_and_summary_statistics(experiment):
    config = experiment["ExperimentConfig"](
        seeds=3, train_size=25, validation_size=30, test_size=35, alphas=(0.01, 1.0)
    )
    first = experiment["run_experiment"](config)
    assert first == experiment["run_experiment"](config)
    assert len(first) == 3 * 7 * 3
    assert {row.degree for row in first} == {1, 2, 3, 5, 8, 12, 20}
    assert {row.alpha for row in first} == {None, 0.01, 1.0}
    assert all(row.train_mse >= 0 and row.test_mse >= 0 for row in first)
    summaries = experiment["summarize"](first)
    assert len(summaries) == 21
    for summary in summaries:
        group = [
            r for r in first if (r.alpha, r.degree) == (summary.alpha, summary.degree)
        ]
        assert {row.seed for row in group} == {0, 1, 2}
        assert summary.n_seeds == 3
        for split in ("train", "validation", "test"):
            values = [getattr(r, f"{split}_mse") for r in group]
            assert getattr(summary, f"{split}_mse_mean") == pytest.approx(
                np.mean(values)
            )
            assert getattr(summary, f"{split}_mse_std") == pytest.approx(
                np.std(values, ddof=1)
            )
    # Nested OLS feature spaces cannot increase optimal training error.
    for seed in range(3):
        errors = [r.train_mse for r in first if r.seed == seed and r.model == "OLS"]
        assert np.all(np.diff(errors) <= 1e-8)


def test_split_streams_are_independent_and_training_is_unchanged(experiment):
    config_type = experiment["ExperimentConfig"]
    generate = experiment["generate_data"]
    first = generate(7, config_type(train_size=30, validation_size=30, test_size=30))
    changed_test = generate(
        7, config_type(train_size=30, validation_size=30, test_size=50)
    )
    for split in ("train", "validation"):
        for actual, expected in zip(first[split], changed_test[split], strict=True):
            assert_array_equal(actual, expected)
    assert not np.array_equal(first["train"][0], first["validation"][0])
    assert not np.array_equal(first["train"][0], first["test"][0])
    second_seed = generate(
        8, config_type(train_size=30, validation_size=30, test_size=30)
    )
    assert not np.array_equal(first["train"][0], second_seed["train"][0])
    for x, y in first.values():
        assert np.all((-1 <= x) & (x <= 1))
        assert x.shape == y.shape == (30,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seeds": 1},
        {"seeds": 2.5},
        {"seed_start": -1},
        {"train_size": 1},
        {"validation_size": 0},
        {"test_size": 0},
        {"noise_std": 0},
        {"noise_std": np.nan},
        {"alphas": ()},
        {"alphas": (0,)},
        {"alphas": (np.inf,)},
        {"alphas": (1.0, 1.0)},
    ],
)
def test_invalid_experiment_configuration(experiment, kwargs):
    with pytest.raises(ValueError):
        experiment["ExperimentConfig"](**kwargs)


def test_cli_writes_numeric_reports_and_publication_figure_formats(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "output"
    environment = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "matplotlib"))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--seeds",
            "2",
            "--seed-start",
            "10",
            "--train-size",
            "25",
            "--validation-size",
            "30",
            "--test-size",
            "35",
            "--alphas",
            "0.01",
            "1.0",
            "--output-dir",
            str(output),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert "mean ± SD" in result.stdout
    with (output / "bias_variance_runs.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 7 * 3
    with (output / "bias_variance_summary.csv").open() as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == 7 * 3
    assert all(int(row["n_seeds"]) == 2 for row in summaries)
    metadata = json.loads((output / "bias_variance_metadata.json").read_text())
    assert metadata["degrees"] == [1, 2, 3, 5, 8, 12, 20]
    assert metadata["seeds"] == [10, 11]
    assert_allclose(metadata["noise_variance"], 0.09)
    assert "matplotlib" in metadata
    for name in ("bias_variance_ols", "bias_variance_ridge", "bias_variance_fits"):
        png = output / "figures" / f"{name}.png"
        pdf = output / "figures" / f"{name}.pdf"
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert pdf.read_bytes().startswith(b"%PDF-")
        assert png.stat().st_size > 10000
        assert pdf.stat().st_size > 10000
