"""Regression metric formulas, limits, and optional reference comparisons."""

import numpy as np
import pytest

from glassboxml.metrics import (
    mean_squared_error,
    mse,
    r2_score,
    rmse,
    root_mean_squared_error,
)

METRICS = [mse, rmse, r2_score]


def test_hand_computed_metrics_and_short_names():
    y = [1, 2, 3]
    prediction = [1, 2, 5]
    assert mse is mean_squared_error
    assert rmse is root_mean_squared_error
    assert mse(y, prediction) == pytest.approx(4 / 3)
    assert rmse(y, prediction) == pytest.approx(np.sqrt(4 / 3))
    assert r2_score(y, prediction) == pytest.approx(-1)


def test_perfect_predictions_and_constant_target_convention():
    assert mse([1, 2], [1, 2]) == 0
    assert rmse([1, 2], [1, 2]) == 0
    assert r2_score([1, 2], [1, 2]) == 1
    assert r2_score([3, 3], [3, 3]) == 1
    assert r2_score([3, 3], [3, 2]) == 0
    assert r2_score([1, 2, 3], [2, 2, 2]) == pytest.approx(0)


def test_single_observation():
    assert mse([2], [4]) == 4
    assert rmse([2], [4]) == 2
    with pytest.raises(ValueError, match="two observations"):
        r2_score([2], [2])


def test_metrics_match_sklearn():
    reference = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(4)
    truth = rng.normal(size=50)
    prediction = truth + rng.normal(scale=0.3, size=50)
    assert mse(truth, prediction) == pytest.approx(
        reference.mean_squared_error(truth, prediction)
    )
    assert rmse(truth, prediction) == pytest.approx(
        np.sqrt(reference.mean_squared_error(truth, prediction))
    )
    assert r2_score(truth, prediction) == pytest.approx(
        reference.r2_score(truth, prediction)
    )


def test_large_and_small_residuals_avoid_unnecessary_squaring_overflow():
    assert rmse([1e200, -1e200], [0, 0]) == pytest.approx(1e200)
    assert rmse([1e-200, -1e-200], [0, 0]) / 1e-200 == pytest.approx(1)
    assert r2_score([1e200, -1e200], [0, 0]) == pytest.approx(0)
    assert r2_score([1e-200, -1e-200], [0, 0]) == pytest.approx(0)
    with pytest.raises(FloatingPointError):
        mse([1e200], [0])


def test_r2_scales_before_centering_and_subtracting_extreme_values():
    truth = [1e308, 1.1e308]
    assert r2_score(truth, truth) == 1.0
    assert r2_score(truth, [1.05e308, 1.05e308]) == pytest.approx(0, abs=1e-13)
    # The unscaled residuals overflow, but their variance ratio is finite.
    assert r2_score([1e308, -1e308], [-1e308, 1e308]) == pytest.approx(-3)


@pytest.mark.parametrize("metric", METRICS)
def test_inputs_are_not_modified(metric):
    truth = np.array([1.0, 3.0, 2.0, 4.0])[::2]
    prediction = np.array([0.0, 3.0])
    truth.flags.writeable = prediction.flags.writeable = False
    metric(truth, prediction)
    np.testing.assert_array_equal(truth, [1, 2])
    np.testing.assert_array_equal(prediction, [0, 3])


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "truth,prediction",
    [
        ([], []),
        (1, 1),
        ([1, 2], [1]),
        ([[1], [2]], [1, 2]),
        ([1, 2], [[1], [2]]),
        ([np.nan, 1], [0, 1]),
        ([0, 1], [np.inf, 1]),
        ([1j, 1], [0, 1]),
        ([0, 1], [1j, 1]),
        (["1", "2"], [1, 2]),
        ([1, 2], [True, False]),
    ],
)
def test_invalid_shapes_and_values(metric, truth, prediction):
    with pytest.raises(ValueError):
        metric(truth, prediction)
