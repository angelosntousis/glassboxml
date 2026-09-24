"""Analytical examples and edge cases for top-label probability calibration."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from glassboxml.metrics import (
    confidence,
    ece,
    expected_calibration_error,
    reliability_diagram_data,
)


def test_confidence_is_maximum_not_true_class_probability():
    probabilities = np.array([[0.1, 0.6, 0.3], [0.8, 0.1, 0.1]])
    before = probabilities.copy()
    result = confidence(probabilities)
    assert_allclose(result, [0.6, 0.8])
    result[:] = 0
    assert_array_equal(probabilities, before)


def test_hand_computed_multiclass_bin_statistics_and_weighted_ece():
    probabilities = [
        [0.6, 0.3, 0.1],
        [0.1, 0.6, 0.3],
        [0.1, 0.2, 0.7],
        [0.1, 0.05, 0.85],
    ]
    # Three in [0.5, 0.75): accuracy 2/3, confidence 1.9/3.
    # One in [0.75, 1]: accuracy 0, confidence 0.85.
    truth = [0, 1, 0, 0]
    data = reliability_diagram_data(truth, probabilities, n_bins=4)
    assert_allclose(data.bin_edges, [0, 0.25, 0.5, 0.75, 1])
    assert_array_equal(data.counts, [0, 0, 3, 1])
    assert_allclose(data.accuracy[2:], [2 / 3, 0])
    assert_allclose(data.mean_confidence[2:], [1.9 / 3, 0.85])
    assert np.isnan(data.accuracy[:2]).all()
    assert np.isnan(data.mean_confidence[:2]).all()
    expected = 0.75 * abs(2 / 3 - 1.9 / 3) + 0.25 * 0.85
    assert data.ece == pytest.approx(expected)
    assert expected_calibration_error(truth, probabilities, n_bins=4) == pytest.approx(
        expected
    )
    assert ece is expected_calibration_error


def test_exact_bin_edges_neighbors_and_unit_confidence():
    scores = np.array([0.5, np.nextafter(0.75, 0), 0.75, np.nextafter(0.75, 1), 1.0])
    probabilities = np.column_stack((scores, 1 - scores))
    data = reliability_diagram_data(np.zeros(5), probabilities, n_bins=4)
    assert_array_equal(data.counts, [0, 0, 2, 3])
    assert data.counts.sum() == 5
    assert data.ece == pytest.approx(np.mean(1 - scores))


def test_uniform_probabilities_ties_choose_first_column():
    data = reliability_diagram_data([0, 1, 2, 3], np.full((4, 4), 0.25), n_bins=4)
    assert_array_equal(data.counts, [0, 4, 0, 0])
    assert data.accuracy[1] == 0.25
    assert data.ece == 0


def test_decimal_edges_and_immediate_neighbors_use_the_declared_side():
    scores = np.array([0.5, np.nextafter(0.6, 0), 0.6, np.nextafter(0.6, 1), 1.0])
    probabilities = np.column_stack((scores, 1 - scores))
    data = reliability_diagram_data(np.zeros(5), probabilities, n_bins=10)
    assert data.bin_edges[6] == 0.6
    assert_array_equal(data.counts, [0, 0, 0, 0, 0, 2, 2, 0, 0, 1])


@pytest.mark.parametrize("n_bins", [1, 10, 100, np.int64(5)])
def test_perfect_and_maximally_overconfident_predictions(n_bins):
    probabilities = np.eye(3)
    assert ece([0, 1, 2], probabilities, n_bins=n_bins) == 0
    assert ece([1, 2, 0], probabilities, n_bins=n_bins) == 1
    single = reliability_diagram_data([0], [[0.6, 0.4]], n_bins=n_bins)
    assert single.counts.sum() == 1
    assert single.ece == pytest.approx(0.4)


@pytest.mark.parametrize(
    "classes,truth",
    [
        (["zebra", "ant", "cat"], ["zebra", "cat", "ant"]),
        ([9, -5, 4], [9, 4, -5]),
    ],
)
def test_original_labels_follow_probability_column_order(classes, truth):
    probabilities = [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]
    assert ece(truth, probabilities, classes=classes) == pytest.approx(1 - 1.9 / 3)
    assert ece([0, 2, 1], probabilities) == pytest.approx(
        ece(truth, probabilities, classes=classes)
    )


def test_float32_probabilities_and_readonly_inputs():
    probabilities = np.array([[0.2, 0.2, 0.6], [0.15, 0.15, 0.7]], dtype=np.float32)
    truth = np.array([2, 0])
    probabilities.setflags(write=False)
    truth.setflags(write=False)
    data = reliability_diagram_data(truth, probabilities)
    assert data.counts.sum() == 2
    assert 0 <= data.ece <= 1


@pytest.mark.parametrize(
    "probabilities",
    [
        [],
        [0.2, 0.8],
        np.empty((0, 2)),
        [[1]],
        np.zeros((2, 2, 2)),
        [[0.2, 0.2]],
        [[-0.1, 1.1]],
        [[np.nan, 1]],
        [[np.inf, 0]],
        [[0.3 + 0j, 0.7 + 0j]],
        [["0.3", "0.7"]],
        [[True, False]],
        np.array([[0.4, 0.6]], dtype=object),
    ],
)
def test_invalid_probabilities_raise_for_all_metrics(probabilities):
    for metric in (
        confidence,
        lambda p: ece([0], p),
        lambda p: reliability_diagram_data([0], p),
    ):
        with pytest.raises(ValueError):
            metric(probabilities)


@pytest.mark.parametrize(
    "truth", [[], [0, 1], [[0]], [-1], [2], [0.5], [np.nan], [np.inf], ["a"], [0j]]
)
def test_invalid_default_labels(truth):
    with pytest.raises(ValueError):
        reliability_diagram_data(truth, [[0.4, 0.6]])


@pytest.mark.parametrize("n_bins", [0, -1, 2.5, True, np.bool_(False), "10"])
def test_invalid_bin_count(n_bins):
    with pytest.raises(ValueError, match="n_bins"):
        ece([1], [[0.4, 0.6]], n_bins=n_bins)


@pytest.mark.parametrize(
    "truth,classes",
    [
        ([0], [0]),
        ([0], [0, 0]),
        ([0], [[0, 1]]),
        ([0], [1, 2]),
        (["a"], [0, 1]),
        ([0], ["a", "b"]),
        ([0], [0, np.nan]),
        ([0], np.array([0, 1], dtype=object)),
    ],
)
def test_invalid_explicit_classes(truth, classes):
    with pytest.raises(ValueError):
        ece(truth, [[0.4, 0.6]], classes=classes)


def test_bin_count_changes_estimate_and_single_bin_can_hide_errors():
    probabilities = [[0.6, 0.4], [0.4, 0.6], [0.9, 0.1], [0.9, 0.1]]
    truth = [0, 1, 1, 0]  # accuracy 0.75 equals overall mean confidence.
    assert ece(truth, probabilities, n_bins=1) == pytest.approx(0)
    assert ece(truth, probabilities, n_bins=5) == pytest.approx(0.4)
