"""Analytical and reference checks for single-target linear regression."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from glassboxml.models import LinearRegression, RidgeRegression

MODELS = [LinearRegression, RidgeRegression]


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_known_multifeature_linear_relationship(fit_intercept):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(100, 4))
    coef = np.array([2.5, -1.0, 0.0, 4.0])
    intercept = 3.25 if fit_intercept else 0.0
    y = X @ coef + intercept
    model = LinearRegression(fit_intercept=fit_intercept)
    assert model.fit(X, y) is model
    assert_allclose(model.coef_, coef, atol=1e-12)
    assert model.intercept_ == pytest.approx(intercept, abs=1e-12)
    assert_allclose(model.predict(X), y, atol=1e-12)
    assert model.coef_.shape == (4,)
    assert model.predict(X[:1]).shape == (1,)


def test_single_feature_lists_and_integer_inputs():
    model = LinearRegression().fit([[0], [1], [2]], [1, 3, 5])
    assert_allclose(model.coef_, [2])
    assert model.intercept_ == pytest.approx(1)
    assert_allclose(model.predict([[3], [4]]), [7, 9])


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("shape", [(70, 4), (5, 8)])
def test_against_sklearn_on_full_rank_and_underdetermined_data(
    model_type, fit_intercept, shape
):
    # The reference is optional; install the experiments extra to run this test.
    reference = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(123)
    X = rng.normal(size=shape) + 2
    y = X @ rng.normal(size=shape[1]) + 7 + rng.normal(scale=0.05, size=shape[0])
    unseen = rng.normal(size=(10, shape[1]))
    kwargs = {"fit_intercept": fit_intercept}
    if model_type is RidgeRegression:
        model = model_type(alpha=2.5, **kwargs)
        expected = reference.Ridge(alpha=2.5, solver="svd", **kwargs)
    else:
        model = model_type(**kwargs)
        expected = reference.LinearRegression(**kwargs)
    model.fit(X, y)
    expected.fit(X, y)
    assert_allclose(model.coef_, expected.coef_, rtol=1e-9, atol=1e-10)
    assert model.intercept_ == pytest.approx(expected.intercept_, abs=1e-10)
    assert_allclose(model.predict(unseen), expected.predict(unseen), atol=1e-10)


def test_rank_deficient_linear_solution_has_minimum_coefficient_norm():
    x = np.arange(-3.0, 4.0)
    X = np.column_stack((x, 2 * x, np.ones_like(x)))
    model = LinearRegression().fit(X, 3 * x + 5)
    assert_allclose(model.coef_, [0.6, 1.2, 0], atol=1e-12)
    assert model.intercept_ == pytest.approx(5)
    assert_allclose(model.predict(X), 3 * x + 5, atol=1e-12)


def test_nearly_collinear_features_keep_a_stable_solution():
    rng = np.random.default_rng(32)
    x = rng.normal(size=100)
    X = np.column_stack((x, x + 1e-6 * rng.normal(size=100)))
    coef = np.array([2.0, -3.0])
    y = X @ coef + 1
    model = LinearRegression().fit(X, y)
    assert_allclose(model.coef_, coef, rtol=0, atol=1e-8)
    assert_allclose(model.predict(X), y, atol=1e-12)


def test_ridge_shrinkage_matches_orthogonal_design_formula():
    X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    coef = np.array([2.0, -3.0])
    y = X @ coef + 10
    unregularized = LinearRegression().fit(X, y)
    norms = []
    for alpha in (0, 1, 4, 100):
        model = RidgeRegression(alpha=alpha)
        assert model.fit(X, y) is model
        assert_allclose(model.coef_, 4 / (4 + alpha) * coef, atol=1e-12)
        assert model.intercept_ == pytest.approx(10)
        norms.append(np.linalg.norm(model.coef_))
    assert norms[0] == pytest.approx(np.linalg.norm(unregularized.coef_))
    assert all(a > b for a, b in zip(norms, norms[1:], strict=False))


@pytest.mark.parametrize("model_type", MODELS)
def test_constant_target_and_constant_features(model_type):
    X = np.full((5, 2), [3.0, -5.0])
    model = model_type().fit(X, np.full(5, 42.0))
    assert_allclose(model.coef_, 0)
    assert model.intercept_ == 42
    assert_allclose(model.predict(X), 42)


def test_ridge_intercept_is_not_regularized_with_shifted_features():
    X = np.arange(20.0).reshape(10, 2)
    y = X @ np.array([2.0, -0.5]) + 7
    base = RidgeRegression(alpha=1e6).fit(X, y)
    shifted = RidgeRegression(alpha=1e6).fit(X, y + 100)
    assert_allclose(base.coef_, shifted.coef_, atol=1e-12)
    assert shifted.intercept_ - base.intercept_ == pytest.approx(100)
    assert np.mean(base.predict(X)) == pytest.approx(np.mean(y))


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_zero_ridge_penalty_matches_least_squares_for_collinear_data(fit_intercept):
    X = np.column_stack((np.arange(6.0), np.arange(6.0)))
    y = np.array([2, 3, 7, 8, 12, 14])
    linear = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    ridge = RidgeRegression(alpha=0, fit_intercept=fit_intercept).fit(X, y)
    assert_array_equal(linear.coef_, ridge.coef_)
    assert linear.intercept_ == ridge.intercept_


@pytest.mark.parametrize("model_type", MODELS)
def test_readonly_noncontiguous_inputs_and_coefficient_ownership(model_type):
    X = np.arange(40, dtype=np.float32).reshape(10, 4)[:, ::2]
    y = np.arange(20, dtype=np.float32)[::2]
    original_X, original_y = X.copy(), y.copy()
    X.flags.writeable = y.flags.writeable = False
    model = model_type().fit(X, y)
    before = model.predict(X)
    model.coef_[:] = 99
    assert_array_equal(model.predict(X), before)
    assert_array_equal(X, original_X)
    assert_array_equal(y, original_y)
    assert model.coef_.dtype == np.float64
    assert before.dtype == np.float64


@pytest.mark.parametrize("model_type", MODELS)
def test_unfitted_access_and_refit(model_type):
    model = model_type()
    for attribute in ("coef_", "intercept_"):
        with pytest.raises(RuntimeError, match="fit"):
            getattr(model, attribute)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict([[1.0]])
    model.fit([[1], [2]], [1, 2])
    model.fit([[1, 2], [3, 4], [5, 6]], [3, 7, 11])
    assert model.coef_.shape == (2,)
    assert model.predict([[1, 2]]).shape == (1,)


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize(
    "X,y",
    [
        ([], []),
        ([1, 2], [1, 2]),
        ([[[1]]], [1]),
        (np.empty((0, 2)), []),
        (np.empty((2, 0)), [1, 2]),
        ([[1], [2]], [1]),
        ([[1], [2]], [[1], [2]]),
        ([[1]], 1),
        ([[np.nan]], [1]),
        ([[1]], [np.inf]),
        ([[1j]], [1]),
        ([[1]], [1j]),
        ([["1"]], [1]),
        ([[1]], ["1"]),
        ([[True]], [1]),
    ],
)
def test_invalid_fit_inputs_preserve_prior_fit(model_type, X, y):
    model = model_type().fit([[0], [1], [2]], [1, 3, 5])
    before = model.predict([[3]])
    with pytest.raises(ValueError):
        model.fit(X, y)
    assert_array_equal(model.predict([[3]]), before)


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("X", [[1, 2], [[1, 2]], [], [[np.inf]], [[1j]], [["1"]]])
def test_invalid_prediction_shapes_and_values(model_type, X):
    model = model_type().fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        model.predict(X)


@pytest.mark.parametrize("alpha", [-1, np.nan, np.inf, True, "1", [1], 1j, 10**1000])
def test_invalid_ridge_penalty(alpha):
    with pytest.raises(ValueError, match="alpha"):
        RidgeRegression(alpha=alpha)


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("fit_intercept", [0, 1, "yes", None])
def test_invalid_intercept_option(model_type, fit_intercept):
    with pytest.raises(ValueError, match="fit_intercept"):
        model_type(fit_intercept=fit_intercept)
