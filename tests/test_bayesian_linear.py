"""Analytical posterior and uncertainty checks for Bayesian linear regression."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from glassboxml.models import BayesianLinearRegression, RidgeRegression


def test_manual_two_feature_posterior_and_predictive_distribution():
    X = np.array([[1.0, 0.0], [1.0, 1.0]])
    y = np.array([1.0, 2.0])
    model = BayesianLinearRegression(alpha=1, beta=2)
    assert model.fit(X, y) is model
    expected_covariance = np.array([[3.0, -2.0], [-2.0, 5.0]]) / 11
    expected_mean = np.array([10.0, 8.0]) / 11
    assert_allclose(model.posterior_covariance_, expected_covariance, atol=1e-14)
    assert_allclose(model.posterior_mean_, expected_mean, atol=1e-14)
    assert_array_equal(model.coef_, model.posterior_mean_)

    mean, std = model.predict([[1, 2]], return_std=True)
    assert mean.shape == std.shape == (1,)
    assert mean[0] == pytest.approx(26 / 11)
    assert std[0] ** 2 == pytest.approx(15 / 11 + 1 / 2)
    assert_array_equal(model.predict([[1, 2]]), mean)


def test_single_feature_posterior_matches_scalar_formula():
    X = np.array([[-2.0], [0.0], [1.0], [3.0]])
    y = np.array([-3.0, 0.5, 2.0, 6.0])
    alpha, beta = 2.5, 4.0
    variance = 1 / (alpha + beta * 14)
    mean = beta * variance * 26
    model = BayesianLinearRegression(alpha=alpha, beta=beta).fit(X, y)
    assert_allclose(model.posterior_covariance_, [[variance]], atol=1e-15)
    assert_allclose(model.posterior_mean_, [mean], atol=1e-14)
    predicted, std = model.predict([[0], [2]], return_std=True)
    assert_allclose(predicted, [0, 2 * mean])
    assert_allclose(std**2, [1 / beta, 1 / beta + 4 * variance])


def test_design_matrix_is_used_without_an_implicit_intercept():
    model = BayesianLinearRegression().fit([[0], [0], [0]], [10, 10, 10])
    assert_array_equal(model.posterior_mean_, [0])
    assert_allclose(model.posterior_covariance_, [[1]])
    assert_array_equal(model.predict([[0], [1]]), [0, 0])


@pytest.mark.parametrize("shape", [(40, 4), (3, 8)])
def test_posterior_mean_matches_ridge_with_alpha_over_beta(shape):
    rng = np.random.default_rng(38)
    X = rng.normal(size=shape)
    y = rng.normal(size=shape[0])
    test_X = rng.normal(size=(15, shape[1]))
    alpha, beta = 3.5, 2.75
    bayesian = BayesianLinearRegression(alpha=alpha, beta=beta).fit(X, y)
    ridge = RidgeRegression(alpha=alpha / beta, fit_intercept=False).fit(X, y)
    assert_allclose(bayesian.coef_, ridge.coef_, atol=1e-13)
    assert_allclose(bayesian.predict(test_X), ridge.predict(test_X), atol=1e-13)


@pytest.mark.parametrize("shape", [(20, 5), (2, 7)])
def test_covariance_is_symmetric_positive_semidefinite_with_collinearity(shape):
    rng = np.random.default_rng(27)
    X = rng.normal(size=shape)
    X[:, -1] = 2 * X[:, 0]
    y = rng.normal(size=shape[0])
    alpha, beta = 0.25, 3.0
    model = BayesianLinearRegression(alpha=alpha, beta=beta).fit(X, y)
    covariance = model.posterior_covariance_
    precision = alpha * np.eye(shape[1]) + beta * X.T @ X
    assert_allclose(covariance, covariance.T, atol=1e-14)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-13
    assert_allclose(precision @ covariance, np.eye(shape[1]), atol=1e-12)
    assert_allclose(precision @ model.posterior_mean_, beta * X.T @ y, atol=1e-12)


def test_orthogonal_features_with_different_scales_remain_accurate():
    X = np.diag([1e-6, 1, 1e6])
    y = np.array([2.0, -3.0, 4.0])
    model = BayesianLinearRegression(alpha=2, beta=3).fit(X, y)
    variance = 1 / (2 + 3 * np.diag(X) ** 2)
    assert_allclose(model.posterior_covariance_, np.diag(variance), rtol=1e-14)
    assert_allclose(
        model.posterior_mean_, 3 * variance * np.diag(X) * y, rtol=1e-14, atol=1e-14
    )


def test_more_observations_reduce_parameter_and_total_predictive_uncertainty():
    rng = np.random.default_rng(19)
    x = rng.uniform(-1, 1, size=100)
    X = np.column_stack((np.ones_like(x), x))
    y = X @ np.array([0.5, 2.0]) + rng.normal(scale=0.2, size=x.size)
    test_x = np.linspace(-2, 2, 51)
    test_X = np.column_stack((np.ones_like(test_x), test_x))
    previous_covariance = None
    previous_std = None
    beta = 25.0
    for n_samples in (5, 20, 100):
        model = BayesianLinearRegression(alpha=1, beta=beta).fit(
            X[:n_samples], y[:n_samples]
        )
        covariance = model.posterior_covariance_
        _, std = model.predict(test_X, return_std=True)
        assert np.all(std >= np.sqrt(1 / beta))
        if previous_covariance is not None:
            # Adding observations reduces covariance in the PSD ordering.
            assert np.linalg.eigvalsh(previous_covariance - covariance).min() >= -1e-13
            assert np.all(std < previous_std)
        previous_covariance, previous_std = covariance, std


def test_uncertainty_grows_away_from_observed_region():
    x = np.linspace(-0.5, 0.5, 20)
    X = np.column_stack((np.ones_like(x), x))
    model = BayesianLinearRegression(alpha=1, beta=25).fit(X, 1 + 2 * x)
    _, std = model.predict([[1, 0], [1, -3], [1, 3]], return_std=True)
    assert std[1] > std[0]
    assert std[2] == pytest.approx(std[1])


def test_zero_feature_prediction_has_observation_noise_only():
    model = BayesianLinearRegression(alpha=2, beta=16).fit([[1, 2]], [3])
    mean, std = model.predict(np.zeros((3, 2)), return_std=True)
    assert_array_equal(mean, np.zeros(3))
    assert_allclose(std, 0.25, atol=1e-15)


def test_predictive_standard_deviation_avoids_overflow_from_squared_features():
    model = BayesianLinearRegression().fit([[1]], [0])
    mean, std = model.predict([[1e200]], return_std=True)
    assert_array_equal(mean, [0])
    assert std[0] == pytest.approx(1e200 / np.sqrt(2))
    assert np.isfinite(std[0])


def test_higher_noise_precision_reduces_total_predictive_uncertainty():
    X = np.array([[1, -1], [1, 0], [1, 1]])
    y = np.array([-1, 1, 3])
    test_X = np.array([[1, -2], [1, 0], [1, 2]])
    noisy = BayesianLinearRegression(alpha=1, beta=1).fit(X, y)
    precise = BayesianLinearRegression(alpha=1, beta=25).fit(X, y)
    _, noisy_std = noisy.predict(test_X, return_std=True)
    _, precise_std = precise.predict(test_X, return_std=True)
    assert np.all(precise_std < noisy_std)
    covariance_difference = noisy.posterior_covariance_ - precise.posterior_covariance_
    assert np.linalg.eigvalsh(covariance_difference).min() > 0


def test_stronger_prior_contracts_posterior_covariance_and_mean():
    X = np.array([[1, -1], [1, 0], [1, 1]])
    y = np.array([-1, 1, 3])
    weak = BayesianLinearRegression(alpha=0.1, beta=4).fit(X, y)
    strong = BayesianLinearRegression(alpha=100, beta=4).fit(X, y)
    covariance_difference = weak.posterior_covariance_ - strong.posterior_covariance_
    assert np.linalg.eigvalsh(covariance_difference).min() > 0
    assert np.linalg.norm(strong.coef_) < np.linalg.norm(weak.coef_)


def test_posterior_covariance_is_independent_of_target_values():
    X = [[1, -1], [1, 0], [1, 1]]
    first = BayesianLinearRegression(alpha=2, beta=3).fit(X, [1, 2, 3])
    second = BayesianLinearRegression(alpha=2, beta=3).fit(X, [-100, 12, 90])
    assert_array_equal(first.posterior_covariance_, second.posterior_covariance_)
    assert not np.allclose(first.posterior_mean_, second.posterior_mean_)


def test_readonly_noncontiguous_inputs_and_defensive_posterior_copies():
    X = np.arange(40, dtype=np.float32).reshape(10, 4)[:, ::2]
    y = np.arange(20, dtype=np.float32)[::2]
    original_X, original_y = X.copy(), y.copy()
    X.flags.writeable = y.flags.writeable = False
    model = BayesianLinearRegression().fit(X, y)
    expected_mean, expected_std = model.predict(X, return_std=True)
    for name in ("coef_", "posterior_mean_", "posterior_covariance_"):
        value = getattr(model, name)
        assert value.dtype == np.float64
        value[...] = 999
    mean, std = model.predict(X, return_std=True)
    assert_array_equal(mean, expected_mean)
    assert_array_equal(std, expected_std)
    assert_array_equal(X, original_X)
    assert_array_equal(y, original_y)
    assert mean.dtype == std.dtype == np.float64


def test_fitted_state_does_not_alias_training_data_or_prediction_outputs():
    X = np.array([[1, -1], [1, 0], [1, 1]], dtype=float)
    y = np.array([-1, 1, 3], dtype=float)
    model = BayesianLinearRegression().fit(X, y)
    expected_mean, expected_std = model.predict([[1, 2]], return_std=True)
    X[:] = y[:] = 0
    mean, std = model.predict([[1, 2]], return_std=True)
    assert_array_equal(mean, expected_mean)
    assert_array_equal(std, expected_std)
    mean[:] = std[:] = 999
    mean, std = model.predict([[1, 2]], return_std=True)
    assert_array_equal(mean, expected_mean)
    assert_array_equal(std, expected_std)


def test_unfitted_access_and_refit_with_a_different_number_of_features():
    model = BayesianLinearRegression()
    for name in ("coef_", "posterior_mean_", "posterior_covariance_"):
        with pytest.raises(RuntimeError, match="fit"):
            getattr(model, name)
    for return_std in (False, True):
        with pytest.raises(RuntimeError, match="fit"):
            model.predict([[1]], return_std=return_std)
    model.fit([[1], [2]], [1, 2])
    model.fit([[1, 2], [3, 4]], [3, 7])
    assert model.posterior_mean_.shape == (2,)
    assert model.posterior_covariance_.shape == (2, 2)
    assert model.predict([[1, 2]]).shape == (1,)
    with pytest.raises(ValueError):
        model.predict([[1]])


def test_fitting_and_predictions_are_deterministic():
    X = np.array([[1, -1], [1, 0], [1, 1]])
    y = np.array([1, 2, 3])
    first = BayesianLinearRegression(alpha=3, beta=4).fit(X, y)
    second = BayesianLinearRegression(alpha=3, beta=4).fit(X, y)
    assert_array_equal(first.posterior_mean_, second.posterior_mean_)
    assert_array_equal(first.posterior_covariance_, second.posterior_covariance_)
    for first_result, second_result in zip(
        first.predict(X, return_std=True),
        second.predict(X, return_std=True),
        strict=True,
    ):
        assert_array_equal(first_result, second_result)


def test_changing_hyperparameters_only_affects_predictions_after_refit():
    X = [[1, -1], [1, 0], [1, 1]]
    y = [1, 2, 3]
    model = BayesianLinearRegression(alpha=1, beta=2).fit(X, y)
    mean, std = model.predict(X, return_std=True)
    covariance = model.posterior_covariance_
    model.alpha, model.beta = 100, 50
    new_mean, new_std = model.predict(X, return_std=True)
    assert_array_equal(new_mean, mean)
    assert_array_equal(new_std, std)
    assert_array_equal(model.posterior_covariance_, covariance)
    model.fit(X, y)
    assert not np.allclose(model.posterior_covariance_, covariance)
    _, refitted_std = model.predict(X, return_std=True)
    assert np.all(refitted_std < std)


@pytest.mark.parametrize("name", ["alpha", "beta"])
@pytest.mark.parametrize(
    "value", [-1, 0, np.nan, np.inf, -np.inf, True, "1", [1], 1j, None, 10**1000]
)
def test_invalid_precision_values(name, value):
    with pytest.raises(ValueError, match=name):
        BayesianLinearRegression(**{name: value})


def test_numpy_scalar_hyperparameters_and_boolean_option_are_accepted():
    model = BayesianLinearRegression(alpha=np.float32(2), beta=np.int64(3))
    model.fit([[1]], [2])
    mean, std = model.predict([[1]], return_std=np.bool_(True))
    assert_allclose(mean, [6 / 5])
    assert_allclose(std**2, [1 / 3 + 1 / 5])


@pytest.mark.parametrize("return_std", [0, 1, "yes", None, [True], np.array(True)])
def test_invalid_return_std_option(return_std):
    model = BayesianLinearRegression().fit([[1]], [2])
    with pytest.raises(ValueError, match="return_std"):
        model.predict([[1]], return_std=return_std)


@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_invalid_mutated_hyperparameters_preserve_previous_fit(name):
    X, y = [[1], [2]], [1, 2]
    model = BayesianLinearRegression().fit(X, y)
    mean, std = model.predict(X, return_std=True)
    covariance = model.posterior_covariance_
    setattr(model, name, -1)
    with pytest.raises(ValueError, match=name):
        model.fit(X, y)
    new_mean, new_std = model.predict(X, return_std=True)
    assert_array_equal(new_mean, mean)
    assert_array_equal(new_std, std)
    assert_array_equal(model.posterior_covariance_, covariance)


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
        ([[np.inf]], [1]),
        ([[1]], [np.nan]),
        ([[1]], [np.inf]),
        ([[1j]], [1]),
        ([[1]], [1j]),
        ([["1"]], [1]),
        ([[1]], ["1"]),
        ([[True]], [1]),
        ([[1]], [True]),
        ([[None]], [1]),
        ([[1]], [None]),
    ],
)
def test_invalid_fit_inputs_preserve_previous_fit(X, y):
    model = BayesianLinearRegression().fit([[1], [2]], [1, 2])
    mean, std = model.predict([[3]], return_std=True)
    covariance = model.posterior_covariance_
    with pytest.raises(ValueError):
        model.fit(X, y)
    new_mean, new_std = model.predict([[3]], return_std=True)
    assert_array_equal(new_mean, mean)
    assert_array_equal(new_std, std)
    assert_array_equal(model.posterior_covariance_, covariance)


@pytest.mark.parametrize("return_std", [False, True])
@pytest.mark.parametrize(
    "X",
    [
        [1, 2],
        [[1, 2]],
        [],
        np.empty((0, 1)),
        np.empty((1, 0)),
        [[[1]]],
        [[np.nan]],
        [[np.inf]],
        [[1j]],
        [["1"]],
        [[True]],
    ],
)
def test_invalid_prediction_shapes_and_values(X, return_std):
    model = BayesianLinearRegression().fit([[1], [2]], [1, 2])
    with pytest.raises(ValueError):
        model.predict(X, return_std=return_std)


def test_tiny_noise_precision_can_have_a_representable_standard_deviation():
    beta = 1e-320
    model = BayesianLinearRegression(alpha=1, beta=beta).fit([[0]], [0])
    mean, std = model.predict([[0]], return_std=True)
    assert_array_equal(mean, [0])
    assert np.all(np.isfinite(std))
    assert_allclose(std, [1 / np.sqrt(beta)], rtol=1e-15)


def test_numerical_fit_failure_preserves_posterior_and_noise():
    model = BayesianLinearRegression(alpha=1, beta=2).fit([[1]], [2])
    before_mean, before_std = model.predict([[1]], return_std=True)
    covariance = model.posterior_covariance_
    model.beta = 1e308
    with pytest.raises(FloatingPointError):
        model.fit([[1e308]], [1])
    after_mean, after_std = model.predict([[1]], return_std=True)
    assert_array_equal(before_mean, after_mean)
    assert_array_equal(before_std, after_std)
    assert_array_equal(covariance, model.posterior_covariance_)
