"""Poisson MLE checks against analytical results and finite differences."""

import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from glassboxml.diagnostics.gradient_check import numerical_gradient, relative_error
from glassboxml.models.poisson import PoissonMLE, _objective_and_gradient


def test_closed_form_is_sample_mean_and_fit_returns_self():
    x = np.array([0, 1, 3, 7, 2])
    model = PoissonMLE()
    assert model.fit(x) is model
    assert model.rate_ == np.mean(x)


@pytest.mark.parametrize("rate", [0.05, 1.0, 4.2, 1000.0])
def test_gradient_agrees_with_closed_form_on_seeded_samples(rate):
    x = np.random.default_rng(42).poisson(rate, size=1000)
    exact = PoissonMLE().fit(x)
    numerical = PoissonMLE(method="gradient").fit(x)
    assert numerical.rate_ == pytest.approx(exact.rate_, rel=1e-10)
    assert numerical.log_likelihood(x) == pytest.approx(exact.log_likelihood(x))


@pytest.mark.parametrize("method", ["closed_form", "gradient"])
@pytest.mark.parametrize("x", [[1], [0, 0, 1], [1.0, 2.0, 5.0], [1000000]])
def test_successful_fits_have_positive_finite_rate(method, x):
    rate = PoissonMLE(method=method).fit(x).rate_
    assert np.isfinite(rate) and rate > 0
    assert rate == pytest.approx(np.mean(x), rel=1e-10)


def test_sparse_counts_and_repeatability():
    x = np.zeros(10000)
    x[-1] = 1
    first = PoissonMLE(method="gradient").fit(x).rate_
    second = PoissonMLE(method="gradient").fit(x).rate_
    assert first == second
    assert first == pytest.approx(1e-4, rel=1e-10)


@pytest.mark.parametrize("theta", [-2.0, 0.3, 2.0])
def test_log_rate_gradient_with_existing_checker(theta):
    mean = float(np.mean([0, 1, 4, 7]))
    numerical = numerical_gradient(
        lambda p: _objective_and_gradient(float(p[0]), mean)[0], [theta]
    )
    _, analytical = _objective_and_gradient(theta, mean)
    assert relative_error(numerical, [analytical]) < 1e-9


@pytest.mark.parametrize("method", ["closed_form", "gradient"])
def test_log_likelihood_includes_normalizing_constants(method):
    model = PoissonMLE(method=method).fit([1, 2, 3])
    x = [0, 1, 4]
    expected = sum(k * math.log(2) - 2 - math.log(math.factorial(k)) for k in x)
    assert model.log_likelihood(x) == pytest.approx(expected)
    assert model.log_likelihood([0, 0]) == pytest.approx(-4)
    assert model.log_likelihood(x + x) == pytest.approx(2 * expected)


def test_log_likelihood_handles_counts_whose_factorial_overflows():
    result = PoissonMLE().fit([1000]).log_likelihood([1000])
    assert np.isfinite(result)
    assert result == pytest.approx(1000 * math.log(1000) - 1000 - math.lgamma(1001))


@pytest.mark.parametrize("counts", [[1e308], [0, 0]])
def test_nonfinite_log_likelihood_raises_and_preserves_fit(counts):
    model = PoissonMLE().fit([1e308])
    with np.errstate(all="raise"):
        with pytest.raises(ValueError, match="log likelihood.*float64"):
            model.log_likelihood(counts)
    assert model.rate_ == 1e308


@pytest.mark.parametrize("method", ["closed_form", "gradient"])
def test_readonly_input_is_preserved_and_refit_updates_rate(method):
    x = np.array([0, 1, 3, 8])[::2]
    original = x.copy()
    x.flags.writeable = False
    model = PoissonMLE(method=method).fit(x)
    model.log_likelihood(x)
    assert_array_equal(x, original)
    model.fit([5, 5])
    assert model.rate_ == pytest.approx(5)


@pytest.mark.parametrize("method", ["closed_form", "gradient"])
@pytest.mark.parametrize(
    "bad", [[], 1, [[1, 2]], [-1, 2], [0.5], [np.nan], [np.inf], [1j], ["1"], [True]]
)
def test_invalid_counts_rejected_by_fit_and_likelihood(method, bad):
    model = PoissonMLE(method=method).fit([1, 2])
    with pytest.raises(ValueError):
        model.fit(bad)
    with pytest.raises(ValueError):
        model.log_likelihood(bad)
    assert model.rate_ == pytest.approx(1.5)


@pytest.mark.parametrize("method", ["closed_form", "gradient"])
def test_all_zero_training_samples_have_no_positive_mle(method):
    model = PoissonMLE(method=method)
    with pytest.raises(ValueError, match="all-zero"):
        model.fit([0, 0, 0])
    with pytest.raises(RuntimeError, match="fit"):
        _ = model.rate_


def test_unfitted_access():
    model = PoissonMLE()
    with pytest.raises(RuntimeError, match="fit"):
        _ = model.rate_
    with pytest.raises(RuntimeError, match="fit"):
        model.log_likelihood([1])


def test_nonconvergence_is_explicit_and_preserves_previous_fit():
    model = PoissonMLE(method="gradient", max_iter=1).fit([1])
    with pytest.raises(RuntimeError, match="converge"):
        model.fit([100])
    assert model.rate_ == 1


def test_invalid_method():
    with pytest.raises(ValueError, match="method"):
        PoissonMLE(method="unknown")


@pytest.mark.parametrize("tol", [0, -1, np.nan, np.inf, True, "small", [0.1], 10**400])
def test_invalid_tolerance(tol):
    with pytest.raises(ValueError, match="tol"):
        PoissonMLE(tol=tol)


def test_large_representable_integer_tolerance_is_accepted():
    assert PoissonMLE(tol=10**100).tol == 1e100


@pytest.mark.parametrize("max_iter", [0, -1, 1.5, True, "10"])
def test_invalid_iteration_limit(max_iter):
    with pytest.raises(ValueError, match="max_iter"):
        PoissonMLE(max_iter=max_iter)
