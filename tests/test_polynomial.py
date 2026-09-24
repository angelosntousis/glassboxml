"""Exact feature-map checks and input validation for 1D polynomials."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from glassboxml.preprocessing import PolynomialFeatures


@pytest.mark.parametrize("include_bias", [False, True])
@pytest.mark.parametrize("column_input", [False, True])
def test_ascending_powers_with_negative_and_zero_inputs(include_bias, column_input):
    x = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    X = x[:, None] if column_input else x
    expected = np.array(
        [[-2, 4, -8], [-0.5, 0.25, -0.125], [0, 0, 0], [1, 1, 1], [3, 9, 27]]
    )
    if include_bias:
        expected = np.column_stack((np.ones(x.size), expected))
    result = PolynomialFeatures(degree=3, include_bias=include_bias).transform(X)
    assert_array_equal(result, expected)
    assert result.shape == (5, 3 + include_bias)
    assert result.dtype == np.float64


def test_default_degree_and_no_bias():
    assert_array_equal(
        PolynomialFeatures().transform([-2, 0, 3]), [[-2, 4], [0, 0], [3, 9]]
    )


def test_zero_degree_with_bias_is_constant():
    result = PolynomialFeatures(degree=0, include_bias=True).transform([-2, 0, 3])
    assert_array_equal(result, np.ones((3, 1)))


@pytest.mark.parametrize("X", [[2], [[2]]])
def test_single_observation_preserves_sample_dimension(X):
    result = PolynomialFeatures(degree=1).transform(X)
    assert_array_equal(result, [[2.0]])
    assert result.shape == (1, 1)


def test_degree_twenty_matches_direct_scalar_powers():
    x = np.array([-1.0, -0.8, 0.0, 0.25, 1.0])
    expected = np.array(
        [[float(value) ** degree for degree in range(21)] for value in x]
    )
    result = PolynomialFeatures(degree=20, include_bias=True).transform(x)
    assert_allclose(result, expected, rtol=2e-15, atol=0)
    assert result.shape == (5, 21)


def test_numpy_scalar_configuration():
    transformer = PolynomialFeatures(degree=np.int64(2), include_bias=np.bool_(True))
    assert_array_equal(transformer.transform([2]), [[1.0, 2.0, 4.0]])


def test_fit_returns_self_and_fit_transform_agrees_with_stateless_transform():
    transformer = PolynomialFeatures(degree=4)
    first = transformer.transform([-1, 0, 2])
    assert transformer.fit([10, 20]) is transformer
    assert_array_equal(transformer.transform([-1, 0, 2]), first)
    assert_array_equal(transformer.fit_transform([-1, 0, 2]), first)
    # Fitting does not learn an input range or alter the polynomial definition.
    assert_array_equal(transformer.transform([3]), [[3, 9, 27, 81]])


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("column_input", [False, True])
def test_readonly_noncontiguous_inputs_are_preserved(dtype, column_input):
    x = np.arange(-6, 6, dtype=dtype)[::2]
    X = x[:, None] if column_input else x
    original = X.copy()
    X.flags.writeable = False
    transformer = PolynomialFeatures(degree=3)
    assert transformer.fit(X) is transformer
    result = transformer.fit_transform(X)
    assert_allclose(result, np.column_stack((x, x**2, x**3)))
    assert result.dtype == np.float64
    assert not np.shares_memory(result, X)
    assert_array_equal(X, original)
    assert not X.flags.writeable


@pytest.mark.parametrize("include_bias", [False, True])
def test_each_transformation_owns_its_output(include_bias):
    x = np.array([1.0, 2.0, 3.0])
    transformer = PolynomialFeatures(degree=1, include_bias=include_bias)
    first = transformer.transform(x)
    expected = first.copy()
    second = transformer.transform(x)
    assert not np.shares_memory(first, x)
    assert not np.shares_memory(second, first)
    first[:] = -100
    assert_array_equal(second, expected)
    assert_array_equal(x, [1, 2, 3])
    x[:] = 9
    assert_array_equal(second, expected)


@pytest.mark.parametrize("degree", [-1, 1.5, 2.0, True, np.bool_(False), "2", None, 1j])
def test_invalid_degree(degree):
    with pytest.raises(ValueError, match="degree"):
        PolynomialFeatures(degree=degree)


def test_zero_degree_without_bias_has_no_features():
    with pytest.raises(ValueError):
        PolynomialFeatures(degree=0, include_bias=False)


@pytest.mark.parametrize("include_bias", [0, 1, "yes", None, [True]])
def test_invalid_bias_option(include_bias):
    with pytest.raises(ValueError, match="include_bias"):
        PolynomialFeatures(include_bias=include_bias)


@pytest.mark.parametrize(
    "X",
    [
        1.0,
        np.array(1.0),
        [],
        np.empty((0, 1)),
        np.empty((2, 0)),
        [[1.0, 2.0]],
        [[1.0, 2.0], [3.0, 4.0]],
        [[[1.0]]],
        [[1.0], [2.0, 3.0]],
        [np.nan],
        [np.inf],
        [-np.inf],
        [1j],
        [1.0 + 0j],
        [True, False],
        ["1", "2"],
        np.array([1.0, 2.0], dtype=object),
    ],
)
@pytest.mark.parametrize("method", ["fit", "transform", "fit_transform"])
def test_invalid_input_shapes_and_values(X, method):
    transformer = PolynomialFeatures(degree=3)
    with pytest.raises(ValueError):
        getattr(transformer, method)(X)


@pytest.mark.parametrize("method", ["transform", "fit_transform"])
def test_overflow_raises_and_preserves_input(method):
    x = np.array([1e200, -1e200])
    original = x.copy()
    transformer = PolynomialFeatures(degree=2)
    with pytest.raises(FloatingPointError):
        getattr(transformer, method)(x)
    assert_array_equal(x, original)


def test_large_finite_linear_feature_does_not_compute_an_unneeded_power():
    x = np.array([-1e308, 1e308])
    assert_array_equal(PolynomialFeatures(degree=1).transform(x), x[:, None])


def test_feature_map_recovers_known_cubic_with_library_regression():
    from glassboxml.models import LinearRegression

    x = np.linspace(-1, 1, 25)
    y = 1.5 - 2 * x + 0.25 * x**2 + 3 * x**3
    transformer = PolynomialFeatures(degree=3)
    model = LinearRegression().fit(transformer.fit_transform(x), y)
    assert_allclose(model.coef_, [-2, 0.25, 3], atol=1e-12)
    assert model.intercept_ == pytest.approx(1.5, abs=1e-12)
    unseen = np.array([-0.9, -0.1, 0.4])
    expected = 1.5 - 2 * unseen + 0.25 * unseen**2 + 3 * unseen**3
    assert_allclose(model.predict(transformer.transform(unseen)), expected, atol=1e-12)
