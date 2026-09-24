"""Deterministic checks against independently derived analytical gradients."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from glassboxml.diagnostics.gradient_check import numerical_gradient, relative_error


@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (2, 1, 3)])
def test_linear_gradient_preserves_shape(shape):
    parameters = np.arange(np.prod(shape), dtype=float).reshape(shape) / 3
    coefficients = np.linspace(-2, 3, parameters.size).reshape(shape)
    gradient = numerical_gradient(lambda x: np.sum(coefficients * x) + 7, parameters)
    assert gradient.shape == shape
    assert gradient.dtype == np.float64
    assert_allclose(gradient, coefficients, rtol=1e-8, atol=1e-9)


def test_quadratic_with_cross_terms():
    matrix = np.array([[4.0, 1.0, -2.0], [1.0, 3.0, 0.5], [-2.0, 0.5, 5.0]])
    linear = np.array([0.5, -1.0, 2.0])
    parameters = np.array([1.2, -0.7, 0.3])
    gradient = numerical_gradient(
        lambda x: 0.5 * x @ matrix @ x + linear @ x, parameters
    )
    expected = matrix @ parameters + linear
    assert_allclose(gradient, expected, rtol=1e-9, atol=1e-9)
    assert relative_error(gradient, expected) < 1e-9
    assert relative_error(gradient, expected + 1) > 0.01


def test_multivariate_transcendental_function():
    x = np.array([0.4, -0.3, 1.5])
    gradient = numerical_gradient(
        lambda p: np.sin(p[0] * p[1]) + np.exp(p[1]) + np.log(p[2]) * p[0], x
    )
    expected = np.array(
        [
            x[1] * np.cos(x[0] * x[1]) + np.log(x[2]),
            x[0] * np.cos(x[0] * x[1]) + np.exp(x[1]),
            x[0] / x[2],
        ]
    )
    assert_allclose(gradient, expected, rtol=1e-9, atol=1e-9)


def test_centered_difference_has_second_order_convergence():
    x = np.array([0.7])
    exact = np.cos(x)
    coarse = abs(numerical_gradient(lambda p: np.sin(p[0]), x, eps=0.02) - exact)
    fine = abs(numerical_gradient(lambda p: np.sin(p[0]), x, eps=0.01) - exact)
    assert_allclose(coarse / fine, 4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_noncontiguous_readonly_input_is_not_modified(dtype):
    values = np.arange(12, dtype=dtype).reshape(3, 4).T[::2]
    original = values.copy()
    values.flags.writeable = False
    gradient = numerical_gradient(lambda x: np.sum(x**2), values)
    assert_allclose(gradient, 2 * values, rtol=1e-8, atol=1e-8)
    assert_array_equal(values, original)


def test_objective_receives_independent_copies_and_exact_evaluation_count():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    calls = []

    def objective(x):
        calls.append(x)
        result = np.sum(x**2)
        x[:] = 0
        return result

    assert_allclose(numerical_gradient(objective, values), 2 * values, rtol=1e-9)
    assert len(calls) == 2 * values.size
    assert len({id(x) for x in calls}) == len(calls)
    assert_array_equal(values, [[1, 2], [3, 4]])


def test_objective_exception_propagates_without_mutating_input():
    values = np.array([1.0])

    def objective(x):
        x[:] = 9
        raise RuntimeError("objective failed")

    with pytest.raises(RuntimeError, match="objective failed"):
        numerical_gradient(objective, values)
    assert_array_equal(values, [1])


def test_empty_input_requires_no_evaluations():
    def objective(x):
        pytest.fail("empty inputs must not evaluate the objective")

    assert numerical_gradient(objective, np.empty((2, 0))).shape == (2, 0)


@pytest.mark.parametrize("eps", [0, -1, np.nan, np.inf, [1e-5], 1j, True])
def test_invalid_epsilon(eps):
    with pytest.raises(ValueError, match="eps"):
        numerical_gradient(lambda x: np.sum(x), np.ones(2), eps=eps)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [1j], ["1"], [True]])
def test_invalid_parameters(values):
    with pytest.raises(ValueError, match="parameters"):
        numerical_gradient(lambda x: np.sum(x), values)


@pytest.mark.parametrize("result", [np.nan, np.inf, 1j, [1.0], np.ones((1, 1))])
def test_invalid_objective_result(result):
    with pytest.raises(ValueError, match="function"):
        numerical_gradient(lambda x: result, np.ones(2))


def test_unrepresentable_step():
    with pytest.raises(ValueError, match="perturbations"):
        numerical_gradient(lambda x: np.sum(x), [1e20])


def test_nonfinite_derivative():
    with pytest.raises(ValueError, match="gradient is not finite"):
        numerical_gradient(lambda x: np.copysign(1e308, x[0]), [0.0])


def test_large_finite_derivative_does_not_overflow_intermediate_arithmetic():
    gradient = numerical_gradient(lambda x: 1e308 * x[0], [0.0])
    assert_allclose(gradient, [1e308], rtol=1e-15)


def test_subnormal_objective_values_are_not_lost_by_intermediate_halving():
    epsilon = np.nextafter(0.0, 1.0)
    gradient = numerical_gradient(lambda x: x[0], [0.0], eps=epsilon)
    assert_array_equal(gradient, [1.0])


def test_finite_derivative_when_objective_difference_overflows():
    gradient = numerical_gradient(lambda x: 1e308 * x[0], [0.0], eps=1.0)
    assert_array_equal(gradient, [1e308])


@pytest.mark.parametrize("coefficient", [0.25, 1.0])
def test_finite_derivative_when_twice_epsilon_overflows(coefficient):
    epsilon = np.finfo(np.float64).max
    gradient = numerical_gradient(lambda x: coefficient * x[0], [0.0], eps=epsilon)
    assert_array_equal(gradient, [coefficient])


@pytest.mark.parametrize("scale", [1.0, 1e300, 1e-300])
def test_relative_error_is_symmetric_and_scale_invariant(scale):
    a = np.array([[3.0, 0.0]]) * scale
    b = np.array([[0.0, 4.0]]) * scale
    assert relative_error(a, b) == pytest.approx(5 / 7)
    assert relative_error(b, a) == pytest.approx(5 / 7)
    assert relative_error(a, a) == 0
    assert relative_error(a, -a) == pytest.approx(1)


def test_relative_error_zero_empty_and_scalar_gradients():
    assert relative_error(np.zeros((2, 3)), np.zeros((2, 3))) == 0
    assert relative_error(np.empty((0, 2)), np.empty((0, 2))) == 0
    assert relative_error([0], [1e-300]) == 1
    assert relative_error(2.0, 1.0) == pytest.approx(1 / 3)


def test_relative_error_rejects_broadcasting():
    with pytest.raises(ValueError, match="shapes"):
        relative_error(np.ones((2, 1)), np.ones(2))


@pytest.mark.parametrize("bad", [[np.nan], [np.inf], [1j]])
def test_relative_error_rejects_invalid_values_in_either_argument(bad):
    with pytest.raises(ValueError):
        relative_error(bad, [1])
    with pytest.raises(ValueError):
        relative_error([1], bad)
