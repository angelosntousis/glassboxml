"""Update equations, ownership, and deterministic optimizer integration tests."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from glassboxml.optim import SGD, Adam, GradientDescent, Momentum, Optimizer

OPTIMIZERS = [GradientDescent, SGD, Momentum, Adam]
STATEFUL = [Momentum, Adam]


@pytest.mark.parametrize("optimizer_type", [GradientDescent, SGD])
def test_plain_descent_equation(optimizer_type):
    optimizer = optimizer_type(learning_rate=0.2)
    params = {"weights": np.array([[1.0, -2.0]]), "bias": np.array(3.0)}
    grads = {"weights": np.array([[0.5, -1.5]]), "bias": np.array(-2.0)}
    updated = optimizer.step(params, grads)
    assert_allclose(updated["weights"], [[0.9, -1.7]])
    assert_allclose(updated["bias"], 3.4)
    second = optimizer.step(updated, grads)
    assert_allclose(second["weights"], [[0.8, -1.4]])
    assert_allclose(second["bias"], 3.8)


def test_momentum_two_steps_and_zero_gradient_carry():
    optimizer = Momentum(learning_rate=0.1, momentum=0.5)
    params = {"w": np.array([1.0, -2.0])}
    params = optimizer.step(params, {"w": np.array([2.0, -4.0])})
    assert_allclose(params["w"], [0.8, -1.6])
    # v = 0.5 * [2, -4] + [4, 2] = [5, 0].
    params = optimizer.step(params, {"w": np.array([4.0, 2.0])})
    assert_allclose(params["w"], [0.3, -1.6])
    params = optimizer.step(params, {"w": np.zeros(2)})
    assert_allclose(params["w"], [0.05, -1.6])


def test_zero_momentum_reduces_to_gradient_descent():
    plain = GradientDescent(learning_rate=0.1)
    momentum = Momentum(learning_rate=0.1, momentum=0)
    a = b = {"w": np.array([2.0, 3.0])}
    for g in ([1.0, -3.0], [0.0, 2.0], [-1.0, 1.0]):
        grads = {"w": np.array(g)}
        a, b = plain.step(a, grads), momentum.step(b, grads)
        assert_array_equal(a["w"], b["w"])


def test_adam_two_steps_with_bias_correction_and_epsilon_outside_sqrt():
    optimizer = Adam(learning_rate=0.2, beta1=0.5, beta2=0.75, eps=0.1)
    params = {"w": np.array([1.0, -2.0])}
    params = optimizer.step(params, {"w": np.array([2.0, -4.0])})
    expected = np.array([1 - 0.4 / 2.1, -2 + 0.8 / 4.1])
    assert_allclose(params["w"], expected, rtol=1e-14)
    # At t=2, m=[2.5, 0], v=[4.75, 4]; corrections are 0.75 and 0.4375.
    params = optimizer.step(params, {"w": np.array([4.0, 2.0])})
    expected[0] -= 0.2 * (10 / 3) / (np.sqrt(76 / 7) + 0.1)
    assert_allclose(params["w"], expected, rtol=1e-14)


def test_adam_zero_betas_and_zero_gradients():
    optimizer = Adam(learning_rate=0.1, beta1=0, beta2=0, eps=0.5)
    params = {"w": np.array([1.0, 2.0])}
    updated = optimizer.step(params, {"w": np.array([0.0, -2.0])})
    assert_allclose(updated["w"], [1, 2.08])
    assert_array_equal(optimizer.step(updated, {"w": np.zeros(2)})["w"], updated["w"])


@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
def test_independent_names_and_mapping_order(optimizer_type):
    joint = optimizer_type()
    one, two = optimizer_type(), optimizer_type()
    params = {"weights": np.array([2.0]), "bias": np.array([-3.0])}
    p1, p2 = {"weights": params["weights"]}, {"bias": params["bias"]}
    for _ in range(3):
        grads = {"bias": np.array([-2.0]), "weights": np.array([1.0])}
        params = joint.step(dict(reversed(list(params.items()))), grads)
        p1 = one.step(p1, {"weights": grads["weights"]})
        p2 = two.step(p2, {"bias": grads["bias"]})
        assert_array_equal(params["weights"], p1["weights"])
        assert_array_equal(params["bias"], p2["bias"])


@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inputs_untouched_and_results_own_their_storage(optimizer_type, dtype):
    values = np.arange(12, dtype=dtype).reshape(3, 4).T[::2]
    values.flags.writeable = False
    original = values.copy()
    # Deliberately alias params and grads, and two parameter names.
    params = {"w": values, "other": values, "scalar": np.array(2.0, dtype=dtype)}
    updated = optimizer_type().step(params, params)
    assert_array_equal(values, original)
    assert_array_equal(params["scalar"], 2.0)
    assert updated is not params
    for name in params:
        assert isinstance(updated[name], np.ndarray)
        assert updated[name].shape == params[name].shape
        assert updated[name].dtype == np.float64
        assert not np.shares_memory(updated[name], params[name])
    assert not np.shares_memory(updated["w"], updated["other"])
    updated["w"][:] = -99
    assert_array_equal(values, original)
    assert not np.any(updated["other"] == -99)


@pytest.mark.parametrize("optimizer_type", STATEFUL)
def test_state_cannot_be_mutated_through_inputs_or_outputs(optimizer_type):
    optimizer, reference = optimizer_type(), optimizer_type()
    params = {"w": np.array([2.0])}
    grads = {"w": np.array([3.0])}
    output = optimizer.step(params, grads)
    reference.step(params, grads)
    output["w"][:] = 99
    params["w"][:] = 88
    grads["w"][:] = 77
    params, grads = {"w": np.array([1.0])}, {"w": np.array([-2.0])}
    assert_array_equal(
        optimizer.step(params, grads)["w"], reference.step(params, grads)["w"]
    )


@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
def test_protocol_reset_and_zero_gradient(optimizer_type):
    optimizer = optimizer_type()
    assert isinstance(optimizer, Optimizer)
    params = {"w": np.array([1.0, 2.0])}
    optimizer.step(params, params)
    optimizer.reset()
    # New names/shapes are allowed after reset; even Adam's counter is cleared.
    params = {"bias": np.array(3.0)}
    grads = {"bias": np.array(-2.0)}
    assert_array_equal(
        optimizer.step(params, grads)["bias"],
        optimizer_type().step(params, grads)["bias"],
    )
    optimizer.reset()
    assert_array_equal(optimizer.step(params, {"bias": np.array(0.0)})["bias"], 3.0)


@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
def test_empty_arrays_are_supported(optimizer_type):
    params = {"w": np.empty((2, 0))}
    result = optimizer_type().step(params, params)
    assert result["w"].shape == (2, 0)


@pytest.mark.parametrize("optimizer_type", STATEFUL)
@pytest.mark.parametrize("changed", [{"new": np.ones(2)}, {"w": np.ones((2, 1))}])
def test_state_schema_changes_require_reset(optimizer_type, changed):
    optimizer = optimizer_type()
    params = {"w": np.ones(2)}
    optimizer.step(params, params)
    with pytest.raises(ValueError, match="reset"):
        optimizer.step(changed, changed)
    optimizer.reset()
    optimizer.step(changed, changed)


@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
@pytest.mark.parametrize(
    "params, grads",
    [
        ({}, {}),
        (np.ones(2), {"w": np.ones(2)}),
        ({"w": np.ones(2)}, np.ones(2)),
        ({1: np.ones(2)}, {1: np.ones(2)}),
        ({"w": np.ones(2)}, {}),
        ({"w": np.ones(2)}, {"other": np.ones(2)}),
        ({"w": np.ones((2, 1))}, {"w": np.ones(2)}),
        ({"w": np.ones(2)}, {"w": np.ones(1)}),
    ],
)
def test_invalid_mappings_and_shapes(optimizer_type, params, grads):
    with pytest.raises(ValueError):
        optimizer_type().step(params, grads)


@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
@pytest.mark.parametrize(
    "bad",
    [
        [1.0],
        np.array([1]),
        np.array([True]),
        np.array([1j]),
        np.array([np.nan]),
        np.array([np.inf]),
    ],
)
def test_invalid_array_values_in_either_argument(optimizer_type, bad):
    good = {"w": np.ones(1)}
    with pytest.raises(ValueError):
        optimizer_type().step({"w": bad}, good)
    with pytest.raises(ValueError):
        optimizer_type().step(good, {"w": bad})


@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
@pytest.mark.parametrize("failure", ["invalid", "overflow"])
def test_failed_steps_do_not_advance_history(optimizer_type, failure):
    optimizer = optimizer_type(learning_rate=10)
    reference = optimizer_type(learning_rate=10)
    params = {"first": np.ones(1), "second": np.ones(1)}
    grads = {"first": np.ones(1), "second": np.ones(1)}
    optimizer.step(params, grads)
    reference.step(params, grads)
    # A failure in the second array must not commit the first array's history.
    bad = {
        "first": np.array([2.0]),
        "second": np.array([np.nan if failure == "invalid" else 1e308]),
    }
    with pytest.raises(ValueError if failure == "invalid" else FloatingPointError):
        optimizer.step(params, bad)
    actual = optimizer.step(params, grads)
    expected = reference.step(params, grads)
    for name in params:
        assert_array_equal(actual[name], expected[name])
        assert_array_equal(params[name], [1.0])
        assert_array_equal(grads[name], [1.0])


@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
@pytest.mark.parametrize("bad", [0, -0.1, np.nan, np.inf, True, "0.1", [0.1], 10**1000])
def test_invalid_learning_rate(optimizer_type, bad):
    with pytest.raises(ValueError, match="learning_rate"):
        optimizer_type(learning_rate=bad)


@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
def test_learning_rate_changes_are_validated_without_advancing_state(optimizer_type):
    optimizer = optimizer_type(learning_rate=0.1)
    reference = optimizer_type(learning_rate=0.1)
    params = {"w": np.array([2.0])}
    grads = {"w": np.array([1.0])}
    optimizer.step(params, grads)
    reference.step(params, grads)
    optimizer.learning_rate = -0.1
    with pytest.raises(ValueError, match="learning_rate"):
        optimizer.step(params, grads)
    optimizer.learning_rate = reference.learning_rate = 0.05
    assert_array_equal(
        optimizer.step(params, grads)["w"], reference.step(params, grads)["w"]
    )


@pytest.mark.parametrize(
    "optimizer_type, name, invalid",
    [
        (Momentum, "momentum", 1),
        (Adam, "beta1", 1),
        (Adam, "beta2", -1),
        (Adam, "eps", 0),
    ],
)
def test_invalid_changed_hyperparameters_leave_state_intact(
    optimizer_type, name, invalid
):
    optimizer, reference = optimizer_type(), optimizer_type()
    params = {"w": np.array([2.0])}
    grads = {"w": np.array([1.0])}
    optimizer.step(params, grads)
    reference.step(params, grads)
    original = getattr(optimizer, name)
    setattr(optimizer, name, invalid)
    with pytest.raises(ValueError, match=name):
        optimizer.step(params, grads)
    setattr(optimizer, name, original)
    assert_array_equal(
        optimizer.step(params, grads)["w"], reference.step(params, grads)["w"]
    )


@pytest.mark.parametrize("name", ["momentum", "beta1", "beta2"])
@pytest.mark.parametrize("bad", [-0.1, 1, np.nan, np.inf, True, "0.9"])
def test_invalid_decay_coefficients(name, bad):
    optimizer_type = Momentum if name == "momentum" else Adam
    with pytest.raises(ValueError, match=name):
        optimizer_type(**{name: bad})


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, True, "small"])
def test_invalid_adam_epsilon(bad):
    with pytest.raises(ValueError, match="eps"):
        Adam(eps=bad)


@pytest.mark.parametrize("optimizer_type", OPTIMIZERS)
def test_convex_quadratic_converges_to_known_minimum(optimizer_type):
    optimizer = optimizer_type(learning_rate=0.05)
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    target = np.array([1.0, -2.0])
    params = {"weights": np.array([4.0, 3.0]), "bias": np.array(-3.0)}
    for _ in range(1500):
        # Loss = 0.5*(w-target).T @ matrix @ (w-target) + 0.5*(bias-0.5)**2.
        grads = {
            "weights": matrix @ (params["weights"] - target),
            "bias": np.asarray(params["bias"] - 0.5),
        }
        params = optimizer.step(params, grads)
    assert_allclose(params["weights"], target, rtol=0, atol=1e-7)
    assert_allclose(params["bias"], 0.5, rtol=0, atol=1e-7)


def test_seeded_minibatch_sgd_is_reproducible_and_minimizes_quadratic():
    # Mean squared distance to observations has its minimum at their mean.
    observations = np.array([[-1.0, 1.0], [1.0, 3.0], [3.0, 5.0], [5.0, 7.0]])

    def train(seed):
        rng = np.random.default_rng(seed)
        optimizer = SGD(learning_rate=0.1)
        params = {"w": np.array([10.0, -10.0])}
        trajectory = []
        for epoch in range(1000):
            optimizer.learning_rate = 0.1 / (1 + epoch / 10)
            order = rng.permutation(len(observations))
            for start in range(0, len(observations), 2):
                batch = observations[order[start : start + 2]]
                grads = {"w": np.mean(params["w"] - batch, axis=0)}
                params = optimizer.step(params, grads)
                trajectory.append(params["w"])
        return np.array(trajectory)

    first = train(42)
    assert_array_equal(first, train(42))
    assert not np.array_equal(first, train(43))
    assert_allclose(first[-1], np.mean(observations, axis=0), rtol=0, atol=2e-3)
