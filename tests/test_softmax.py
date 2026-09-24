"""Numerical, training-loop, and reference checks for softmax regression."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from glassboxml.diagnostics.gradient_check import numerical_gradient, relative_error
from glassboxml.models import SoftmaxRegression
from glassboxml.models.softmax import _loss_and_gradients, _softmax
from glassboxml.optim import SGD, Adam, GradientDescent, Momentum


@pytest.fixture
def multiclass_data():
    """Separate training and held-out draws from three well-spaced clusters."""
    rng = np.random.default_rng(42)
    centers = np.array([[-3.0, -2.0], [3.0, -2.0], [0.0, 3.0]])
    train_y = np.repeat(np.array([-7, 4, 19]), 30)
    test_y = np.repeat(np.array([-7, 4, 19]), 15)
    train_X = np.repeat(centers, 30, axis=0) + rng.normal(scale=0.4, size=(90, 2))
    test_X = np.repeat(centers, 15, axis=0) + rng.normal(scale=0.4, size=(45, 2))
    return train_X, train_y, test_X, test_y


def test_softmax_is_normalized_stable_and_invariant_to_row_shifts():
    logits = np.array([[1000.0, 999.0, -1000.0], [-1000.0] * 3, [2.0, 4.0, 3.0]])
    original = logits.copy()
    probabilities = _softmax(logits)
    assert_allclose(probabilities.sum(axis=1), 1, atol=1e-15)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert_allclose(probabilities[1], np.full(3, 1 / 3))
    assert_allclose(probabilities[0, :2], [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))])
    assert_allclose(_softmax(logits + [[-10], [2000], [100]]), probabilities)
    assert_array_equal(logits, original)


def test_cross_entropy_stays_finite_when_true_class_probability_underflows():
    X = np.array([[1.0], [-1.0]])
    y = np.array([1, 0])
    params = {"weights": np.array([[1000.0, -1000.0]]), "bias": np.zeros(2)}
    loss, gradients = _loss_and_gradients(X, y, params)
    assert loss == pytest.approx(2000.0)
    assert_allclose(gradients["weights"], [[1, -1]])
    assert_allclose(gradients["bias"], [0, 0])


@pytest.mark.parametrize("l2", [0.0, 0.37])
def test_vectorized_gradients_match_centered_finite_differences(l2):
    rng = np.random.default_rng(18)
    X = rng.normal(size=(7, 3))
    y = np.array([0, 2, 1, 2, 0, 1, 0])
    params = {"weights": rng.normal(size=(3, 3)), "bias": rng.normal(size=3)}
    saved = {name: value.copy() for name, value in params.items()}
    _, gradients = _loss_and_gradients(X, y, params, l2=l2)
    for name in params:

        def objective(value, name=name):
            return _loss_and_gradients(X, y, {**params, name: value}, l2=l2)[0]

        numeric = numerical_gradient(objective, params[name])
        assert relative_error(numeric, gradients[name]) < 1e-8
        assert_allclose(numeric, gradients[name], rtol=1e-7, atol=1e-9)
        assert_array_equal(params[name], saved[name])


def test_l2_penalty_and_gradient_apply_to_weights_only():
    X = np.array([[1.0, 2.0], [-1.0, 0.0], [0.0, -2.0]])
    y = np.array([0, 1, 2])
    params = {
        "weights": np.arange(6.0).reshape(2, 3) / 5,
        "bias": np.array([1.0, 2.0, 3.0]),
    }
    plain_loss, plain_grad = _loss_and_gradients(X, y, params)
    loss, gradients = _loss_and_gradients(X, y, params, l2=0.7)
    assert loss - plain_loss == pytest.approx(0.35 * np.sum(params["weights"] ** 2))
    assert_allclose(
        gradients["weights"] - plain_grad["weights"], 0.7 * params["weights"]
    )
    assert_array_equal(gradients["bias"], plain_grad["bias"])


def test_small_l2_prevents_unnecessary_overflow_when_squaring_large_weights():
    params = {"weights": np.array([[1e200, -1e200]]), "bias": np.zeros(2)}
    loss, gradients = _loss_and_gradients(
        np.zeros((2, 1)), np.array([0, 1]), params, l2=1e-200
    )
    assert np.isfinite(loss)
    assert loss == pytest.approx(1e200)
    assert_allclose(gradients["weights"], [[1, -1]])
    assert_array_equal(gradients["bias"], [0, 0])


def test_optimizer_requires_an_instance_and_callable_methods():
    with pytest.raises(ValueError, match="optimizer"):
        SoftmaxRegression(optimizer=SGD)

    class InvalidOptimizer:
        step = 1

        def reset(self):
            pass

    with pytest.raises(ValueError, match="optimizer"):
        SoftmaxRegression(optimizer=InvalidOptimizer())


def test_one_full_batch_step_matches_uniform_initial_probability_equation():
    X = np.array([[2.0, -1.0], [0.0, 3.0], [-2.0, 1.0], [1.0, 1.0]])
    y = np.array([4, 8, 12, 4])
    residual = np.full((4, 3), 1 / 3) - np.eye(3)[[0, 1, 2, 0]]
    model = SoftmaxRegression(
        optimizer=GradientDescent(learning_rate=0.2),
        max_epochs=1,
        batch_size=None,
        shuffle=False,
        l2=0.5,
    ).fit(X, y)
    assert_allclose(model.coef_, -0.2 * X.T @ residual / 4, atol=1e-15)
    assert_allclose(model.intercept_, -0.2 * residual.mean(axis=0), atol=1e-15)
    assert_array_equal(model.classes_, [4, 8, 12])


@pytest.mark.parametrize("shuffle", [False, True])
def test_mini_batches_include_tail_and_reproduce_manual_epoch_updates(shuffle):
    X = np.array([[2.0, 0.0], [-1.0, 2.0], [0.0, -1.0], [1.0, 3.0], [-2.0, -2.0]])
    y = np.array([0, 1, 2, 0, 1])
    model = SoftmaxRegression(
        optimizer=SGD(learning_rate=0.1),
        max_epochs=3,
        batch_size=2,
        shuffle=shuffle,
        random_state=123,
        l2=0.4,
    ).fit(X, y)
    params = {"weights": np.zeros((2, 3)), "bias": np.zeros(3)}
    rng = np.random.default_rng(123)
    for _ in range(3):
        order = rng.permutation(len(y)) if shuffle else np.arange(len(y))
        for start in range(0, len(y), 2):
            indices = order[start : start + 2]
            batch_X, batch_y = X[indices], y[indices]
            logits = batch_X @ params["weights"] + params["bias"]
            exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
            residual = exponentials / exponentials.sum(axis=1, keepdims=True)
            residual -= np.eye(3)[batch_y]
            weight_gradient = (
                batch_X.T @ residual / len(indices) + 0.4 * params["weights"]
            )
            bias_gradient = residual.mean(axis=0)
            params["weights"] -= 0.1 * weight_gradient
            params["bias"] -= 0.1 * bias_gradient
    assert_allclose(model.coef_, params["weights"], atol=1e-14)
    assert_allclose(model.intercept_, params["bias"], atol=1e-14)


@pytest.mark.parametrize("batch_size", [1, 15, None, 1000])
def test_default_optimizer_learns_multiclass_probabilities(multiclass_data, batch_size):
    X, y, test_X, test_y = multiclass_data
    model = SoftmaxRegression(max_epochs=100, batch_size=batch_size, random_state=11)
    assert model.fit(X, y) is model
    assert model.score(test_X, test_y) >= 0.98
    probabilities = model.predict_proba(test_X)
    assert probabilities.shape == (len(test_y), 3)
    assert probabilities.dtype == np.float64
    assert_allclose(probabilities.sum(axis=1), 1, atol=1e-15)
    assert_array_equal(
        model.predict(test_X), model.classes_[probabilities.argmax(axis=1)]
    )
    assert model.coef_.shape == (2, 3)
    assert model.intercept_.shape == (3,)
    assert model.n_iter_ == 100
    assert model.history_["train_loss"][-1] < model.history_["train_loss"][0]


@pytest.mark.parametrize(
    "optimizer", [GradientDescent(0.1), SGD(0.1), Momentum(0.03), Adam(0.03)]
)
def test_all_existing_optimizers_can_train_the_model(multiclass_data, optimizer):
    X, y, test_X, test_y = multiclass_data
    model = SoftmaxRegression(
        optimizer=optimizer,
        max_epochs=60,
        batch_size=15,
        random_state=3,
    ).fit(X, y)
    assert model.score(test_X, test_y) >= 0.98


def test_predictions_agree_with_sklearn_reference(multiclass_data):
    reference = pytest.importorskip("sklearn.linear_model")
    X, y, test_X, test_y = multiclass_data
    model = SoftmaxRegression(l2=0.01, max_epochs=150, random_state=12).fit(X, y)
    expected = reference.LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    assert_array_equal(model.predict(test_X), expected.predict(test_X))
    assert model.score(test_X, test_y) == pytest.approx(expected.score(test_X, test_y))


def test_stronger_l2_regularization_shrinks_weight_norm(multiclass_data):
    X, y, _, _ = multiclass_data
    kwargs = {"optimizer": GradientDescent(0.2), "max_epochs": 200, "batch_size": None}
    weak = SoftmaxRegression(l2=0, **kwargs).fit(X, y)
    strong = SoftmaxRegression(l2=1, **kwargs).fit(X, y)
    assert np.linalg.norm(strong.coef_) < np.linalg.norm(weak.coef_)
    assert strong.score(X, y) > 0.95


@pytest.mark.parametrize(
    "labels", [["zebra", "ant", "yak"], [5.0, -2.0, 9.0], [False, True, False]]
)
def test_noncontiguous_string_integer_float_and_boolean_classes(labels):
    X = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    y = np.asarray(labels)
    model = SoftmaxRegression(max_epochs=150, batch_size=None).fit(X, y)
    assert_array_equal(model.classes_, np.unique(y))
    assert_array_equal(model.predict(X), y)
    assert model.predict(X).dtype == y.dtype


def test_history_contains_full_dataset_metrics_and_unpenalized_validation_loss(
    multiclass_data,
):
    X, y, test_X, test_y = multiclass_data
    model = SoftmaxRegression(l2=0.2, max_epochs=5, batch_size=17, random_state=6).fit(
        X, y, validation_data=(test_X, test_y)
    )
    history = model.history_
    assert set(history) == {
        "train_loss",
        "train_objective",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    }
    assert all(len(values) == model.n_iter_ for values in history.values())
    train_encoded = np.searchsorted(model.classes_, y)
    val_encoded = np.searchsorted(model.classes_, test_y)
    train_ce = -np.log(model.predict_proba(X)[np.arange(len(y)), train_encoded]).mean()
    validation_ce = -np.log(
        model.predict_proba(test_X)[np.arange(len(test_y)), val_encoded]
    ).mean()
    assert history["train_loss"][-1] == pytest.approx(train_ce)
    assert history["train_objective"][-1] == pytest.approx(
        train_ce + 0.1 * np.sum(model.coef_**2)
    )
    assert history["validation_loss"][-1] == pytest.approx(validation_ce)
    assert history["train_accuracy"][-1] == model.score(X, y)
    assert history["validation_accuracy"][-1] == model.score(test_X, test_y)
    assert model.best_epoch_ == int(np.argmin(history["validation_loss"])) + 1


def test_without_validation_history_lists_are_empty():
    model = SoftmaxRegression(max_epochs=2).fit([[-1], [1]], [0, 1])
    assert model.history_["validation_loss"] == []
    assert model.history_["validation_accuracy"] == []
    assert model.best_epoch_ is None


class _ScriptedOptimizer:
    """Emit chosen weights to exercise stopping independently of convergence."""

    def __init__(self, scales):
        self.scales = scales
        self.steps = 0

    def reset(self):
        self.steps = 0

    def step(self, params, grads):
        scale = self.scales[self.steps]
        self.steps += 1
        return {"weights": np.array([[-scale, scale]]), "bias": np.zeros(2)}


def test_early_stopping_retains_history_and_restores_best_parameters():
    X, y = np.array([[-1.0], [1.0]]), np.array([0, 1])
    model = SoftmaxRegression(
        optimizer=_ScriptedOptimizer([1.0, 2.0, 3.0, 4.0, 5.0]),
        max_epochs=5,
        batch_size=None,
        early_stopping=True,
        patience=2,
        min_delta=0,
    ).fit(X, y, validation_data=(X, 1 - y))
    assert model.n_iter_ == 3
    assert model.best_epoch_ == 1
    assert len(model.history_["train_loss"]) == 3
    assert_array_equal(model.coef_, [[-1, 1]])
    assert model.history_["validation_loss"][0] < model.history_["validation_loss"][-1]


def test_min_delta_controls_patience_but_absolute_best_epoch_is_restored():
    X, y = np.array([[-1.0], [1.0]]), np.array([0, 1])
    model = SoftmaxRegression(
        optimizer=_ScriptedOptimizer([1.0, 1.01, 1.02, 1.03, 1.04]),
        max_epochs=5,
        batch_size=None,
        early_stopping=True,
        patience=2,
        min_delta=0.1,
    ).fit(X, y, validation_data=(X, y))
    assert model.n_iter_ == 3
    assert model.best_epoch_ == 3
    assert_array_equal(model.coef_, [[-1.02, 1.02]])


def test_validation_does_not_stop_or_restore_when_early_stopping_disabled():
    X, y = np.array([[-1.0], [1.0]]), np.array([0, 1])
    model = SoftmaxRegression(
        optimizer=_ScriptedOptimizer([1.0, 2.0, 3.0, 4.0]),
        max_epochs=4,
        batch_size=None,
        early_stopping=False,
        patience=1,
    ).fit(X, y, validation_data=(X, 1 - y))
    assert model.n_iter_ == 4
    assert model.best_epoch_ == 1
    assert_array_equal(model.coef_, [[-4, 4]])


def test_validation_may_omit_some_training_classes(multiclass_data):
    X, y, test_X, test_y = multiclass_data
    model = SoftmaxRegression(max_epochs=3).fit(
        X, y, validation_data=(test_X[:5], test_y[:5])
    )
    assert len(model.history_["validation_loss"]) == 3


def test_validation_never_contributes_to_training_gradients(multiclass_data):
    X, y, test_X, test_y = multiclass_data
    kwargs = {"max_epochs": 8, "batch_size": 13, "random_state": 21}
    without_validation = SoftmaxRegression(**kwargs).fit(X, y)
    with_validation = SoftmaxRegression(**kwargs).fit(
        X, y, validation_data=(test_X, test_y)
    )
    assert_array_equal(with_validation.coef_, without_validation.coef_)
    assert_array_equal(with_validation.intercept_, without_validation.intercept_)
    assert (
        with_validation.history_["train_loss"]
        == without_validation.history_["train_loss"]
    )


def test_bias_alone_fits_unequal_class_frequencies_despite_strong_l2():
    y = np.repeat([0, 1, 2], [3, 6, 9])
    X = np.zeros((len(y), 2))
    model = SoftmaxRegression(
        optimizer=GradientDescent(1), l2=10000, max_epochs=150, batch_size=None
    ).fit(X, y)
    assert_array_equal(model.coef_, np.zeros((2, 3)))
    assert_allclose(model.predict_proba([[0, 0]])[0], [1 / 6, 1 / 3, 1 / 2], atol=1e-10)


def test_seeded_training_refits_and_cloned_optimizer_are_reproducible(multiclass_data):
    X, y, _, _ = multiclass_data
    supplied = Adam(learning_rate=0.03)
    unrelated = {"unrelated": np.array([1.0])}
    supplied.step(unrelated, unrelated)
    expected_optimizer_result = Adam(learning_rate=0.03)
    expected_optimizer_result.step(unrelated, unrelated)
    first = SoftmaxRegression(
        optimizer=supplied, max_epochs=8, batch_size=13, random_state=5
    )
    second = SoftmaxRegression(
        optimizer=Adam(0.03), max_epochs=8, batch_size=13, random_state=5
    )
    first.fit(X, y)
    second.fit(X, y)
    assert_array_equal(first.coef_, second.coef_)
    assert_array_equal(first.intercept_, second.intercept_)
    assert first.history_ == second.history_
    first.fit(X, y)
    assert_array_equal(first.coef_, second.coef_)
    assert first.history_ == second.history_
    assert_array_equal(
        supplied.step(unrelated, unrelated)["unrelated"],
        expected_optimizer_result.step(unrelated, unrelated)["unrelated"],
    )


def test_shuffling_uses_local_rng_without_changing_numpy_global_state(multiclass_data):
    X, y, _, _ = multiclass_data
    before = np.random.get_state()
    SoftmaxRegression(max_epochs=3, random_state=18, batch_size=7).fit(X, y)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_different_seeds_change_mini_batch_trajectory(multiclass_data):
    X, y, _, _ = multiclass_data
    first = SoftmaxRegression(max_epochs=2, random_state=1, batch_size=11).fit(X, y)
    second = SoftmaxRegression(max_epochs=2, random_state=2, batch_size=11).fit(X, y)
    assert not np.array_equal(first.coef_, second.coef_)


def test_readonly_noncontiguous_data_and_fitted_attributes_are_independent():
    X = np.array([[-2.0, 9.0, 0.0, 9.0], [2.0, 9.0, 0.0, 9.0], [0.0, 9.0, 2.0, 9.0]])[
        :, ::2
    ]
    y = np.array([1, 9, 2, 9, 3, 9])[::2]
    original_X, original_y = X.copy(), y.copy()
    X.flags.writeable = y.flags.writeable = False
    model = SoftmaxRegression(max_epochs=5).fit(X, y, validation_data=(X, y))
    probabilities, history = model.predict_proba(X), model.history_
    model.coef_[:] = 999
    model.intercept_[:] = 999
    model.classes_[:] = 999
    model.history_["train_loss"].append(999)
    model.history_["validation_loss"][0] = 999
    assert_array_equal(model.predict_proba(X), probabilities)
    assert model.history_ == history
    assert_array_equal(X, original_X)
    assert_array_equal(y, original_y)


def test_refit_can_change_feature_count_and_classes():
    model = SoftmaxRegression(max_epochs=3).fit([[-1], [1]], [0, 1])
    model.fit([[-1, 0], [1, 0], [0, 1]], ["a", "b", "c"])
    assert model.coef_.shape == (2, 3)
    assert_array_equal(model.classes_, ["a", "b", "c"])
    assert len(model.history_["train_loss"]) == 3
    with pytest.raises(ValueError):
        model.predict([[1]])


def test_unfitted_methods_and_attributes_raise_clear_errors():
    model = SoftmaxRegression()
    for attribute in (
        "coef_",
        "intercept_",
        "classes_",
        "history_",
        "n_iter_",
        "best_epoch_",
    ):
        with pytest.raises(RuntimeError, match="fit"):
            getattr(model, attribute)
    for method in (model.predict, model.predict_proba):
        with pytest.raises(RuntimeError, match="fit"):
            method([[1]])
    with pytest.raises(RuntimeError, match="fit"):
        model.score([[1]], [0])


@pytest.mark.parametrize(
    "X,y",
    [
        ([], []),
        ([1, 2], [0, 1]),
        ([[[1]], [[2]]], [0, 1]),
        (np.empty((0, 2)), []),
        (np.empty((2, 0)), [0, 1]),
        ([[1], [2]], [0]),
        ([[1], [2]], [[0], [1]]),
        ([[1]], 0),
        ([[1], [2]], [0, 0]),
        ([[np.nan], [2]], [0, 1]),
        ([[np.inf], [2]], [0, 1]),
        ([[1j], [2]], [0, 1]),
        ([["1"], ["2"]], [0, 1]),
        ([[True], [False]], [0, 1]),
        ([[1], [2]], [0, np.nan]),
        ([[1], [2]], [0, np.inf]),
        ([[1], [2]], [0.1, 0.2]),
        ([[1], [2]], [0j, 1j]),
        ([[1], [2]], np.array([0, 1], dtype=object)),
    ],
)
def test_invalid_training_inputs_preserve_the_previous_fitted_model(X, y):
    model = SoftmaxRegression(max_epochs=2).fit([[-1], [1]], [0, 1])
    before, history = model.predict_proba([[2]]), model.history_
    classes, epochs = model.classes_, model.n_iter_
    with pytest.raises(ValueError):
        model.fit(X, y)
    assert_array_equal(model.predict_proba([[2]]), before)
    assert_array_equal(model.classes_, classes)
    assert model.history_ == history
    assert model.n_iter_ == epochs


@pytest.mark.parametrize(
    "validation_data",
    [
        (),
        ([[0]],),
        ([[0]], [0], "extra"),
        ([[0, 1]], [0]),
        ([[0], [1]], [0]),
        ([[0]], [99]),
        ([[np.nan]], [0]),
        ([[0]], [[0]]),
        (np.empty((0, 1)), []),
        ([[0]], [0.5]),
    ],
)
def test_invalid_validation_inputs_raise(validation_data):
    with pytest.raises(ValueError):
        SoftmaxRegression(max_epochs=1).fit(
            [[-1], [1]], [0, 1], validation_data=validation_data
        )


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
@pytest.mark.parametrize(
    "X", [[1], [], [[1, 2]], [[np.inf]], [[np.nan]], [[1j]], [["1"]]]
)
def test_invalid_prediction_inputs(method, X):
    model = SoftmaxRegression(max_epochs=1).fit([[-1], [1]], [0, 1])
    with pytest.raises(ValueError):
        getattr(model, method)(X)


def test_early_stopping_requires_validation_data():
    with pytest.raises(ValueError, match="validation"):
        SoftmaxRegression(early_stopping=True).fit([[-1], [1]], [0, 1])


def test_score_is_accuracy_and_counts_unseen_labels_as_incorrect():
    X = np.array([[-2.0], [2.0]])
    model = SoftmaxRegression(max_epochs=50).fit(X, [10, 20])
    assert model.score(X, [10, 20]) == 1
    assert model.score(X, [99, 20]) == 0.5
    assert model.score(X, [99, 88]) == 0
    for y in ([10], [[10], [20]], [0.5, 1.5], [np.nan, 20]):
        with pytest.raises(ValueError):
            model.score(X, y)


def test_numerical_failure_preserves_all_previous_fitted_results():
    model = SoftmaxRegression(max_epochs=2).fit([[-1], [1]], [0, 1])
    predictions, history = model.predict_proba([[2]]), model.history_
    weights, bias = model.coef_, model.intercept_
    model.optimizer = GradientDescent(learning_rate=1e308)
    with pytest.raises(FloatingPointError):
        model.fit([[-1e308], [1e308]], [0, 1])
    assert_array_equal(model.predict_proba([[2]]), predictions)
    assert_array_equal(model.coef_, weights)
    assert_array_equal(model.intercept_, bias)
    assert model.history_ == history
    assert model.n_iter_ == 2


@pytest.mark.parametrize(
    "name,value",
    [
        ("l2", -1),
        ("l2", np.nan),
        ("l2", np.inf),
        ("l2", True),
        ("l2", "1"),
        ("l2", [1]),
        ("l2", 1j),
        ("max_epochs", 0),
        ("max_epochs", -1),
        ("max_epochs", 1.5),
        ("max_epochs", True),
        ("max_epochs", "1"),
        ("max_epochs", None),
        ("batch_size", 0),
        ("batch_size", -1),
        ("batch_size", 1.5),
        ("batch_size", True),
        ("batch_size", "1"),
        ("patience", 0),
        ("patience", -1),
        ("patience", 1.5),
        ("patience", True),
        ("min_delta", -1),
        ("min_delta", np.nan),
        ("min_delta", np.inf),
        ("min_delta", True),
        ("min_delta", "1"),
        ("shuffle", 0),
        ("shuffle", "yes"),
        ("shuffle", None),
        ("early_stopping", 0),
        ("early_stopping", "yes"),
        ("random_state", -1),
        ("random_state", 0.5),
        ("random_state", True),
        ("random_state", "1"),
        ("optimizer", "adam"),
        ("optimizer", object()),
    ],
)
def test_invalid_hyperparameters(name, value):
    with pytest.raises(ValueError, match=name):
        SoftmaxRegression(**{name: value})


@pytest.mark.parametrize("name,value", [("l2", -1), ("batch_size", 0), ("shuffle", 1)])
def test_mutated_invalid_hyperparameters_are_revalidated_on_fit(name, value):
    model = SoftmaxRegression(max_epochs=2).fit([[-1], [1]], [0, 1])
    before = model.predict_proba([[2]])
    setattr(model, name, value)
    with pytest.raises(ValueError, match=name):
        model.fit([[-1], [1]], [0, 1])
    assert_array_equal(model.predict_proba([[2]]), before)
