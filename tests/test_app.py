"""Streamlit interaction tests; optional when the demo extra is not installed."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("matplotlib")
from streamlit.testing.v1 import AppTest  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "app" / "app.py"


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def test_bayesian_explorer_controls_change_model_and_preserve_ui():
    app = AppTest.from_file(str(SCRIPT), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Bayesian Regression Explorer"
    before = widget(app.metric, "Training MSE").value
    widget(app.slider, "Polynomial degree").set_value(0).run()
    assert not app.exception
    assert widget(app.metric, "Basis size").value == "1"
    assert widget(app.metric, "Training MSE").value != before
    widget(app.selectbox, "Observation layout").set_value("Gap in the middle").run()
    assert not app.exception
    widget(app.selectbox, "Generating function").set_value("Sine").run()
    widget(app.select_slider, "Prior precision α").set_value(10.0).run()
    widget(app.select_slider, "Assumed noise σ").set_value(0.4).run()
    assert not app.exception


def test_mnist_explorer_artifact_loading_browsing_and_missing_files(tmp_path):
    dataset = tmp_path / "images.npz"
    np.savez(
        dataset,
        x_test=np.zeros((20, 28, 28), dtype=np.uint8),
        y_test=np.arange(20, dtype=np.uint8) % 10,
    )
    weights = tmp_path / "models.npz"
    np.savez(
        weights,
        Adam_weights=np.zeros((784, 10)),
        Adam_bias=np.arange(10.0),
        Adam_classes=np.arange(10),
        SGD_weights=np.zeros((784, 10)),
        SGD_bias=-np.arange(10.0),
        SGD_classes=np.arange(10),
        Momentum_weights=np.full((784, 10), np.nan),
        Momentum_bias=np.zeros(10),
        Momentum_classes=np.arange(10),
    )
    app = AppTest.from_file(str(SCRIPT), default_timeout=30).run()
    app.sidebar.radio[0].set_value("MNIST Softmax").run()
    widget(app.text_input, "Weights NPZ").set_value(str(weights))
    widget(app.text_input, "MNIST NPZ").set_value(str(dataset))
    widget(app.text_input, "Metrics JSON (optional)").set_value("").run()
    assert not app.exception
    assert widget(app.metric, "Test images").value == "20"
    assert widget(app.metric, "Predicted digit").value == "9"
    assert any("Cannot compare Momentum" in warning.value for warning in app.warning)
    assert len(app.tabs) == 4
    widget(app.selectbox, "Browse predictions").set_value(
        "Most confident correct"
    ).run()
    assert widget(app.metric, "True digit").value == "9"
    widget(app.selectbox, "Browse predictions").set_value(
        "Most confident incorrect"
    ).run()
    assert widget(app.metric, "True digit").value != "9"
    widget(app.number_input, "Position in selection").set_value(2).run()
    assert not app.exception
    widget(app.selectbox, "Trained optimizer").set_value("SGD").run()
    assert widget(app.metric, "Predicted digit").value == "0"
    widget(app.slider, "Horizontal shift (pixels)").set_value(3).run()
    widget(app.slider, "Pixel noise").set_value(0.2).run()
    widget(app.checkbox, "Occlude a horizontal band").check().run()
    assert not app.exception
    widget(app.button, "Reset image changes").click().run()
    assert widget(app.slider, "Horizontal shift (pixels)").value == 0
    assert widget(app.slider, "Pixel noise").value == 0
    assert not widget(app.checkbox, "Occlude a horizontal band").value
    widget(app.slider, "Minimum confidence to answer").set_value(1.0).run()
    assert widget(app.metric, "Images answered").value == "0"
    assert widget(app.metric, "Accuracy on answered").value == "—"
    # Exercise the Python half of the custom component's state handoff.
    pixels = np.zeros((280, 280), dtype=int)
    pixels[40:240, 120:140] = 255
    app.session_state["digit_canvas"] = {"pixels": pixels.ravel().tolist()}
    app.run()
    assert not app.exception
    assert len([m for m in app.metric if m.label == "Predicted digit"]) == 2
    app.session_state["digit_canvas"] = {"pixels": None}
    app.run()
    assert len([m for m in app.metric if m.label == "Predicted digit"]) == 1
    widget(app.text_input, "Weights NPZ").set_value(str(tmp_path / "missing.npz")).run()
    assert not app.exception
    assert any("trained model artifact" in info.value for info in app.info)


def test_guided_presets_posterior_draws_and_empty_comparison():
    app = AppTest.from_file(str(SCRIPT), default_timeout=30).run()
    widget(app.button, "Sparse data").click().run()
    assert not app.exception
    assert widget(app.slider, "Number of observations").value == 5
    widget(app.radio, "Uncertainty band").set_value("Latent mean").run()
    widget(app.checkbox, "Posterior samples").check().run()
    assert not app.exception
    widget(app.button, "Mind the gap").click().run()
    assert widget(app.selectbox, "Observation layout").value == "Gap in the middle"
    widget(app.button, "Complexity trap").click().run()
    assert widget(app.slider, "Polynomial degree").value == 12
    assert not app.exception
    widget(app.multiselect, "Degrees to compare").set_value([]).run()
    assert any("at least one degree" in item.value for item in app.info)
    assert not app.exception
