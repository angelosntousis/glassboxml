"""GlassBoxML interactive lab: UI composition only; mathematics lives in src/."""

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import charts
import numpy as np
import streamlit as st

from glassboxml.exploration import (
    SAMPLING,
    SIGNALS,
    MnistExploration,
    RegressionExploration,
    RegressionSettings,
    available_softmax_models,
    explore_mnist,
    explore_regression,
)
from glassboxml.lab import (
    DigitInspection,
    compare_regressions,
    confidence_tradeoff,
    inspect_digit,
    normalized_digit,
    perturb_digit,
    prepare_drawing,
    regression_report,
    training_traces,
)

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent


@st.cache_data(show_spinner=False, max_entries=32)
def regression_result(settings: RegressionSettings) -> RegressionExploration:
    """Cache deterministic model results, including independent test diagnostics."""
    return explore_regression(settings)


@st.cache_data(show_spinner=False, max_entries=16)
def comparison_results(
    settings: RegressionSettings,
    degrees: tuple[int, ...],
) -> list[tuple[RegressionSettings, RegressionExploration]]:
    """Keep comparisons fast without changing their common data."""
    return compare_regressions(settings, degrees)


@st.cache_data(show_spinner=False, max_entries=6)
def mnist_result(
    weights: str,
    data: str,
    name: str,
    metadata: str,
    revision: tuple[tuple[int, int], ...],
) -> MnistExploration:
    """File revisions invalidate cached inference when artifacts change."""
    return explore_mnist(
        Path(weights), Path(data), name, Path(metadata) if metadata else None
    )


def note(title: str, body: str) -> None:
    """Present escaped explanatory text in a consistent callout."""
    st.html(f'<div class="lab-note"><b>{escape(title)}</b> {escape(body)}</div>')


def metrics(items: list[tuple[str, str]]) -> None:
    """Render a compact row of metrics without computing model quantities."""
    for column, (label, value) in zip(st.columns(len(items)), items, strict=True):
        column.metric(label, value)


def regression_preset(name: str) -> None:
    """Apply a guided scenario before Streamlit constructs its controls."""
    state = {
        "reg_signal": "Cubic",
        "reg_sampling": "Random",
        "reg_n": 20,
        "reg_generation_noise": 0.2,
        "reg_seed": 42,
        "reg_degree": 3,
        "reg_alpha": 1.0,
        "reg_noise": 0.2,
    }
    if name == "Sparse data":
        state.update(reg_n=5, reg_alpha=0.1)
    elif name == "Mind the gap":
        state.update(reg_sampling="Gap in the middle", reg_degree=5, reg_alpha=0.01)
    else:
        state.update(
            reg_signal="Sine",
            reg_n=10,
            reg_degree=12,
            reg_alpha=0.001,
            reg_generation_noise=0.25,
        )
    st.session_state.update(state)


def regression_controls() -> RegressionSettings:
    """Group data controls separately from model assumptions."""
    with st.sidebar:
        st.subheader("Experiment controls")
        signal = st.selectbox("Generating function", SIGNALS, key="reg_signal")
        sampling = st.selectbox("Observation layout", SAMPLING, key="reg_sampling")
        n = st.slider("Number of observations", 5, 100, 20, step=5, key="reg_n")
        degree = st.slider("Polynomial degree", 0, 12, 3, key="reg_degree")
        alpha = st.select_slider(
            "Prior precision α",
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            value=1.0,
            key="reg_alpha",
        )
        noise = st.select_slider(
            "Assumed noise σ", [0.05, 0.1, 0.2, 0.4, 0.8], value=0.2, key="reg_noise"
        )
        with st.expander("Data generation & reproducibility"):
            generation_noise = st.slider(
                "Generated noise σ",
                0.05,
                0.8,
                0.2,
                step=0.05,
                key="reg_generation_noise",
            )
            seed = int(
                st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=1_000_000,
                    value=42,
                    step=1,
                    key="reg_seed",
                )
            )
        st.caption(
            "Model controls preserve your observations. "
            "Larger α shrinks weights toward zero."
        )
    return RegressionSettings(
        n_samples=n,
        degree=degree,
        alpha=alpha,
        noise_std=noise,
        generation_noise_std=generation_noise,
        seed=seed,
        signal=signal,
        sampling=sampling,
    )


def bayesian_explorer() -> None:
    """Explore posterior uncertainty, compare complexity, and export the experiment."""
    st.html('<div class="lab-kicker">01 / UNCERTAINTY LAB</div>')
    st.title("Bayesian Regression Explorer")
    st.write("How much can a model know from a handful of observations?")
    for column, preset in zip(
        st.columns(3), ("Sparse data", "Mind the gap", "Complexity trap"), strict=True
    ):
        column.button(
            preset,
            on_click=regression_preset,
            args=(preset,),
            width="stretch",
            help="Load a reproducible scenario to investigate.",
        )
    settings = regression_controls()
    try:
        result = regression_result(settings)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        st.error(f"Unable to fit these settings: {error}")
        return
    metrics(
        [
            ("Training MSE", f"{result.training_mse:.3f}"),
            ("Held-out MSE", f"{result.test_mse:.3f}"),
            ("95% interval coverage", f"{result.test_coverage:.1%}"),
            ("Basis size", str(settings.degree + 1)),
        ]
    )
    live, compare, details = st.tabs(["Live fit", "Compare degrees", "Model details"])
    with live:
        controls = st.columns([2, 1, 1])
        band = controls[0].radio(
            "Uncertainty band", ["Future observation", "Latent mean"], horizontal=True
        )
        show_truth = controls[1].checkbox("Show generating function", value=True)
        show_draws = controls[2].checkbox("Posterior samples", value=False)
        st.altair_chart(
            charts.regression_chart(
                result, truth=show_truth, draws=show_draws, latent=band == "Latent mean"
            ),
            width="stretch",
        )
        st.caption(
            "Teal: posterior mean · amber: generating function · dots: observations. "
            "Hover for values; scroll to zoom and drag to pan. "
            "The sampling domain is [−1, 1]."
        )
        if band == "Latent mean":
            note(
                "Uncertainty about the function.",
                "This narrower band excludes future observation noise. "
                "The posterior samples "
                "are plausible functions, not extra noisy observations.",
            )
        else:
            note(
                "Uncertainty has two sources.",
                "The predictive band includes uncertainty in the weights and "
                "irreducible observation noise. Switch to Latent mean to separate "
                "them; add data to see what contracts.",
            )
        st.caption(
            "Coverage and held-out MSE use 1,000 independent noisy observations "
            "on [−1, 1]. Coverage is empirical; 95% is nominal and depends on "
            "the model assumptions."
        )
    with compare:
        st.subheader("Same observations. Different complexity.")
        degrees = st.multiselect(
            "Degrees to compare", list(range(13)), default=[1, 3, 9]
        )
        if degrees:
            compared = comparison_results(settings, tuple(degrees))
            st.altair_chart(
                charts.comparison_chart([(s.degree, r) for s, r in compared]),
                width="stretch",
            )
            st.caption(
                "Dots: shared observations · gray dashed line: generating function. "
                "Curves outside [−1, 1] show extrapolation."
            )
            st.dataframe(
                [
                    {
                        "Degree": s.degree,
                        "Training MSE": r.training_mse,
                        "Held-out MSE": r.test_mse,
                        "95% coverage": r.test_coverage,
                        "Mean interval width": r.mean_interval_width,
                    }
                    for s, r in compared
                ],
                hide_index=True,
                width="stretch",
            )
            note(
                "Look beyond training loss.",
                "A smaller training error can accompany a larger held-out error. "
                "All rows share "
                "the same data, seed, prior precision, and assumed noise.",
            )
        else:
            st.info("Choose at least one degree to compare.")
    with details:
        st.subheader("The mathematics behind the curve")
        st.latex(
            r"S_N^{-1}=\alpha I+\beta\Phi^\top\Phi,\qquad m_N=\beta S_N\Phi^\top y"
        )
        st.latex(
            r"\operatorname{Var}(y_*)=\underbrace{\phi_*^\top S_N\phi_*}"
            r"_{\text{parameter uncertainty}}"
            r"+\underbrace{\beta^{-1}}_{\text{observation noise}}"
        )
        st.write(
            "The basis includes a constant term. All weights share a Gaussian prior; "
            "QR-based linear algebra computes the posterior without matrix inversion."
        )
        st.dataframe(
            {
                "Power of x": list(range(settings.degree + 1)),
                "Posterior mean weight": result.weights,
            },
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Intervals are pointwise and conditional on the selected hyperparameters."
        )
    st.download_button(
        "Download this experiment · JSON",
        regression_report(settings, result),
        file_name="glassboxml_regression.json",
        mime="application/json",
    )


@dataclass(frozen=True)
class ModelView:
    """Artifact locations and cached data needed by the MNIST presentation."""

    data: MnistExploration
    name: str
    models: tuple[str, ...]
    paths: tuple[str, str, str]
    revision: tuple[tuple[int, int], ...]


def load_mnist_view() -> ModelView | None:
    """Load artifacts and explain missing files without downloading or training."""
    with st.sidebar.expander("Artifact files"):
        weights = st.text_input(
            "Weights NPZ", str(ROOT / "artifacts/mnist_optimization_weights.npz")
        )
        dataset = st.text_input("MNIST NPZ", str(ROOT / "data/mnist.npz"))
        metadata = st.text_input(
            "Metrics JSON (optional)",
            str(ROOT / "artifacts/mnist_optimization_metrics.json"),
        )
        st.caption("Clear the JSON path to evaluate the full official test pool.")
    paths = [Path(weights).expanduser(), Path(dataset).expanduser()]
    if metadata.strip():
        paths.append(Path(metadata).expanduser())
    try:
        revision = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        models = available_softmax_models(paths[0])
        name = st.sidebar.selectbox(
            "Trained optimizer",
            models,
            index=models.index("Adam") if "Adam" in models else 0,
        )
        source = (
            str(paths[0].resolve()),
            str(paths[1].resolve()),
            str(paths[2].resolve()) if len(paths) > 2 else "",
        )
        with st.spinner("Loading saved weights and test diagnostics…"):
            data = mnist_result(source[0], source[1], name, source[2], revision)
        return ModelView(data, name, models, source, revision)
    except (OSError, ValueError, FloatingPointError) as error:
        st.info(
            "MNIST needs a trained model artifact and a local dataset. "
            "Generate them once, then return here."
        )
        st.code(
            "python -m pip install -e '.[experiments]'\n"
            "python experiments/mnist_optimization.py",
            language="bash",
        )
        st.caption(f"Artifact status: {error}")
        return None


def move_position(key: str, position: int) -> None:
    """Select a gallery image through widget state, before the next render."""
    st.session_state[key] = position


def reset_perturbations() -> None:
    """Return the stress-test controls to the original image."""
    st.session_state.update(shift_x=0, shift_y=0, pixel_noise=0.0, occlude=False)


def show_inspection(
    data: MnistExploration, inspection: DigitInspection, true_label: int | None = None
) -> None:
    """Present predictions and a mathematically exact pixel-evidence comparison."""
    labels = [
        ("Predicted digit", str(inspection.prediction)),
        ("Confidence", f"{inspection.confidence:.2%}"),
        ("Runner-up", str(inspection.runner_up)),
    ]
    if true_label is not None:
        labels.insert(1, ("True digit", str(true_label)))
    metrics(labels)
    st.altair_chart(
        charts.probabilities_chart(data.classes, inspection.probabilities),
        width="stretch",
    )
    with st.expander("Why this prediction? Inspect pixel evidence.", expanded=True):
        heatmap, explanation = st.columns([1, 2])
        with heatmap:
            st.pyplot(charts.evidence_chart(inspection), width="stretch")
        with explanation:
            st.subheader(f"{inspection.prediction} versus {inspection.runner_up}")
            st.write(
                "Red pixels favor the predicted digit; blue favors the runner-up. "
                "Each value is image intensity times the difference in class weights."
            )
            st.latex(r"z_a-z_b=\sum_j x_j(W_{ja}-W_{jb})+(b_a-b_b)")
            st.caption(
                "Pixel evidence + bias difference = "
                f"{inspection.logit_margin:.3f} logit units. "
                f"Bias contribution: {inspection.bias_difference:.3f}. "
                "This is an exact linear decomposition, not a causal explanation."
            )


def digit_browser(view: ModelView) -> None:
    """Browse real examples, perturb the input, and inspect its changing evidence."""
    data = view.data
    left, right = st.columns([2, 1])
    group = left.selectbox("Browse predictions", list(data.groups))
    indices = data.groups[group]
    if not len(indices):
        st.info("No examples in this group for the loaded model.")
        return
    key = f"position_{view.name}_{group}_{view.revision}"
    position = int(
        right.number_input(
            "Position in selection",
            min_value=1,
            max_value=len(indices),
            value=1,
            step=1,
            key=key,
        )
    )
    i = int(indices[position - 1])
    original = normalized_digit(data, i)
    with st.expander("Stress-test this digit", expanded=False):
        st.caption(
            "These controls change the input only. The trained model stays fixed."
        )
        shift_x = st.slider("Horizontal shift (pixels)", -5, 5, 0, key="shift_x")
        shift_y = st.slider("Vertical shift (pixels)", -5, 5, 0, key="shift_y")
        noise = st.slider("Pixel noise", 0.0, 0.5, 0.0, step=0.05, key="pixel_noise")
        hide = (
            st.slider("Rows to hide", 0, 28, (10, 18))
            if st.checkbox("Occlude a horizontal band", key="occlude")
            else (0, 0)
        )
        st.button("Reset image changes", on_click=reset_perturbations)
    transformed = perturb_digit(
        original, shift_x=shift_x, shift_y=shift_y, noise=noise, hide_rows=hide
    )
    inspection = inspect_digit(data, transformed)
    preview, result = st.columns([1, 3])
    with preview:
        st.image(
            transformed, width=190, clamp=True, caption=f"Test image #{data.indices[i]}"
        )
        if shift_x or shift_y or noise or hide != (0, 0):
            st.image(original, width=90, clamp=True, caption="Original")
            st.caption(
                f"Original prediction: {data.predictions[i]} "
                f"({data.confidence[i]:.1%} confidence)"
            )
        st.caption("White pixels are high intensity; black pixels are zero.")
    with result:
        show_inspection(data, inspection, int(data.labels[i]))
    st.subheader("Explore this selection")
    start = ((position - 1) // 6) * 6
    for column, offset in zip(
        st.columns(6), range(start, min(start + 6, len(indices))), strict=False
    ):
        j = int(indices[offset])
        with column:
            st.image(data.images[j], width=80)
            st.caption(
                f"True {data.labels[j]} → {data.predictions[j]} · "
                f"{data.confidence[j]:.1%}"
            )
            st.button(
                f"Inspect #{data.indices[j]}",
                key=f"gallery_{offset}",
                on_click=move_position,
                args=(key, offset + 1),
                width="stretch",
            )
    with st.expander("All ten learned class weight maps"):
        st.pyplot(charts.weight_chart(data), width="stretch")
        st.caption(
            "Shared signed scale. These global weights differ from "
            "the image-specific evidence above."
        )


@st.cache_resource
def drawing_component() -> Any:
    """Register the local canvas once, using no external JavaScript dependencies."""
    return st.components.v2.component(
        "glassboxml_digit_pad",
        html='<div class="pad"><canvas width="280" height="280" '
        'aria-label="Draw a digit here"></canvas>'
        '<button type="button">Clear drawing</button></div>',
        css=".pad{display:flex;flex-direction:column;gap:12px;align-items:center}"
        "canvas{width:280px;max-width:100%;aspect-ratio:1;"
        "border-radius:16px;touch-action:none;"
        "cursor:crosshair;background:black;border:1px solid #21334b}"
        "button{background:#edf2f7;color:#17283f;border:1px solid #ccd7e4;"
        "padding:9px 20px;border-radius:8px;cursor:pointer;font:inherit}",
        js=(APP_DIR / "drawing.js").read_text(),
    )


def draw_digit(view: ModelView) -> None:
    """Capture strokes in the browser; delegate all image preparation and prediction."""
    note(
        "Try the model on your handwriting.",
        "Draw one large digit. Release the pointer to crop, rescale, and center "
        "the ink before prediction. Your handwriting can differ substantially "
        "from the MNIST training distribution.",
    )
    pad, result = st.columns([1, 2])
    with pad:
        previous = st.session_state.get("digit_canvas", {})
        state = drawing_component()(
            data={"pixels": previous.get("pixels")},
            default={"pixels": None},
            on_pixels_change=lambda: None,
            key="digit_canvas",
            height=345,
        )
        st.caption("Mouse or touch · white ink on black · processed at 28 × 28")
    with result:
        if state.pixels is None:
            st.info("Your prediction and pixel evidence will appear here.")
            return
        try:
            image = prepare_drawing(np.asarray(state.pixels).reshape(280, 280))
        except ValueError as error:
            st.error(f"Cannot read this drawing: {error}")
            return
        if image is None:
            st.info("No visible ink after resizing. Draw a larger digit to begin.")
            return
        st.image(image, width=84, clamp=True, caption="The model's actual input")
        show_inspection(view.data, inspect_digit(view.data, image))


def optimizer_arena(view: ModelView) -> None:
    """Compare all saved estimators on the exact same loaded evaluation split."""
    st.subheader("Three update rules. One linear model.")
    rows = []
    for name in view.models:
        try:
            evaluated = mnist_result(
                view.paths[0], view.paths[1], name, view.paths[2], view.revision
            )
        except (OSError, ValueError, FloatingPointError) as error:
            st.warning(f"Cannot compare {name}: {error}")
            continue
        rows.append(
            {
                "Optimizer": name,
                "Test accuracy": evaluated.accuracy,
                "Test ECE": evaluated.calibration.ece,
                "Mean confidence": evaluated.mean_confidence,
            }
        )
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Test accuracy": st.column_config.NumberColumn(format="percent"),
            "Mean confidence": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(
        "Live diagnostics from saved weights, using identical test images. "
        "The comparison describes these trained artifacts; "
        "it does not establish a universal optimizer ranking."
    )
    if not view.paths[2]:
        st.info("Add the experiment JSON to display its recorded training traces.")
        return
    try:
        traces = training_traces(Path(view.paths[2]))
    except (ValueError, OSError, TypeError, AttributeError) as error:
        st.warning(f"Training traces unavailable: {error}")
        return
    if traces:
        split = st.radio("Recorded loss", ("Validation", "Training"), horizontal=True)
        st.altair_chart(charts.training_chart(traces, split), width="stretch")
        st.caption(
            "Recorded epoch histories from the selected metrics JSON. "
            "The app does not retrain models or reconstruct timings."
        )
    else:
        st.info("No epoch histories found in the selected JSON.")


def calibration_and_errors(view: ModelView) -> None:
    """Expose confidence failure and the coverage/accuracy tradeoff of abstention."""
    data = view.data
    left, right = st.columns(2)
    with left:
        st.subheader("Does confidence match accuracy?")
        st.altair_chart(charts.reliability_chart(data), width="stretch")
        st.caption(
            "Each dot is a nonempty confidence bin; size shows sample count. "
            "The diagonal indicates perfect top-label calibration."
        )
    with right:
        st.subheader("What if the model can abstain?")
        threshold = st.slider("Minimum confidence to answer", 0.0, 1.0, 0.9, step=0.01)
        retained = confidence_tradeoff(data, threshold)
        accuracy = retained["accuracy"]
        metrics(
            [
                ("Images answered", str(retained["retained"])),
                (
                    "Accuracy on answered",
                    f"{accuracy:.1%}" if accuracy is not None else "—",
                ),
            ]
        )
        st.progress(
            float(retained["coverage"] or 0),
            text=f"Coverage: {float(retained['coverage'] or 0):.1%}",
        )
        note(
            "Confidence is not correctness.",
            "A high threshold can discard many images and retain confident mistakes. "
            "This test-split diagnostic does not validate a deployment policy.",
        )
    st.divider()
    matrix, table = st.columns([3, 2])
    with matrix:
        st.subheader("Where the model gets confused")
        st.altair_chart(charts.confusion_chart(data), width="stretch")
    with table:
        st.subheader("Frequent error pairs")
        if data.common_errors:
            st.dataframe(data.common_errors, hide_index=True, width="stretch")
        else:
            st.info("No errors on this test split.")
        st.caption(
            "Choose Most confident incorrect in Explore digits to inspect failures."
        )


def mnist_explorer() -> None:
    """Compose an inference playground around the package's saved models."""
    st.html('<div class="lab-kicker">02 / CLASSIFICATION LAB</div>')
    st.title("MNIST Softmax Explorer")
    st.write("Make a prediction. Break it. Look inside the decision.")
    view = load_mnist_view()
    if view is None:
        return
    st.sidebar.caption("Saved models · NumPy softmax · no training on the test set")
    data = view.data
    metrics(
        [
            ("Test accuracy", f"{data.accuracy:.2%}"),
            ("Test ECE", f"{data.calibration.ece:.4f}"),
            ("Mean confidence", f"{data.mean_confidence:.2%}"),
            ("Test images", f"{len(data.labels):,}"),
        ]
    )
    explore, draw, arena, diagnostics = st.tabs(
        ["Explore digits", "Draw a digit", "Optimizer arena", "Calibration & errors"]
    )
    with explore:
        digit_browser(view)
    with draw:
        draw_digit(view)
    with arena:
        optimizer_arena(view)
    with diagnostics:
        calibration_and_errors(view)


def main() -> None:
    """Apply a coherent visual shell while keeping model logic in the package."""
    st.set_page_config(
        page_title="GlassBoxML · Interactive ML Lab", page_icon="◈", layout="wide"
    )
    st.html(f"<style>{(APP_DIR / 'style.css').read_text()}</style>")
    with st.sidebar:
        st.html('<div class="lab-brand">Glass<span>Box</span>ML</div>')
        st.caption("AN INTERACTIVE MACHINE LEARNING LAB")
        mode = st.radio(
            "Explorer",
            ("Bayesian Regression", "MNIST Softmax"),
            label_visibility="collapsed",
        )
        st.divider()
    if mode == "Bayesian Regression":
        bayesian_explorer()
    else:
        mnist_explorer()
    st.html(
        '<div class="lab-footer">GLASSBOXML · NUMPY MODELS · '
        "REPRODUCIBLE EXPERIMENTS</div>"
    )


if __name__ == "__main__":
    main()
