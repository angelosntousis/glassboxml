"""Chart construction only; all predictions and diagnostics come from the package."""

from typing import Any

import altair as alt
import numpy as np
from matplotlib.figure import Figure

from glassboxml.exploration import MnistExploration, RegressionExploration
from glassboxml.lab import DigitInspection

TEAL, INK, GOLD = "#087f8c", "#182941", "#de8a38"
COLORS = [TEAL, "#8264d4", GOLD, "#cb527d", "#408acd"]


def finish(chart: Any, height: int = 330) -> Any:
    """Apply one restrained chart language to the whole lab."""
    return (
        chart.properties(height=height)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#617084",
            titleColor="#34465c",
            gridColor="#edf1f5",
            domain=False,
            titlePadding=12,
        )
        .configure_legend(orient="bottom", title=None, labelColor="#526176")
    )


def regression_chart(
    result: RegressionExploration,
    *,
    truth: bool = True,
    draws: bool = False,
    latent: bool = False,
) -> Any:
    """Layer observations, credible/predictive bounds, and optional posterior draws."""
    low, high = (
        (result.latent_lower, result.latent_upper)
        if latent
        else (result.lower, result.upper)
    )
    records = [
        {
            "x": float(x),
            "Mean": float(mean),
            "Lower": float(lo),
            "Upper": float(hi),
            "True function": float(target),
        }
        for x, mean, lo, hi, target in zip(
            result.grid, result.mean, low, high, result.truth, strict=True
        )
    ]
    base = alt.Chart(alt.InlineData(values=records)).encode(
        x=alt.X("x:Q", title="Input x", axis=alt.Axis(tickCount=9))
    )
    band = base.mark_area(color=TEAL, opacity=0.13).encode(
        y=alt.Y("Lower:Q", title="Response y"), y2="Upper:Q"
    )
    mean = base.mark_line(color=TEAL, strokeWidth=3).encode(
        y="Mean:Q",
        tooltip=[
            alt.Tooltip("x:Q", format=".2f"),
            alt.Tooltip("Mean:Q", format=".3f"),
            alt.Tooltip("Lower:Q", format=".3f"),
            alt.Tooltip("Upper:Q", format=".3f"),
        ],
    )
    layers = [band, mean]
    if truth:
        layers.append(
            base.mark_line(color=GOLD, strokeDash=[5, 4], strokeWidth=2).encode(
                y="True function:Q"
            )
        )
    if draws:
        rows = [
            {"x": float(x), "y": float(y), "Draw": str(j + 1)}
            for j in range(result.posterior_draws.shape[1])
            for x, y in zip(result.grid, result.posterior_draws[:, j], strict=True)
        ]
        layers.append(
            alt.Chart(alt.InlineData(values=rows))
            .mark_line(opacity=0.3, strokeWidth=1, color=TEAL)
            .encode(x="x:Q", y="y:Q", detail="Draw:N")
        )
    observations = [
        {"x": float(x), "y": float(y)} for x, y in zip(result.x, result.y, strict=True)
    ]
    layers.append(
        alt.Chart(alt.InlineData(values=observations))
        .mark_circle(color=INK, size=60, opacity=1, stroke="white", strokeWidth=1)
        .encode(x="x:Q", y="y:Q", tooltip=["x:Q", "y:Q"])
    )
    return finish(alt.layer(*layers).interactive(), 380)


def comparison_chart(results: list[tuple[int, RegressionExploration]]) -> Any:
    """Compare predictions against their common observations and generating function."""
    records = [
        {"x": float(x), "Prediction": float(y), "Degree": str(degree)}
        for degree, result in results
        for x, y in zip(result.grid, result.mean, strict=True)
    ]
    curves = (
        alt.Chart(alt.InlineData(values=records))
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("x:Q", title="Input x", axis=alt.Axis(tickCount=9)),
            y=alt.Y("Prediction:Q", title="Response y"),
            color=alt.Color(
                "Degree:N",
                scale=alt.Scale(range=COLORS),
                legend=alt.Legend(title="Polynomial degree"),
            ),
            tooltip=["Degree:N", "x:Q", alt.Tooltip("Prediction:Q", format=".3f")],
        )
    )
    reference = results[0][1]
    observations = (
        alt.Chart(
            alt.InlineData(
                values=[
                    {"x": float(x), "y": float(y)}
                    for x, y in zip(reference.x, reference.y, strict=True)
                ]
            )
        )
        .mark_circle(color=INK, size=40, opacity=0.8)
        .encode(x=alt.X("x:Q", title="Input x"), y=alt.Y("y:Q", title="Response y"))
    )
    truth = (
        alt.Chart(
            alt.InlineData(
                values=[
                    {"x": float(x), "y": float(y)}
                    for x, y in zip(reference.grid, reference.truth, strict=True)
                ]
            )
        )
        .mark_line(color="#aab4c1", strokeDash=[4, 4])
        .encode(x=alt.X("x:Q", title="Input x"), y=alt.Y("y:Q", title="Response y"))
    )
    return finish((truth + curves + observations).interactive())


def probabilities_chart(
    classes: np.ndarray[Any, Any], probabilities: np.ndarray[Any, Any]
) -> Any:
    records = [
        {
            "Digit": str(k),
            "Probability": float(p),
            "Winner": i == int(probabilities.argmax()),
        }
        for i, (k, p) in enumerate(zip(classes, probabilities, strict=True))
    ]
    return finish(
        alt.Chart(alt.InlineData(values=records))
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "Digit:N", sort=[str(k) for k in range(10)], axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y(
                "Probability:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=alt.condition("datum.Winner", alt.value(TEAL), alt.value("#cdd8e6")),
            tooltip=["Digit:N", alt.Tooltip("Probability:Q", format=".2%")],
        ),
        240,
    )


def evidence_chart(inspection: DigitInspection) -> Figure:
    figure = Figure(figsize=(3.8, 3.3), layout="constrained")
    axis = figure.subplots()
    limit = max(float(np.abs(inspection.contributions).max()), np.finfo(float).eps)
    image = axis.imshow(
        inspection.contributions, cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    axis.set(xticks=[], yticks=[])
    figure.colorbar(image, ax=axis, shrink=0.8, label="Logit contribution")
    return figure


def weight_chart(data: MnistExploration) -> Figure:
    figure = Figure(figsize=(10, 4.2), layout="constrained")
    axes = figure.subplots(2, 5)
    limit = max(float(np.abs(data.weights).max()), np.finfo(float).eps)
    for i, axis in enumerate(axes.flat):
        picture = axis.imshow(
            data.weights[:, i].reshape(28, 28), cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        axis.set(title=f"Digit {data.classes[i]}", xticks=[], yticks=[])
    figure.colorbar(picture, ax=list(axes.flat), shrink=0.8, label="Weight")
    return figure


def reliability_chart(data: MnistExploration) -> Any:
    bins = data.calibration
    records = [
        {"Confidence": float(c), "Accuracy": float(a), "Images": int(n)}
        for c, a, n in zip(
            bins.mean_confidence, bins.accuracy, bins.counts, strict=True
        )
        if n
    ]
    ideal = (
        alt.Chart(alt.InlineData(values=[{"x": 0, "y": 0}, {"x": 1, "y": 1}]))
        .mark_line(color="#adb9c7", strokeDash=[5, 5])
        .encode(
            x=alt.X(
                "x:Q", title="Mean confidence", axis=alt.Axis(format="%", tickCount=6)
            ),
            y=alt.Y(
                "y:Q",
                title="Empirical accuracy",
                axis=alt.Axis(format="%", tickCount=6),
            ),
        )
    )
    dots = (
        alt.Chart(alt.InlineData(values=records))
        .mark_circle(color=TEAL, opacity=0.9)
        .encode(
            x=alt.X(
                "Confidence:Q", title="Mean confidence", scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y(
                "Accuracy:Q", title="Empirical accuracy", scale=alt.Scale(domain=[0, 1])
            ),
            size=alt.Size("Images:Q", scale=alt.Scale(range=[40, 500]), legend=None),
            tooltip=[
                alt.Tooltip("Confidence:Q", format=".2%"),
                alt.Tooltip("Accuracy:Q", format=".2%"),
                "Images:Q",
            ],
        )
    )
    return finish(ideal + dots, 280)


def confusion_chart(data: MnistExploration) -> Any:
    records = [
        {"True": str(i), "Predicted": str(j), "Images": int(data.confusion[i, j])}
        for i, j in np.ndindex(10, 10)
    ]
    return finish(
        alt.Chart(alt.InlineData(values=records))
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("Predicted:N", axis=alt.Axis(labelAngle=0)),
            y="True:N",
            color=alt.Color("Images:Q", scale=alt.Scale(scheme="tealblues")),
            tooltip=["True:N", "Predicted:N", "Images:Q"],
        ),
        310,
    )


def training_chart(rows: list[dict[str, Any]], split: str) -> Any:
    return finish(
        alt.Chart(alt.InlineData(values=[r for r in rows if r["Split"] == split]))
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("Epoch:Q", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("NLL:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Optimizer:N", scale=alt.Scale(range=COLORS)),
            tooltip=["Optimizer:N", "Epoch:Q", alt.Tooltip("NLL:Q", format=".4f")],
        )
        .interactive()
    )
