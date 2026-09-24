"""Repeated polynomial regression experiment; run as a script, not a notebook.

Study under/overfitting through prediction error, not a direct bias-squared /
variance decomposition. Every model receives identical data within each seed.
Validation and test observations are independent of training observations;
neither is used to fit or choose models. Features are raw monomials on [-1, 1].

Example (after installing the package and its experiments extra)::

    python experiments/bias_variance.py --seeds 30

Writes per-seed and summary CSVs plus reproducibility metadata to artifacts/,
and 300-dpi PNG / vector PDF figures to artifacts/figures/. No sklearn is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from glassboxml.metrics import mse
from glassboxml.models import LinearRegression, RidgeRegression
from glassboxml.preprocessing import PolynomialFeatures

DEGREES = (1, 2, 3, 5, 8, 12, 20)
DEFAULT_ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class ExperimentConfig:
    """Simulation settings; at least two seeds are needed for sample SD."""

    seeds: int = 30
    seed_start: int = 0
    train_size: int = 30
    validation_size: int = 500
    test_size: int = 1000
    noise_std: float = 0.3
    alphas: tuple[float, ...] = DEFAULT_ALPHAS

    def __post_init__(self) -> None:
        for name in ("seeds", "train_size", "validation_size", "test_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least 2")
        if (
            isinstance(self.seed_start, bool)
            or not isinstance(self.seed_start, int)
            or self.seed_start < 0
        ):
            raise ValueError("seed_start must be a nonnegative integer")
        if not np.isfinite(self.noise_std) or self.noise_std <= 0:
            raise ValueError("noise_std must be positive and finite")
        if not self.alphas or any(not np.isfinite(a) or a <= 0 for a in self.alphas):
            raise ValueError("alphas must contain positive finite strengths")
        if len(set(self.alphas)) != len(self.alphas):
            raise ValueError("alphas must not contain duplicates")


@dataclass(frozen=True)
class Result:
    """Prediction errors for one configuration and one independently drawn data set."""

    seed: int
    model: str
    alpha: float | None
    degree: int
    train_mse: float
    validation_mse: float
    test_mse: float


@dataclass(frozen=True)
class Summary:
    """Arithmetic means and sample standard deviations across seeds (ddof=1)."""

    model: str
    alpha: float | None
    degree: int
    n_seeds: int
    train_mse_mean: float
    train_mse_std: float
    validation_mse_mean: float
    validation_mse_std: float
    test_mse_mean: float
    test_mse_std: float


def true_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """The noiseless nonlinear target on [-1, 1]."""
    return np.asarray(np.sin(np.pi * x) + 0.5 * x, dtype=np.float64)


def generate_data(
    seed: int, config: ExperimentConfig
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Draw independent split streams so changing test size cannot change training."""
    streams = np.random.SeedSequence(seed).spawn(3)
    sizes = (config.train_size, config.validation_size, config.test_size)
    data = {}
    for split, size, stream in zip(SPLITS, sizes, streams, strict=True):
        rng = np.random.default_rng(stream)
        x = rng.uniform(-1, 1, size=size)
        y = true_function(x) + rng.normal(0, config.noise_std, size=size)
        data[split] = (x, y)
    return data


def run_experiment(config: ExperimentConfig) -> list[Result]:
    """Fit on training data only, pairing all configurations within each seed."""
    results = []
    for seed in range(config.seed_start, config.seed_start + config.seeds):
        data = generate_data(seed, config)
        for degree in DEGREES:
            polynomial = PolynomialFeatures(degree=degree, include_bias=False)
            features = {
                split: polynomial.transform(x) for split, (x, _) in data.items()
            }
            for alpha in (None, *config.alphas):
                model = LinearRegression() if alpha is None else RidgeRegression(alpha)
                model.fit(features["train"], data["train"][1])
                errors = {
                    split: mse(y, model.predict(features[split]))
                    for split, (_, y) in data.items()
                }
                results.append(
                    Result(
                        seed=seed,
                        model="OLS" if alpha is None else "Ridge",
                        alpha=alpha,
                        degree=degree,
                        train_mse=errors["train"],
                        validation_mse=errors["validation"],
                        test_mse=errors["test"],
                    )
                )
    return results


def summarize(results: list[Result]) -> list[Summary]:
    """Aggregate every (model, alpha, degree), preserving the requested mean/SD."""
    groups: dict[tuple[str, float | None, int], list[Result]] = {}
    for result in results:
        groups.setdefault((result.model, result.alpha, result.degree), []).append(
            result
        )
    summaries = []
    for (model, alpha, degree), rows in groups.items():
        if len(rows) < 2:
            raise ValueError("sample standard deviation requires at least two seeds")
        values = np.array([[r.train_mse, r.validation_mse, r.test_mse] for r in rows])
        means, stds = values.mean(axis=0), values.std(axis=0, ddof=1)
        summaries.append(
            Summary(
                model,
                alpha,
                degree,
                len(rows),
                float(means[0]),
                float(stds[0]),
                float(means[1]),
                float(stds[1]),
                float(means[2]),
                float(stds[2]),
            )
        )
    return summaries


def save_tables(
    results: list[Result],
    summaries: list[Summary],
    config: ExperimentConfig,
    output: Path,
) -> None:
    """Save exact numeric reports and settings without adding a pandas dependency."""
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("runs", results), ("summary", summaries)):
        with (output / f"bias_variance_{name}.csv").open("w", newline="") as handle:
            rows = [asdict(record) for record in records]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metadata = {
        "config": asdict(config),
        "degrees": DEGREES,
        "seeds": list(range(config.seed_start, config.seed_start + config.seeds)),
        "target": "sin(pi*x) + 0.5*x",
        "sampling": "x ~ Uniform(-1, 1); independent Gaussian noise on every split",
        "split_rng": "three SeedSequence(seed).spawn(3) streams; NumPy default_rng",
        "noise_variance": config.noise_std**2,
        "standard_deviation": "sample standard deviation across seeds, ddof=1",
        "ridge_objective": "sum squared residuals + alpha * sum squared coefficients",
        "features": "raw monomials; no scaling; model intercept is unpenalized",
        "selection": "fixed grid; no model selection or tuning on test data",
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (output / "bias_variance_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def plot_results(
    summaries: list[Summary], config: ExperimentConfig, output: Path
) -> None:
    """Save unclipped mean/SD plots with accessible colors and vector PDFs."""
    # Import plotting only when requested, allowing numerical tests without matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")

    def save(figure: Figure, name: str) -> None:
        figure.savefig(figures / f"{name}.png", dpi=300, bbox_inches="tight")
        figure.savefig(figures / f"{name}.pdf", bbox_inches="tight")
        plt.close(figure)

    def style_axis(axis: Axes, title: str, ylabel: str) -> None:
        axis.set(title=title, xlabel="Polynomial degree", ylabel=ylabel)
        axis.set_xticks(DEGREES)
        values = np.concatenate([np.asarray(line.get_ydata()) for line in axis.lines])
        if np.all(values > 0):
            axis.set_yscale("log")
            axis.set_ylim(float(values.min()) * 0.65, float(values.max()) * 1.7)
        else:
            # Preserve exact zero errors without inventing a positive floor.
            positive = values[values > 0]
            threshold = float(positive.min()) * 0.1 if positive.size else 1e-8
            axis.set_yscale("symlog", linthresh=threshold)
            axis.set_ylim(0, max(float(values.max()) * 1.7, threshold))
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        ols = sorted((r for r in summaries if r.model == "OLS"), key=lambda r: r.degree)
        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), layout="constrained")
        for split, color in zip(SPLITS, palette, strict=False):
            for axis, statistic in zip(axes, ("mean", "std"), strict=True):
                values = [getattr(r, f"{split}_mse_{statistic}") for r in ols]
                axis.plot(
                    DEGREES, values, marker="o", ms=4, color=color, label=split.title()
                )
        axes[0].axhline(
            config.noise_std**2, color="#666666", ls="--", lw=1, label="Noise variance"
        )
        style_axis(axes[0], "A   Mean prediction error", "Mean MSE")
        style_axis(axes[1], "B   Variation across data sets", "Sample SD of MSE")
        axes[0].legend(frameon=False)
        axes[1].legend(frameon=False)
        figure.suptitle(
            f"Unregularized polynomial regression · {config.seeds} seeds", fontsize=14
        )
        save(figure, "bias_variance_ols")

        figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), layout="constrained")
        panels = (
            ("train", "mean", "A   Training error"),
            ("validation", "mean", "B   Validation error"),
            ("test", "mean", "C   Test error"),
            ("test", "std", "D   Test-error variability"),
        )
        for axis, (split, statistic, title) in zip(axes.flat, panels, strict=True):
            for index, alpha in enumerate((None, *config.alphas)):
                rows = sorted(
                    (r for r in summaries if r.alpha == alpha), key=lambda r: r.degree
                )
                values = [getattr(r, f"{split}_mse_{statistic}") for r in rows]
                label = "OLS" if alpha is None else rf"Ridge $\alpha={alpha:g}$"
                axis.plot(
                    DEGREES,
                    values,
                    marker="o",
                    ms=3,
                    color="#333333"
                    if alpha is None
                    else palette[(index - 1) % len(palette)],
                    ls="--" if alpha is None else "-",
                    label=label,
                )
            if statistic == "mean":
                axis.axhline(
                    config.noise_std**2,
                    color="#999999",
                    ls=":",
                    lw=1,
                    label="Noise variance",
                )
            style_axis(
                axis, title, "Mean MSE" if statistic == "mean" else "Sample SD of MSE"
            )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="outside lower center", ncol=4, frameon=False
        )
        figure.suptitle(
            f"Ridge regularization · paired samples across {config.seeds} seeds",
            fontsize=14,
        )
        save(figure, "bias_variance_ridge")

        # Fixed illustrative seed and alpha; these are not selected by test error.
        data = generate_data(config.seed_start, config)
        x, y = data["train"]
        grid = np.linspace(-1, 1, 600)
        example_alpha = config.alphas[len(config.alphas) // 2]
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.9), layout="constrained")
        for axis, degree in zip(axes, (1, 5, 20), strict=True):
            polynomial = PolynomialFeatures(degree)
            train_features, grid_features = (
                polynomial.transform(x),
                polynomial.transform(grid),
            )
            axis.scatter(
                x, y, s=18, color="#777777", alpha=0.65, label="Training data", zorder=3
            )
            axis.plot(
                grid,
                true_function(grid),
                color="#111111",
                ls="--",
                label="True function",
            )
            for model, label, color in (
                (LinearRegression(), "OLS", palette[1]),
                (
                    RidgeRegression(example_alpha),
                    rf"Ridge $\alpha={example_alpha:g}$",
                    palette[0],
                ),
            ):
                axis.plot(
                    grid,
                    model.fit(train_features, y).predict(grid_features),
                    color=color,
                    label=label,
                )
            axis.set(
                title=f"Degree {degree}",
                xlabel="x",
                ylabel="Predicted response",
                xlim=(-1, 1),
            )
            axis.set_yscale("symlog", linthresh=2)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", alpha=0.2)
        axes[0].legend(frameon=False, fontsize=8)
        figure.suptitle(
            f"Illustrative fits · seed {config.seed_start} · "
            "symlog y preserves excursions",
            fontsize=13,
        )
        save(figure, "bias_variance_fits")

    metadata_path = output / "bias_variance_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["matplotlib"] = matplotlib.__version__
    metadata["plot_scale"] = (
        "error plots: log, or symlog if zero values occur; fits: symlog, linear below 2"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    """Run the configured experiment and print mean ± sample SD for every model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=30)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig(
            seeds=args.seeds,
            seed_start=args.seed_start,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            noise_std=args.noise_std,
            alphas=tuple(args.alphas),
        )
    except ValueError as error:
        parser.error(str(error))
    results = run_experiment(config)
    summaries = summarize(results)
    save_tables(results, summaries, config, args.output_dir)
    plot_results(summaries, config, args.output_dir)
    print(
        "Model   Alpha       Degree   Train MSE (mean ± SD)   "
        "Validation MSE (mean ± SD)   Test MSE (mean ± SD)"
    )
    for row in summaries:
        alpha_label = "—" if row.alpha is None else f"{row.alpha:g}"
        errors = "   ".join(
            f"{getattr(row, f'{split}_mse_mean'):.5g} ± "
            f"{getattr(row, f'{split}_mse_std'):.5g}"
            for split in SPLITS
        )
        print(f"{row.model:<7} {alpha_label:<11} {row.degree:<8} {errors}")
    print(
        f"\nReports: {args.output_dir.resolve()}\n"
        f"Figures: {(args.output_dir / 'figures').resolve()}"
    )


if __name__ == "__main__":
    main()
