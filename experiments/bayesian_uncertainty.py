"""Show Gaussian predictive uncertainty for nested samples of N=5, 20, and 100.

Run ``python experiments/bayesian_uncertainty.py`` after installing the package's
experiments extra. Save pointwise posterior/predictive bands as PNG and PDF under
artifacts/figures/, with prediction/training CSVs and JSON metadata in artifacts/.

The true function is a cubic, so the degree-three design includes the truth.
The observation variance is known and held fixed, as is the prior precision.
Each smaller data set is a prefix of the same 100 observations: changes in the
posterior reflect added evidence rather than unrelated resampling or tuning.
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

from glassboxml.models import BayesianLinearRegression
from glassboxml.preprocessing import PolynomialFeatures

SAMPLE_SIZES = (5, 20, 100)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts"


@dataclass(frozen=True)
class ExperimentConfig:
    """Fixed simulation settings; observation precision is derived from noise_std."""

    seed: int = 42
    alpha: float = 1.0
    noise_std: float = 0.25

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        for name in ("alpha", "noise_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        with np.errstate(over="ignore", divide="ignore", under="ignore"):
            precision = np.square(np.float64(1.0) / self.noise_std)
        if not np.isfinite(precision) or precision <= 0:
            raise ValueError("noise_std must yield a finite positive float64 precision")

    @property
    def beta(self) -> float:
        """Known noise precision, beta = 1 / noise_std**2."""
        return float((1.0 / self.noise_std) ** 2)


@dataclass(frozen=True)
class Prediction:
    """A posterior evaluated on the common grid, including both uncertainty terms."""

    n: int
    mean: NDArray[np.float64]
    epistemic_std: NDArray[np.float64]
    predictive_std: NDArray[np.float64]
    posterior_trace: float


def true_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the fixed cubic signal 0.5 + 1.2*x - 0.8*x**2 - 0.6*x**3."""
    return np.asarray(0.5 + 1.2 * x - 0.8 * x**2 - 0.6 * x**3, dtype=np.float64)


def run_experiment(
    config: ExperimentConfig,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], list[Prediction]
]:
    """Fit the three nested training sets with the same basis, alpha, and beta."""
    rng = np.random.default_rng(config.seed)
    x = rng.uniform(-1, 1, size=max(SAMPLE_SIZES))
    y = true_function(x) + rng.normal(0, config.noise_std, size=x.size)
    grid = np.linspace(-1.5, 1.5, 601)
    features = PolynomialFeatures(degree=3, include_bias=True)
    design, grid_design = features.transform(x), features.transform(grid)
    predictions = []
    for n in SAMPLE_SIZES:
        model = BayesianLinearRegression(alpha=config.alpha, beta=config.beta)
        model.fit(design[:n], y[:n])
        mean, predictive_std = model.predict(grid_design, return_std=True)
        covariance = model.posterior_covariance_
        epistemic_variance = np.einsum(
            "ij,jk,ik->i", grid_design, covariance, grid_design
        )
        # Clip roundoff-sized negative quadratic forms, not observation noise.
        epistemic_std = np.sqrt(np.maximum(epistemic_variance, 0.0))
        predictions.append(
            Prediction(
                n, mean, epistemic_std, predictive_std, float(np.trace(covariance))
            )
        )
    return x, y, grid, predictions


def save_results(
    config: ExperimentConfig,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    grid: NDArray[np.float64],
    predictions: list[Prediction],
    output: Path,
) -> None:
    """Write training data, full predictive curves, summary statistics, and settings."""
    output.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output / "bayesian_uncertainty_training.csv",
        np.column_stack((np.arange(x.size), x, y)),
        delimiter=",",
        header="index,x,y",
        comments="",
    )
    with (output / "bayesian_uncertainty_predictions.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "x", "mean", "epistemic_std", "predictive_std"])
        for prediction in predictions:
            for values in zip(
                grid,
                prediction.mean,
                prediction.epistemic_std,
                prediction.predictive_std,
                strict=True,
            ):
                writer.writerow([prediction.n, *values])
    inside = np.abs(grid) <= 1
    with (output / "bayesian_uncertainty_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "n",
                "posterior_trace",
                "mean_epistemic_std",
                "mean_predictive_std",
                "noise_std",
            ]
        )
        for prediction in predictions:
            writer.writerow(
                [
                    prediction.n,
                    prediction.posterior_trace,
                    float(prediction.epistemic_std[inside].mean()),
                    float(prediction.predictive_std[inside].mean()),
                    config.noise_std,
                ]
            )
    metadata = {
        "config": asdict(config),
        "beta": config.beta,
        "sample_sizes": SAMPLE_SIZES,
        "basis": "[1, x, x^2, x^3]; all four weights have prior precision alpha",
        "true_weights": [0.5, 1.2, -0.8, -0.6],
        "training_domain": [-1, 1],
        "prediction_domain": [-1.5, 1.5],
        "sampling": "Uniform(-1, 1) with independent Gaussian noise; nested prefixes",
        "intervals": "pointwise mean +/- 1.96*std; conditional on fixed alpha and beta",
        "summary_domain": "uniform grid points inside [-1, 1]",
        "interpretation": "epistemic uncertainty contracts; observation noise persists",
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (output / "bayesian_uncertainty_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def plot_results(
    config: ExperimentConfig,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    grid: NDArray[np.float64],
    predictions: list[Prediction],
    output: Path,
) -> None:
    """Save pointwise intervals and uncertainty/noise-floor curves without a GUI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(
            1, 3, figsize=(12, 4.4), sharey=True, layout="constrained"
        )
        for axis, prediction in zip(axes, predictions, strict=True):
            axis.fill_between(
                grid,
                prediction.mean - 1.96 * prediction.predictive_std,
                prediction.mean + 1.96 * prediction.predictive_std,
                color="#0072B2",
                alpha=0.15,
                label="95% future-observation interval",
            )
            axis.fill_between(
                grid,
                prediction.mean - 1.96 * prediction.epistemic_std,
                prediction.mean + 1.96 * prediction.epistemic_std,
                color="#0072B2",
                alpha=0.35,
                label="95% latent-function interval",
            )
            axis.plot(
                grid,
                true_function(grid),
                color="#222222",
                ls="--",
                label="True function",
            )
            axis.plot(grid, prediction.mean, color="#0072B2", label="Posterior mean")
            axis.scatter(
                x[: prediction.n],
                y[: prediction.n],
                s=18,
                color="#D55E00",
                alpha=0.8,
                label="Observed data",
                zorder=3,
            )
            for boundary in (-1, 1):
                axis.axvline(boundary, color="#888888", ls=":", lw=0.8)
            axis.set(title=f"N = {prediction.n}", xlabel="x", xlim=(-1.5, 1.5))
            axis.grid(axis="y", alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Response")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        figure.suptitle(
            "Bayesian linear regression · nested observations, fixed prior and noise",
            fontsize=14,
        )
        for extension in ("png", "pdf"):
            figure.savefig(
                figures / f"bayesian_uncertainty.{extension}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(figure)

        figure, axes = plt.subplots(
            1, 2, figsize=(10, 3.8), sharex=True, layout="constrained"
        )
        for prediction, color in zip(
            predictions, ("#D55E00", "#009E73", "#0072B2"), strict=True
        ):
            axes[0].plot(
                grid, prediction.epistemic_std, color=color, label=f"N = {prediction.n}"
            )
            axes[1].plot(
                grid,
                prediction.predictive_std,
                color=color,
                label=f"N = {prediction.n}",
            )
        axes[1].axhline(
            config.noise_std, color="#555555", ls="--", label="Observation-noise floor"
        )
        for axis, title in zip(
            axes, ("Parameter uncertainty", "Total predictive uncertainty"), strict=True
        ):
            axis.set(
                title=title,
                xlabel="x",
                ylabel="Standard deviation",
                xlim=(-1.5, 1.5),
                ylim=(0, None),
            )
            axis.grid(axis="y", alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=False, fontsize=8)
            for boundary in (-1, 1):
                axis.axvline(boundary, color="#888888", ls=":", lw=0.8)
        figure.suptitle(
            "More observations reduce epistemic uncertainty, not observation noise",
            fontsize=13,
        )
        for extension in ("png", "pdf"):
            figure.savefig(
                figures / f"bayesian_uncertainty_components.{extension}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(figure)
    metadata_path = output / "bayesian_uncertainty_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["matplotlib"] = matplotlib.__version__
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    """Run the experiment, save artifacts, and print uncertainty summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig(
            seed=args.seed, alpha=args.alpha, noise_std=args.noise_std
        )
    except ValueError as error:
        parser.error(str(error))
    x, y, grid, predictions = run_experiment(config)
    save_results(config, x, y, grid, predictions, args.output_dir)
    plot_results(config, x, y, grid, predictions, args.output_dir)
    inside = np.abs(grid) <= 1
    print("N    Mean epistemic SD    Mean predictive SD    Observation noise SD")
    for prediction in predictions:
        print(
            f"{prediction.n:<4} {prediction.epistemic_std[inside].mean():<20.6f} "
            f"{prediction.predictive_std[inside].mean():<21.6f} {config.noise_std:.6f}"
        )
    print(f"Reports and figures saved under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
