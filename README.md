# GlassBoxML

GlassBoxML is a small machine-learning framework built from first principles
with NumPy to expose the mathematics behind classical ML.

Implemented and tested:

- Poisson MLE with closed-form and numerical log-rate fitting.
- Linear and ridge regression using stable least-squares solves.
- Gradient descent, SGD, momentum, and Adam for named NumPy arrays.
- Centered finite-difference gradient checking and relative error comparison.
- MSE, RMSE, and R² regression metrics.
- One-dimensional polynomial features and a repeated bias–variance experiment.
- Bayesian linear regression with an analytical posterior and uncertainty demo.
- Multiclass softmax regression with mini-batches, validation, and early stopping.
- A reproducible MNIST benchmark comparing SGD, momentum, and Adam.
- Top-label confidence, expected calibration error, and reliability diagnostics.
- A Streamlit app for Bayesian regression and trained MNIST model exploration.

General classification metrics and the Poisson experiment script are still
placeholders, not usable implementations yet.

## Setup

Requires Python 3.12 or newer. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

NumPy and SciPy are the core dependencies. Optional extras are `experiments`
(matplotlib and scikit-learn), `demo` (Streamlit), and `typing` (mypy):

```sh
python -m pip install -e '.[dev,experiments,demo,typing]'
```

scikit-learn is reserved for datasets, baselines, and correctness comparisons;
it must not implement GlassBoxML models. pandas may be added when useful.

## Layout

```text
src/glassboxml/
    models/          # Statistical and predictive models
    optim/           # Optimization algorithms
    preprocessing/   # Feature transformations
    metrics/         # Regression, classification, and calibration metrics
    diagnostics/     # Numerical gradient checking
experiments/         # Reproducible experiment scripts
app/                 # Streamlit explorers (presentation only)
tests/               # Deterministic unit and correctness-reference tests
```

## Development

```sh
python -m pytest
python -m ruff check .
python -m ruff format --check .
# Optional, after installing the typing extra:
python -m mypy
```

pytest discovers tests in `tests/` and uses the installed package from the `src/`
layout. Install `.[dev,experiments]` to also run the scikit-learn reference
comparisons; those tests skip if scikit-learn is absent. No production model
imports scikit-learn.

New mathematical code should use type hints and docstrings, favor clear equations
over abstraction, and prioritize numerical stability. Use fixed random seeds in
tests and experiments where applicable.

## Estimator and metric API

```python
import numpy as np

from glassboxml.metrics import mse, r2_score, rmse
from glassboxml.models import LinearRegression, PoissonMLE, RidgeRegression

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([1.0, 3.0, 5.0, 7.0])
linear = LinearRegression().fit(X, y)
ridge = RidgeRegression(alpha=1.0).fit(X, y)
prediction = linear.predict(X)
print(linear.coef_, linear.intercept_)
print(mse(y, prediction), rmse(y, prediction), r2_score(y, prediction))

poisson = PoissonMLE(method="gradient").fit([0, 1, 2, 4])
print(poisson.rate_, poisson.log_likelihood([0, 1, 2]))
```

Linear models require `X.shape == (n_samples, n_features)` and a one-dimensional
target. Ridge minimizes the **sum** of squared residuals plus `alpha * ||coef||²`;
the intercept is never penalized. Poisson fitting rejects all-zero samples because
their exact MLE is zero and this estimator requires a strictly positive rate.
R² requires at least two observations; constant targets score 1 for an exact
prediction and 0 otherwise.

## Optimizers and mini-batches

`step(params, grads)` returns a new dictionary of float64 parameter arrays;
it never mutates the inputs. Momentum and Adam retain history by name and shape.
Call `reset()` before using either for a new training run. All optimizers accept
finite floating-point NumPy arrays with exactly matching names and shapes.

SGD has the same update equation as gradient descent. Its stochastic behavior
comes from the training loop's batch selection, not a different equation.
`SGD.step` explicitly delegates the shared update to `GradientDescent.step`.
There is no autograd: a model computes the analytical gradients itself.

```python
from glassboxml.optim import SGD

rng = np.random.default_rng(42)
optimizer = SGD(learning_rate=0.05)
params = {"weights": np.zeros(1), "bias": np.array(0.0)}

for _ in range(100):
    order = rng.permutation(len(y))
    for start in range(0, len(y), 2):
        batch = order[start : start + 2]
        residual = X[batch] @ params["weights"] + params["bias"] - y[batch]
        # Gradient of half the mean squared error on this batch.
        grads = {
            "weights": X[batch].T @ residual / len(batch),
            "bias": np.asarray(residual.mean()),
        }
        params = optimizer.step(params, grads)
```

The example continues from the data above. A seeded local generator makes the
batch order reproducible. `GradientDescent`, `Momentum`, and `Adam` use the same
parameter/gradient dictionary interface.

## Polynomial features and the bias–variance experiment

```python
from glassboxml.preprocessing import PolynomialFeatures

features = PolynomialFeatures(degree=3).fit_transform([-1.0, 0.0, 1.0])
# Columns are x, x², x³. The regression model fits the intercept separately.
```

The transformer accepts a vector or a single-column matrix, returns float64
features, and does not learn scaling from the data. `include_bias=True` adds a
column of ones. High-degree monomials are sensitive to input scale and can be
ill-conditioned even with a stable least-squares solver.

Run the complete experiment from the repository root:

```sh
python -m pip install -e '.[dev,experiments]'
python experiments/bias_variance.py
```

Defaults: degrees **1, 2, 3, 5, 8, 12, 20**; **30 seeds**; 30 training, 500
validation, and 1,000 test samples per seed; Gaussian noise with standard deviation
0.3 around `sin(pi*x) + 0.5*x` for uniform inputs on `[-1, 1]`. Ridge strengths
are `1e-6, 1e-4, 1e-2, 1, 100`, using the summed-error penalty convention.
Every configuration sees the same samples within each seed. Validation and test
data are independent of training data, and neither is used for fitting or model
selection. Raw monomial columns are not standardized, so alpha values depend on
this feature basis and the training-set size.

Outputs:

- `artifacts/bias_variance_runs.csv`: each seed/configuration's train, validation,
  and test MSE.
- `artifacts/bias_variance_summary.csv`: arithmetic means and sample standard
  deviations (`ddof=1`) across seeds.
- `artifacts/bias_variance_metadata.json`: settings, seed list, and versions.
- `artifacts/figures/bias_variance_{ols,ridge,fits}.{png,pdf}`: 300-dpi PNG and
  vector PDF figures with labeled axes and accessible colors.

Mean and standard deviation appear in separate panels because high-degree OLS
errors can be highly skewed. Plots preserve all values; error axes use log scales,
falling back to symlog if exact zeros occur. The illustrative-fit figure uses a
symlog response
axis to retain large excursions. This studies under/overfitting via prediction
error; it does not directly estimate a bias-squared/variance decomposition.

Use `--help` for options, for example:

```sh
python experiments/bias_variance.py --seeds 50 --train-size 40 --alphas 0.0001 0.01 1
```

Generated artifacts are ignored by Git. No notebook, pandas, or sklearn feature
transformer is used. The experiment is separate from reusable package code.

## Bayesian linear regression

`BayesianLinearRegression(alpha=1.0, beta=1.0)` models
`y = Phi @ w + noise`, with `w ~ N(0, I/alpha)` and observation noise
`N(0, I/beta)`. The precisions are fixed inputs, not learned hyperparameters.

```python
from glassboxml.models import BayesianLinearRegression

# Unlike ordinary/ridge regression, X is the complete design matrix Phi.
# An explicit constant column gives the bias weight the same prior as the rest.
basis = PolynomialFeatures(degree=3, include_bias=True)
Phi = basis.transform(X)
bayesian = BayesianLinearRegression(alpha=1.0, beta=16.0).fit(Phi, y)
mean, std = bayesian.predict(Phi, return_std=True)
posterior_mean = bayesian.posterior_mean_
posterior_covariance = bayesian.posterior_covariance_
```

The example uses the single-feature `X` and `y` defined above. The Gaussian
posterior has precision `alpha*I + beta*Phi.T@Phi`. An augmented NumPy QR solve
computes its mean and covariance without forming the Gram matrix. `predict`
returns the mean; `return_std=True` also returns the total standard deviation
`sqrt(phi.T @ S_N @ phi + 1/beta)`, including epistemic uncertainty and future
observation noise. It does not include uncertainty about alpha or beta.

```sh
python experiments/bayesian_uncertainty.py
```

This experiment uses nested samples of **5, 20, and 100** observations from a
cubic signal, a fixed degree-three basis including bias, alpha=1, and known noise
standard deviation 0.25 (beta=16). It saves 300-dpi PNG/vector PDF figures under
`artifacts/figures/bayesian_uncertainty*`, plus training data, predictive curves,
summary CSVs, and metadata under `artifacts/`. Pointwise 95% intervals show both
latent-function uncertainty and total predictive uncertainty. Vertical dotted
lines mark the training domain; predictions also extend beyond it.

With fixed precisions and nested data, parameter uncertainty contracts as more
observations arrive, while the observation-noise floor remains. The default seed
is 42; `--seed`, `--alpha`, `--noise-std`, and `--output-dir` are configurable.

## Multiclass softmax regression

```python
from glassboxml.models import SoftmaxRegression
from glassboxml.optim import Adam

X_class = np.array([[-2, -1], [-1, -2], [2, -1], [1, -2], [0, 2], [0.5, 3]])
y_class = np.array(["left", "left", "right", "right", "top", "top"])
X_validation = np.array([[-1.5, -1.5], [1.5, -1.5], [0.2, 2.5]])
y_validation = np.array(["left", "right", "top"])

classifier = SoftmaxRegression(
    optimizer=Adam(learning_rate=0.03),
    l2=0.001,
    max_epochs=200,
    batch_size=3,
    random_state=42,
    early_stopping=True,
    patience=20,
)
classifier.fit(X_class, y_class, validation_data=(X_validation, y_validation))
probabilities = classifier.predict_proba(X_validation)
labels = classifier.predict(X_validation)
accuracy = classifier.score(X_validation, y_validation)
history = classifier.history_
```

The model computes `logits = X @ W + b`, with max-logit subtraction and log-softmax
cross-entropy to avoid overflow and taking the log of an underflowed probability.
Its objective is **mean cross-entropy + (l2/2) * ||W||²**; the bias is unpenalized.
Gradients are vectorized NumPy calculations. Probability columns correspond to
the sorted original labels in `classes_`. `coef_` has shape
`(n_features, n_classes)` and `intercept_` has shape `(n_classes,)`.

Each fit starts from zero weights/bias and a copied, reset optimizer. The supplied
optimizer and input arrays are left untouched. A local seeded generator shuffles
each epoch; every observation is used once, including the final partial batch.
`batch_size=None` uses full batches. `GradientDescent`, `SGD`, `Momentum`, and
`Adam` all work through the same interface; Adam with learning rate 0.01 is the
default. Scale poorly scaled input features before training.

Validation data never supplies gradients. When early stopping is enabled,
validation data is required. Training stops after `patience` epochs without a
validation cross-entropy decrease exceeding `min_delta`, and restores the lowest
validation-loss epoch. Even improvements smaller than `min_delta` can determine
the restored epoch. Without early stopping, training completes `max_epochs`.

`history_` records complete-data, post-update `train_loss` (plain cross-entropy),
`train_objective` (including L2), `train_accuracy`, `validation_loss`, and
`validation_accuracy`. Validation lists are empty if no validation data is given.
`n_iter_` counts completed epochs and `best_epoch_` identifies the lowest
validation loss (1-based, or None without validation). History retains all
completed epochs even when the final model restores an earlier one.

## MNIST optimizer comparison

```sh
python -m pip install -e '.[experiments]'
python experiments/mnist_optimization.py

# Smaller run; keep its outputs separate from the full benchmark.
python experiments/mnist_optimization.py --train-size 10000 --validation-size 2000 \
    --epochs 10 --output-dir artifacts/mnist_quick
```

The first run downloads the public MNIST NPZ archive into `data/mnist.npz`;
subsequent runs work offline. `--data-path` accepts an existing archive with
`x_train`, `y_train`, `x_test`, and `y_test` arrays. Data and generated artifacts
are gitignored. No TensorFlow or sklearn model code is used.

Defaults use 50,000 training and 10,000 validation images from the official
training pool, plus all 10,000 official test images. Seed 42 controls splitting
and shuffling. Every optimizer receives the same split, zero initialization,
epoch permutations, 128-image mini-batches, 20 epochs, and L2 coefficient 0.0001.
Pixels are flattened and divided by 255. Early stopping is disabled so all
optimizers receive the same number of updates.

Fixed learning rates are 0.1 for SGD, 0.02 for momentum (coefficient 0.9), and
0.001 for Adam (beta1=0.9, beta2=0.999, epsilon=1e-8). These are declared in the
script, not tuned using test accuracy. Results compare these particular settings;
they do not establish that one optimizer always outperforms another.

The experiment saves:

- Eight comparisons as 300-dpi PNG and PDF under `artifacts/figures/mnist_*`:
  training loss, validation loss, confusion matrices, signed class weights with
  a shared color scale, and the six most confident incorrect predictions for
  each optimizer (including official test indices), reliability diagrams with
  confidence histograms, most confident correct predictions, and least confident
  predictions (whether correct or incorrect).
- `artifacts/mnist_optimization_metrics.json`: test accuracy/NLL, full epoch
  histories, fitting time, convergence statistics, hyperparameters, exact split
  indices, dataset SHA-256, software versions, and thread environment settings.
  It also includes test ECE, all calibration bins (empty-bin means are `null`),
  and the six ranked examples in each confidence group for each optimizer.
- `artifacts/mnist_optimization_weights.npz`: final weights, biases, and class
  labels for each optimizer, loadable with `np.load(..., allow_pickle=False)`.

Loss curves show mean negative log likelihood, excluding the L2 penalty. The JSON
also records the regularized training objective. Convergence is measured by the
first completed epoch and number of updates reaching validation NLL <= 0.35
(`--target-nll`); unreached targets are `null`. Reported fitting time includes
epoch diagnostics, excludes downloading/plotting/test evaluation, and varies
with hardware and load. Mean time per epoch is an average, not time-to-target.
The final epoch determines all test results. `--seed`, `--batch-size`, `--l2`,
sample sizes, epoch count, and output location are configurable via `--help`.

## Probability calibration

```python
from glassboxml.metrics import (
    confidence,
    expected_calibration_error,
    reliability_diagram_data,
)

probabilities = classifier.predict_proba(X_validation)
scores = confidence(probabilities)
error = expected_calibration_error(
    y_validation,
    probabilities,
    classes=classifier.classes_,
    n_bins=10,
)
bins = reliability_diagram_data(
    y_validation,
    probabilities,
    classes=classifier.classes_,
    n_bins=10,
)
occupied = bins.counts > 0
# Plot bins.mean_confidence[occupied] against bins.accuracy[occupied].
```

Confidence is the largest probability in each row. These are **top-label**
diagnostics: ECE is the count-weighted average absolute difference between
accuracy and mean confidence in each bin. `ece` is an alias for
`expected_calibration_error`; `bins.ece` gives the same statistic. Without
`classes`, labels must be probability-column indices `0, ..., K-1`; otherwise
pass the class labels in column order. Argmax ties select the first column.

Probabilities must be a nonempty `(N, K)` matrix, with at least two columns,
finite entries in `[0, 1]`, and row sums close to one (rtol=1e-6, atol=1e-8).
Binary classification uses two columns. Inputs are not modified or normalized.
The default is ten equal-width bins over `[0, 1]`, left-closed/right-open except
the last bin includes 1. Empty bins have zero count and `NaN` accuracy/confidence
in the reusable NumPy output; exclude these bins from plots. Every observation
contributes once, including confidence exactly one.

The MNIST script generates `artifacts/figures/mnist_calibration.png` and `.pdf`
for all three optimizers. Its upper panels show nonempty bins against the
perfect-calibration diagonal; lower panels show the fraction of examples per
confidence bin. `--calibration-bins` controls bin count. The additional image
grids are `mnist_confident_correct`, `mnist_confident_errors`, and
`mnist_least_confident`, with confidence, true/predicted labels, and test indices.
Ranking ties preserve test-split order. These use held-out test predictions for
diagnosis only; no probabilities or model parameters are recalibrated.

ECE depends on bin count and sample size: sparse bins are noisy, and opposing
errors within a bin can cancel. A small ECE does not establish high accuracy or
calibration of every class.

## Streamlit explorers

From the repository root:

```sh
python -m pip install -e '.[demo]'
streamlit run app/app.py
```

Open the local URL printed by Streamlit. The demo extra includes Streamlit,
Altair, and matplotlib; Bayesian exploration works immediately without downloaded
data. The app uses a light teal theme, interactive charts with hover/zoom, and
separate views for experiments, comparisons, and model explanations.

**Bayesian Regression Explorer** lets you select a cubic, sine, or linear signal;
choose random, evenly spaced, or centrally gapped observations; and change
sample count, noise, and seed. Polynomial degree, prior precision alpha, and
assumed observation noise control the model independently of generated data.
Start with **Sparse data**, **Mind the gap**, or **Complexity trap** to load a
reproducible scenario. Switch between the **Future observation** band (parameter
uncertainty plus observation noise) and the **Latent mean** band (parameter
uncertainty alone). Overlay posterior function samples to see plausible curves.
Intervals are pointwise and conditional on the chosen hyperparameters.

**Compare degrees** fits several models to identical observations and reports
training MSE, held-out MSE, and empirical 95% interval coverage. The held-out
metrics use the same 1,000 independent noisy draws for every degree, on the
training domain [-1, 1]. The plot also extends beyond that domain to expose
extrapolation. Nominal 95% coverage need not be achieved with misspecified model
or noise assumptions. **Model details** shows the posterior equations and
weights; the JSON download records settings, observations, weights, and metrics.

**MNIST Softmax Explorer** loads the experiment's saved NPZ weights. Prepare its
artifacts once if they are not already present:

```sh
python -m pip install -e '.[experiments]'
python experiments/mnist_optimization.py
```

The sidebar's **Artifact files** panel defaults to
`artifacts/mnist_optimization_weights.npz`, `data/mnist.npz`, and
`artifacts/mnist_optimization_metrics.json`. Select an optimizer to inspect:

- **Explore digits:** browse correct, incorrect, and low-confidence examples.
  Shift images, add reproducible pixel noise, or occlude a band of pixels and
  inspect the resulting probabilities. An exact signed pixel-evidence map
  decomposes the winning logit versus the runner-up, including its bias term.
  All ten global class weight maps are also available.
- **Draw a digit:** sketch with a mouse or touch. Local preprocessing crops,
  rescales, and centers the ink; the app displays the actual 28 × 28 model input,
  probabilities, and evidence. A blank canvas makes no prediction. Handwriting
  may differ from MNIST, and this linear classifier can be confidently wrong.
- **Optimizer arena:** compare saved SGD, Momentum, and Adam models on identical
  test images, alongside their recorded training/validation loss histories.
- **Calibration & errors:** inspect reliability, confusion counts, and frequent
  error pairs. Raise a confidence threshold to see the tradeoff between the
  number of images answered and accuracy on those images.

The optional JSON restores the recorded test indices and bin count after checking
the dataset SHA-256. Clear its path to evaluate the full official test pool.
Diagnostics are always recomputed from the loaded weights. Confidence is the
largest class probability; it is not posterior parameter uncertainty or a
guarantee of correctness. Pixel evidence is an exact linear decomposition,
not a causal attribution. The confidence threshold is a descriptive test-split
diagnostic, not a validated deployment policy.
Missing or malformed artifacts produce an explanation instead of training or
downloading data in the UI. Cached results invalidate when artifact sizes or
modification times change.

The app handles widgets and figures only. `glassboxml.exploration` handles data,
artifact validation, and model calls. `glassboxml.lab` handles shared-data
comparisons, drawing preprocessing, perturbations, evidence, and report exports.
`app/charts.py` constructs charts; `app/style.css` provides the visual theme.
The local `app/drawing.js` canvas uses Streamlit components v2 without external
JavaScript libraries or network services.
`SoftmaxRegression.from_parameters(coef, intercept, classes)` restores validated
copies for inference without pickle; it preserves class-column order. Training
history and optimizer state are not reconstructed, and a later `fit` starts a
fresh training run. Library tests and Streamlit `AppTest` interaction tests cover
the workflows, presets, perturbation resets, drawing state, and empty confidence
selections. App tests skip when demo dependencies are unavailable. The optional
canvas event test uses Node.js's built-in test runner and skips if Node is absent;
Node is not required to run the app.

A short project tour:

1. Load **Complexity trap** and compare degrees 1, 3, and 9. Compare training
   error with held-out error, then increase the prior precision.
2. Load **Sparse data**, switch the uncertainty band, and show posterior samples.
   Increase the number of observations while keeping the model assumptions fixed.
3. Switch to MNIST, find a confident error, and inspect which pixels favor the
   wrong digit. Try a small horizontal shift and reset it.
4. Draw your own digit, compare optimizers, and raise the confidence threshold
   to see how many images the model would leave unanswered.
