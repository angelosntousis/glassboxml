# GlassBoxML

GlassBoxML is a small machine-learning framework built from first principles
with NumPy to expose the mathematics behind classical ML.

**Status:** repository scaffold only. The modules and experiment scripts are
placeholders; no algorithms or Streamlit demo have been implemented yet.

Planned work includes maximum-likelihood estimation, linear and ridge regression,
polynomial features, Bayesian linear regression, softmax regression, optimizers,
gradient checking, metrics, calibration diagnostics, and reproducible experiments.

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
app/                 # Future Streamlit demo
tests/               # Future deterministic tests
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
layout. No tests exist yet, so pytest currently exits with code 5 (no tests
collected). New mathematical code should use type hints and docstrings, favor
clear equations over abstraction, and prioritize numerical stability. Use fixed
random seeds in tests and experiments where applicable.
