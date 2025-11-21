# PyRL 

[![Documentation Status](https://github.com/bwhewe-13/pyrl/actions/workflows/docs.yml/badge.svg)](https://bwhewe-13.github.io/pyrl/)
[![Tests](https://github.com/bwhewe-13/pyrl/actions/workflows/tests.yml/badge.svg)](https://github.com/bwhewe-13/pyrl/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/bwhewe-13/pyrl/branch/master/graph/badge.svg)](https://codecov.io/gh/bwhewe-13/pyrl)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)]

## Exploring Reinforcement Learning in Python
Features:
    - Write reinforcement learning algorithms in python
    - Train RL models to use with current RL python packages
    - Tune hyperparameters using Optuna
    - Use wrappers to track model performance
    - Create custom RL environments to use with RL and MARL packages

## Documentation

The documentation is built automatically using Sphinx and deployed to GitHub Pages. You can find the latest documentation at:

[https://bwhewe-13.github.io/pyrl/](https://bwhewe-13.github.io/pyrl/)

To build the documentation locally:

```bash
# Install dependencies
python -m pip install -r docs/requirements.txt

# Build HTML docs
cd docs
make html
# Output will be in docs/build/html
```

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/bwhewe-13/pyrl.git
cd pyrl

# Install package with development dependencies
python -m pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

To run the tests with coverage reporting:

```bash
pytest
```

Coverage reports will be generated in `coverage_html/index.html` and `coverage.xml`.