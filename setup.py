"""Exploring Reinforcement Learning in Python.

Features:
    - Write reinforcement learning algorithms in python
    - Train RL models to use with current RL python packages
    - Tune hyperparameters using Optuna
    - Use wrappers to track model performance
    - Create custom RL environments to use with RL and MARL packages
"""

from setuptools import find_packages, setup

setup(
    name="pyrl",
    description="""Exploring Reinforcement Learning in Python
        Features:
            - Write reinforcement learning algorithms in python
            - Train RL models to use with current RL python packages
            - Tune hyperparameters using Optuna
            - Use wrappers to track model performance
            - Create custom RL environments to use with RL and MARL packages""",
    version="0.1.0",
    author="Ben Whewell",
    author_email="ben.whewell@pm.me",
    url="https://github.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "ale-py",
        "gymnasium",
        "imageio",
        "numpy",
        "matplotlib",
        "optuna",
        "pettingzoo",
        "seaborn",
        "stable-baselines3",
        "sb3-contrib",
        "torch",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
            "pre-commit",
        ],
    },
)
