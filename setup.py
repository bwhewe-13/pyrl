from setuptools import find_packages, setup

setup(
    name="pyrl",
    description="""
    Applying RL using Python and Gymnasium.
    """,
    version="0.1.0",
    author="Ben Whewell",
    install_requires=[
        "ale-py",
        "gymnasium",
        "imageio",
        "numpy",
        "matplotlib",
        "optuna",
        "seaborn",
        "stable-baselines3",
        "sb3-contrib",
        "torch",
        "tqdm",
    ],
    packages=find_packages(),
)
