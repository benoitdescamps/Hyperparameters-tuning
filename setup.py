from setuptools import setup, find_packages
setup(
    name="HyperBand",
    version="0.1",
    packages=find_packages(exclude=["*.pickle", "*.pdf", "*.ipynb"]),
    author="Benoit Descamps",
    author_email="benoitdescamps@hotmail.com",
    description="Implementation of SuccessiveHalving with wrappers for sklearn and xgboost",
    license="",
    keywords="Machine Learning Hyperparameters",
    url="https://github.com/benoitdescamps/Hyperparameters-tuning",  # project home page, if any
    project_urls={
        "Source Code": "https://github.com/benoitdescamps/Hyperparameters-tuning/Hyperband"
    }
)