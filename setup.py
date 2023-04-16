from setuptools import setup, find_packages

setup(
    name="ChurnPrediction-E2E-ML-Pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'dill',
        'xgboost'
    ],
)
