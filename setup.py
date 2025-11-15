"""
Setup script for Smadex LTV Prediction
"""
from setuptools import setup, find_packages

setup(
    name="smadex-ltv-prediction",
    version="1.0.0",
    description="Production-ready LTV prediction system for Smadex Datathon",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "xgboost>=2.0.0",
        "dask[dataframe]>=2023.0.0",
        "pyarrow>=12.0.0",
        "streamlit>=1.28.0",
        "plotly>=5.14.0",
        "pyyaml>=6.0",
        "joblib>=1.3.0"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)