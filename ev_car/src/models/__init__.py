"""
EV Car Price Prediction — Model Training Framework
====================================================

Modular training pipeline for 4 regression models:
- Linear Regression (Ridge)
- Support Vector Regression (SVR)
- Random Forest Regression
- XGBoost Regression

Usage:
    uv run python -m src.models.run_all --models linear_regression svr
    uv run python -m src.models.run_all --models linear_regression svr --benchmark
"""
