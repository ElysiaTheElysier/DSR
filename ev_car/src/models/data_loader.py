"""
Data loader — serves the correct feature/target files for each model type.
"""

import pandas as pd
import numpy as np
from .config import PROCESSED_DIR


def load_data(model_type: str) -> tuple[
    pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Load train/test data for the given model type.

    Parameters
    ----------
    model_type : str
        "scaled_log"   → scaled X + log(price) target  (for LR, SVR)
        "unscaled_raw" → unscaled X + raw VND target    (for RF, XGBoost)

    Returns
    -------
    X_train, X_test, y_train, y_test, sample_weights
    """
    if model_type == "scaled_log":
        X_train = pd.read_csv(PROCESSED_DIR / "X_train_scaled.csv")
        X_test = pd.read_csv(PROCESSED_DIR / "X_test_scaled.csv")
        y_train = pd.read_csv(PROCESSED_DIR / "y_train_log.csv").squeeze().values
        y_test = pd.read_csv(PROCESSED_DIR / "y_test_log.csv").squeeze().values
    elif model_type == "unscaled_raw":
        X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
        X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
        y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze().values
        y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze().values
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Use 'scaled_log' or 'unscaled_raw'."
        )

    sample_weights = (
        pd.read_csv(PROCESSED_DIR / "sample_weights_train.csv").squeeze().values
    )

    return X_train, X_test, y_train, y_test, sample_weights


def load_raw_test_target() -> np.ndarray:
    """Load raw VND test target (for metric comparison across all models)."""
    return pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze().values


def load_feature_names() -> list[str]:
    """Load the list of feature names."""
    return pd.read_csv(PROCESSED_DIR / "feature_names.csv").squeeze().tolist()
