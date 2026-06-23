"""
Configuration — paths, hyperparameter grids, constants.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
BENCHMARK_DIR = REPORTS_DIR / "model_benchmark"

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS = 5

# ── Hyperparameter grids ────────────────────────────────────────────────────
PARAM_GRIDS = {
    "linear_regression": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "svr": {
        "C": [1.0, 10.0, 100.0, 1000.0],
        "epsilon": [0.001, 0.01, 0.1],
        "gamma": ["scale", "auto"],
    },
    "random_forest": {
        "n_estimators": [300, 600],
        "max_depth": [10, 20, None],
        "max_features": ["sqrt", 0.5, 0.8],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [2, 4],
    },
    "xgboost": {
        "n_estimators": [300, 600],
        "max_depth": [4, 8, 12],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [3, 5],
    },
}

# ── Model registry (populated lazily to avoid circular imports) ──────────────
MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model class in the global registry."""

    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return wrapper
