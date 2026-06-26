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
# Lưới tham số được tinh chỉnh dựa trên kết quả tốt nhất của các lượt chạy trước để tìm tối ưu toàn cục
PARAM_GRIDS = {
    "linear_regression": {
        "alpha": [1.0, 10.0, 100.0, 500.0, 1000.0],  # Mở rộng dải alpha cao hơn vì trước đó chọn alpha=100
    },
    "svr": {
        "C": [0.1, 1.0, 10.0, 50.0],                # Thu hẹp quanh mức C=1.0 tốt nhất
        "epsilon": [0.05, 0.1, 0.2],                # Thu hẹp quanh mức epsilon=0.1 tốt nhất
        "gamma": ["scale", "auto"],
    },
    "random_forest": {
        "n_estimators": [300, 500],                 # Tăng số lượng cây để đảm bảo độ chính xác
        "max_depth": [15, 20, None],                # Tập trung vào cây sâu
        "max_features": ["sqrt", 0.5, 0.7],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [2, 4],
    },
    "xgboost": {
        "n_estimators": [300, 500],                 # Mở rộng cận trên vì trước đó chọn 600 cây
        "max_depth": [6, 8, 10],                    # Thay đổi dải độ sâu vì trước đó chọn độ sâu=8
        "learning_rate": [0.05, 0.1, 0.15],         # Tập trung xung quanh mức 0.1 tốt nhất
        "subsample": [0.7, 0.8, 0.9],               # Tinh chỉnh quanh mức 0.8
        "colsample_bytree": [0.7, 0.8, 0.9],        # Tinh chỉnh quanh mức 0.8
    },
    # "lightgbm" disabled per user request
    # "lightgbm": {
    #     "n_estimators": [300, 500],
    #     "max_depth": [6, 8, -1],
    #     "num_leaves": [31, 63, 127],
    #     "learning_rate": [0.05, 0.1, 0.15],
    #     "subsample": [0.7, 0.8, 0.9],
    #     "colsample_bytree": [0.7, 0.8, 0.9],
    # },
}

# ── Model registry (populated lazily to avoid circular imports) ──────────────
MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Decorator to register a model class in the global registry."""

    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return wrapper
