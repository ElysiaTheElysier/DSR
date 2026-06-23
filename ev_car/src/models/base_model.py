"""
Abstract base class for all regression models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from .config import RANDOM_STATE, CV_FOLDS, PARAM_GRIDS, BENCHMARK_DIR
from .data_loader import load_data, load_raw_test_target
from .evaluation import compute_metrics, bootstrap_ci, compute_segment_metrics


class BaseModel(ABC):
    """
    Abstract base for all EV price prediction models.

    Subclasses must implement:
        - name (str)          : e.g. "linear_regression"
        - model_type (str)    : "scaled_log" or "unscaled_raw"
        - create_model()      : return an unfitted sklearn estimator
    """

    name: str = ""
    model_type: str = ""  # "scaled_log" or "unscaled_raw"

    def __init__(self):
        self.estimator = None
        self.best_params_ = {}
        self.cv_results_ = None
        self.train_time_ = 0.0

    # ── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def create_model(self):
        """Return an unfitted scikit-learn estimator."""

    def get_param_grid(self) -> dict:
        """Return the hyperparameter grid for GridSearchCV."""
        return PARAM_GRIDS.get(self.name, {})

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
            self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            sample_weights: np.ndarray | None = None,
    ) -> "BaseModel":
        """
        Run GridSearchCV and store the best estimator.
        """
        param_grid = self.get_param_grid()
        base_estimator = self.create_model()

        if param_grid:
            grid = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                cv=CV_FOLDS,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=1,
                return_train_score=True,
            )
        else:
            # No grid → just wrap in a single-param GridSearch
            grid = GridSearchCV(
                estimator=base_estimator,
                param_grid={},
                cv=CV_FOLDS,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=1,
                return_train_score=True,
            )

        fit_params = {}
        if sample_weights is not None:
            fit_params["sample_weight"] = sample_weights

        print(f"\n{'=' * 60}")
        print(f"Training {self.name} with GridSearchCV ({CV_FOLDS}-fold)")
        print(f"Parameter grid: {param_grid}")
        print(f"{'=' * 60}")

        start = time.time()
        grid.fit(X_train, y_train, **fit_params)
        self.train_time_ = time.time() - start

        self.estimator = grid.best_estimator_
        self.best_params_ = grid.best_params_
        self.cv_results_ = grid.cv_results_

        print(f"\nBest params: {self.best_params_}")
        print(f"Best CV MSE: {-grid.best_score_:.6f}")
        print(f"Training time: {self.train_time_:.1f}s")

        return self

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict on X. If model uses log target, apply expm1 to get VND.
        """
        raw_pred = self.estimator.predict(X)
        if self.model_type == "scaled_log":
            return np.expm1(raw_pred)
        return raw_pred

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(
            self,
            X_test: pd.DataFrame,
            y_test_raw: np.ndarray,
    ) -> dict:
        """
        Predict and compute all metrics in original VND scale.
        """
        y_pred = self.predict(X_test)
        n_features = X_test.shape[1]

        metrics = compute_metrics(y_test_raw, y_pred, n_features)

        # Bootstrap 95 % CI on RMSE
        ci_low, ci_high = bootstrap_ci(y_test_raw, y_pred)
        metrics["rmse_ci_lower"] = ci_low
        metrics["rmse_ci_upper"] = ci_high
        metrics["train_time_s"] = self.train_time_
        metrics["best_params"] = str(self.best_params_)

        # Segment-specific metrics
        seg_metrics = compute_segment_metrics(y_test_raw, y_pred)
        metrics.update(seg_metrics)

        print(f"\n{'-' * 60}")
        print(f"  {self.name} -- Test-set metrics (VND scale)")
        print(f"{'-' * 60}")
        for k, v in metrics.items():
            if k == "best_params" or k.startswith("n_"):
                continue
            if isinstance(v, float):
                print(f"  {k:>30s}: {v:>18,.4f}")
            else:
                print(f"  {k:>30s}: {v}")
        print(f"{'-' * 60}")

        return metrics

    # ── Save ─────────────────────────────────────────────────────────────────

    def save_metrics(self, metrics: dict, output_dir: Path | None = None) -> Path:
        """Save metrics dict to CSV."""
        out = output_dir or BENCHMARK_DIR
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.name}_metrics.csv"
        pd.DataFrame([metrics]).to_csv(path, index=False)
        print(f"Metrics saved -> {path}")
        return path

    # ── Full pipeline ────────────────────────────────────────────────────────

    def run(self, use_sample_weights: bool = False) -> dict:
        """
        End-to-end: load data → train → evaluate → save.
        Returns the metrics dict.
        """
        X_train, X_test, y_train, y_test, sw = load_data(self.model_type)
        weights = sw if use_sample_weights else None

        self.train(X_train, y_train, weights)

        # Always evaluate in raw VND scale
        y_test_raw = (
            load_raw_test_target()
            if self.model_type == "scaled_log"
            else y_test
        )
        metrics = self.evaluate(X_test, y_test_raw)
        self.save_metrics(metrics)

        # Generate plots
        from .plotting import generate_all_plots

        y_pred = self.predict(X_test)
        generate_all_plots(
            model_name=self.name,
            y_true=y_test_raw,
            y_pred=y_pred,
            estimator=self.estimator,
            feature_names=X_train.columns.tolist(),
            X_train=X_train,
            y_train=y_train,
            model_type=self.model_type,
        )

        return metrics
