"""
Plot generation for model evaluation.

All plots are LaTeX-safe (no Vietnamese Unicode) and saved as PDF.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

from .config import BENCHMARK_DIR, RANDOM_STATE, CV_FOLDS

# ── Matplotlib LaTeX rcParams ────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def _save(fig: plt.Figure, name: str, model_name: str, output_dir: Path | None = None):
    out = output_dir or BENCHMARK_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{model_name}_{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {path}")


# ── 1. Actual vs Predicted ───────────────────────────────────────────────────

def plot_actual_vs_predicted(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Convert to billions for readability
    yt = y_true / 1e9
    yp = y_pred / 1e9

    ax.scatter(yt, yp, alpha=0.3, s=12, edgecolors="none")

    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "r--", linewidth=1.2, label="Ideal")

    ax.set_xlabel(r"Actual Price (billion VND)")
    ax.set_ylabel(r"Predicted Price (billion VND)")
    ax.set_title(rf"\textbf{{{model_name.replace('_', ' ').title()}}} — Actual vs Predicted")
    ax.legend(loc="upper left")
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "actual_vs_predicted", model_name)


# ── 2. Residual Distribution ────────────────────────────────────────────────

def plot_residual_distribution(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
):
    fig, ax = plt.subplots(figsize=(7, 5))

    residuals = (y_true - y_pred) / 1e9  # in billions

    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7, density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)

    ax.set_xlabel(r"Residual (billion VND)")
    ax.set_ylabel(r"Density")
    ax.set_title(rf"\textbf{{{model_name.replace('_', ' ').title()}}} — Residual Distribution")
    fig.tight_layout()
    _save(fig, "residual_distribution", model_name)


# ── 3. Residual vs Predicted ────────────────────────────────────────────────

def plot_residual_vs_predicted(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray, **kwargs
):
    fig, ax = plt.subplots(figsize=(7, 5))

    residuals = (y_true - y_pred) / 1e9
    predicted = y_pred / 1e9

    ax.scatter(predicted, residuals, alpha=0.3, s=12, edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)

    ax.set_xlabel(r"Predicted Price (billion VND)")
    ax.set_ylabel(r"Residual (billion VND)")
    ax.set_title(rf"\textbf{{{model_name.replace('_', ' ').title()}}} — Residuals vs Predicted")
    fig.tight_layout()
    _save(fig, "residual_vs_predicted", model_name)


# ── 4. Feature Importance ───────────────────────────────────────────────────

def plot_feature_importance(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    estimator=None,
    feature_names: list[str] | None = None,
    X_train=None,
    y_train=None,
    model_type: str = "",
    **kwargs,
):
    fig, ax = plt.subplots(figsize=(8, 7))
    top_n = 20

    # Try coefficient-based importance (Ridge)
    if hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_)
        method = "Absolute Coefficient"
    else:
        # Permutation importance for SVR and others
        if X_train is not None and y_train is not None:
            perm = permutation_importance(
                estimator, X_train, y_train,
                n_repeats=10,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                scoring="neg_mean_squared_error",
            )
            importance = perm.importances_mean
            method = "Permutation Importance"
        else:
            plt.close(fig)
            return

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance))]

    # Get top N
    idx = np.argsort(importance)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals = importance[idx]

    # Escape underscores for LaTeX
    names_tex = [n.replace("_", r"\_") for n in names]

    ax.barh(range(len(idx)), vals, color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(names_tex, fontsize=8)
    ax.set_xlabel(method)
    ax.set_title(
        rf"\textbf{{{model_name.replace('_', ' ').title()}}} — Top {top_n} Features ({method})"
    )
    fig.tight_layout()
    _save(fig, "feature_importance", model_name)


# ── 5. Learning Curves ──────────────────────────────────────────────────────

def plot_learning_curves(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    estimator=None,
    X_train=None,
    y_train=None,
    **kwargs,
):
    if estimator is None or X_train is None or y_train is None:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    train_rmse = np.sqrt(-train_scores)
    val_rmse = np.sqrt(-val_scores)

    train_mean = train_rmse.mean(axis=1) / 1e9
    train_std = train_rmse.std(axis=1) / 1e9
    val_mean = val_rmse.mean(axis=1) / 1e9
    val_std = val_rmse.std(axis=1) / 1e9

    ax.plot(train_sizes, train_mean, "o-", label="Train RMSE", markersize=4)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)

    ax.plot(train_sizes, val_mean, "s-", label="Validation RMSE", markersize=4)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)

    ax.set_xlabel(r"Training Set Size")
    ax.set_ylabel(r"RMSE (billion VND)")
    ax.set_title(rf"\textbf{{{model_name.replace('_', ' ').title()}}} — Learning Curves")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "learning_curves", model_name)


# ── Convenience: generate all plots ─────────────────────────────────────────

def generate_all_plots(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    estimator=None,
    feature_names: list[str] | None = None,
    X_train=None,
    y_train=None,
    model_type: str = "",
    output_dir: Path | None = None,
):
    """Generate all 5 evaluation plots for a model."""
    print(f"\nGenerating plots for {model_name}...")

    common = dict(
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        estimator=estimator,
        feature_names=feature_names,
        X_train=X_train,
        y_train=y_train,
        model_type=model_type,
    )

    plot_actual_vs_predicted(**common)
    plot_residual_distribution(**common)
    plot_residual_vs_predicted(**common)
    plot_feature_importance(**common)
    plot_learning_curves(**common)

    print(f"All plots for {model_name} complete.")
