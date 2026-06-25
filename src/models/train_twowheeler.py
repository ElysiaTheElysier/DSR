"""
Model training and benchmarking for Two-Wheelers (Bicycles & Motorbikes).
Trains Linear Regression, SVR, Random Forest, XGBoost, and LightGBM.
Saves metrics and evaluation charts to reports/two_wheelers/.
"""

import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
# Set Agg backend only if not running inside an interactive Jupyter notebook
if 'ipykernel' not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Reconfigure stdout to support UTF-8 characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def get_root():
    current = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path("..").resolve()

ROOT = get_root()
PROCESSED = ROOT / "data" / "processed" / "two_wheelers"
REPORTS = ROOT / "reports" / "two_wheelers"

# Styling matching the repository's matplotlib styles
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

COLOR_PRIMARY = "#2E86C1"
COLOR_ACCENT = "#E67E22"
COLOR_MUTED = "#7F8C8D"

def compute_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_and_evaluate():
    logger = print  # simple log function for self-containment
    logger("=== STEP 3: HUẤN LUYỆN & SO SÁNH MÔ HÌNH XE HAI BÁNH ===")
    
    # Load processed data
    X_train = pd.read_csv(PROCESSED / "X_train.csv")
    X_test = pd.read_csv(PROCESSED / "X_test.csv")
    X_train_scaled = pd.read_csv(PROCESSED / "X_train_scaled.csv")
    X_test_scaled = pd.read_csv(PROCESSED / "X_test_scaled.csv")
    
    y_train = pd.read_csv(PROCESSED / "y_train.csv")["price_vnd"].values
    y_test = pd.read_csv(PROCESSED / "y_test.csv")["price_vnd"].values
    
    logger(f"Train features shape: {X_train_scaled.shape}")
    logger(f"Test features shape: {X_test_scaled.shape}")
    
    # Models to train
    models = {
        "Linear Regression": (LinearRegression(), True),  # uses scaled data
        "SVR": (SVR(C=1.0, epsilon=0.2), True),          # uses scaled data
        "Random Forest": (RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_leaf=2, max_features=0.8, random_state=42, n_jobs=-1
        ), False), # unscaled
        "XGBoost": (XGBRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42, n_jobs=-1
        ), False), # unscaled
        "LightGBM": (LGBMRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.08, num_leaves=31, min_child_samples=15,
            subsample=0.7, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1
        ), False) # unscaled
    }
    
    results = []
    trained_models = {}
    
    for name, (model, use_scaled) in models.items():
        logger(f"Huấn luyện {name}...")
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test
        
        # Log-transform target for training to stabilize variance
        y_tr_log = np.log1p(y_train)
        
        model.fit(X_tr, y_tr_log)
        trained_models[name] = (model, use_scaled)
        
        # Predict (and inverse-transform log price to raw VND)
        y_tr_pred = np.expm1(model.predict(X_tr))
        y_te_pred = np.expm1(model.predict(X_te))
        
        # Calculate metrics
        r2_tr = r2_score(y_train, y_tr_pred)
        r2_te = r2_score(y_test, y_te_pred)
        
        mae_tr = mean_absolute_error(y_train, y_tr_pred)
        mae_te = mean_absolute_error(y_test, y_te_pred)
        
        rmse_tr = np.sqrt(mean_squared_error(y_train, y_tr_pred))
        rmse_te = np.sqrt(mean_squared_error(y_test, y_te_pred))
        
        mape_tr = compute_mape(y_train, y_tr_pred)
        mape_te = compute_mape(y_test, y_te_pred)
        
        results.append({
            "Model": name,
            "Train R2": r2_tr,
            "Test R2": r2_te,
            "Train MAE (VND)": mae_tr,
            "Test MAE (VND)": mae_te,
            "Train RMSE (VND)": rmse_tr,
            "Test RMSE (VND)": rmse_te,
            "Train MAPE (%)": mape_tr,
            "Test MAPE (%)": mape_te
        })
        
    metrics_df = pd.DataFrame(results)
    REPORTS.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(REPORTS / "model_metrics.csv", index=False)
    
    logger("\nBảng so sánh kết quả các mô hình:")
    logger(metrics_df.to_string(index=False))
    
    # Save best model plots
    # Choose best model based on Test R2
    best_row = metrics_df.sort_values(by="Test R2", ascending=False).iloc[0]
    best_name = best_row["Model"]
    logger(f"\nMô hình tốt nhất: {best_name} với Test R2 = {best_row['Test R2']:.4f}")
    
    best_model, use_scaled = trained_models[best_name]
    X_te = X_test_scaled if use_scaled else X_test
    best_pred = np.expm1(best_model.predict(X_te))
    
    # Plot 1: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(6, 5))
    yt = y_test / 1e6  # in Million VND
    yp = best_pred / 1e6
    ax.scatter(yt, yp, alpha=0.4, s=15, color=COLOR_PRIMARY, edgecolors="none")
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Ideal")
    ax.set_xlabel("Actual Price (Million VND)")
    ax.set_ylabel("Predicted Price (Million VND)")
    ax.set_title(f"Actual vs Predicted ({best_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "actual_vs_predicted.pdf")
    plt.close(fig)
    
    # Plot 2: Residuals Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    residuals = (y_test - best_pred) / 1e6  # in Million VND
    ax.hist(residuals, bins=40, color=COLOR_ACCENT, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (Million VND)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution ({best_name})")
    fig.tight_layout()
    fig.savefig(REPORTS / "residual_distribution.pdf")
    plt.close(fig)
    
    # Plot 3: Feature Importance (for Random Forest / XGBoost / LightGBM)
    tree_models = ["Random Forest", "XGBoost", "LightGBM"]
    avail_tree = [m for m in tree_models if m in trained_models]
    if avail_tree:
        # Use Random Forest or the best available tree model
        tree_name = best_name if best_name in tree_models else avail_tree[0]
        tree_model, _ = trained_models[tree_name]
        
        # Get importances
        importances = tree_model.feature_importances_
        feature_names = X_train.columns
        
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        imp_df = imp_df.sort_values(by="Importance", ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(data=imp_df, x="Importance", y="Feature", palette="Blues_d", ax=ax)
        ax.set_title(f"Top 15 Feature Importances ({tree_name})")
        fig.tight_layout()
        fig.savefig(REPORTS / "feature_importance.pdf")
        plt.close(fig)
        logger("Các biểu đồ đánh giá đã được lưu vào thư mục reports/two_wheelers/")
        
if __name__ == "__main__":
    train_and_evaluate()
