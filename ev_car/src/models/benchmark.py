"""
Benchmark — collect and compare results across all trained models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import BENCHMARK_DIR

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


def collect_all_results(model_names: list[str] | None = None) -> pd.DataFrame:
    """
    Read all *_metrics.csv files from the benchmark directory.
    Optionally filter by model_names list.
    """
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(BENCHMARK_DIR.glob("*_metrics.csv"))

    if not files:
        print("No metrics files found in", BENCHMARK_DIR)
        return pd.DataFrame()

    frames = []
    for f in files:
        name = f.stem.replace("_metrics", "")
        if model_names and name not in model_names:
            continue
        df = pd.read_csv(f)
        df.insert(0, "model", name)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Print and save a formatted comparison table."""
    if df.empty:
        print("No results to compare.")
        return df

    cols = ["model", "rmse", "mae", "r2", "adj_r2", "mape",
            "rmse_ci_lower", "rmse_ci_upper", "train_time_s"]
    display_cols = [c for c in cols if c in df.columns]
    table = df[display_cols].copy()

    # Format numbers for display
    print("\n" + "=" * 90)
    print("  MODEL BENCHMARK -- Summary")
    print("=" * 90)
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print("=" * 90)

    # Save
    path = BENCHMARK_DIR / "benchmark_summary.csv"
    table.to_csv(path, index=False)
    print(f"Summary saved -> {path}")

    return table


def comparison_bar_chart(
    df: pd.DataFrame,
    output_dir: Path | None = None,
):
    """
    Grouped bar chart comparing RMSE, MAE, and R2 across models,
    with 95 % CI error bars on RMSE.
    """
    if df.empty:
        return

    out = output_dir or BENCHMARK_DIR
    out.mkdir(parents=True, exist_ok=True)
    models = df["model"].tolist()

    # ── RMSE + MAE comparison ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # RMSE with CI error bars
    rmse_vals = df["rmse"].values / 1e9
    ci_low = df["rmse_ci_lower"].values / 1e9
    ci_high = df["rmse_ci_upper"].values / 1e9
    err_low = rmse_vals - ci_low
    err_high = ci_high - rmse_vals

    model_labels = [m.replace("_", r"\_") for m in models]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    axes[0].bar(
        range(len(models)), rmse_vals,
        yerr=[err_low, err_high],
        capsize=5,
        color=colors[: len(models)],
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(model_labels, rotation=15, ha="right")
    axes[0].set_ylabel(r"RMSE (billion VND)")
    axes[0].set_title(r"\textbf{RMSE} (with 95\% CI)")

    # MAE
    mae_vals = df["mae"].values / 1e9
    axes[1].bar(
        range(len(models)), mae_vals,
        color=colors[: len(models)],
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(model_labels, rotation=15, ha="right")
    axes[1].set_ylabel(r"MAE (billion VND)")
    axes[1].set_title(r"\textbf{MAE}")

    # R2
    r2_vals = df["r2"].values
    axes[2].bar(
        range(len(models)), r2_vals,
        color=colors[: len(models)],
        edgecolor="black",
        linewidth=0.5,
    )
    axes[2].set_xticks(range(len(models)))
    axes[2].set_xticklabels(model_labels, rotation=15, ha="right")
    axes[2].set_ylabel(r"$R^2$")
    axes[2].set_title(r"\textbf{$R^2$ Score}")
    axes[2].set_ylim(0, 1)

    fig.suptitle(r"\textbf{Model Comparison}", fontsize=14, y=1.02)
    fig.tight_layout()

    path = out / "model_comparison.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison chart saved -> {path}")


def run_benchmark(model_names: list[str] | None = None):
    """Full benchmark pipeline: collect → table → chart."""
    df = collect_all_results(model_names)
    if df.empty:
        print("No results found. Train some models first.")
        return
    summary_table(df)
    comparison_bar_chart(df)
