"""
Unified plot style for the EV car price prediction project.

All plot-generating code (notebooks, model plotting, benchmarks) should import
from this module to ensure a consistent visual style across all figures.
"""

import re

import matplotlib.pyplot as plt
import seaborn as sns

# ── Color palette (colorblind-friendly, scientific) ─────────────────────────
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # blue, orange, green, red

COLOR_PRIMARY = PALETTE[0]  # single-series bars, scatter, histograms
COLOR_ACCENT = PALETTE[1]  # secondary emphasis
COLOR_GOOD = PALETTE[2]  # "after", positive, improvement
COLOR_BAD = PALETTE[3]  # "before", negative, error

TABLE_HEADER = "#4C72B0"
TABLE_ROW_ALT = "#D6DCE4"

# ── Unified rcParams ────────────────────────────────────────────────────────
RCPARAMS: dict = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
}

# ── Standard figure sizes ───────────────────────────────────────────────────
FIGSIZE_SINGLE = (7, 5)  # single-column paper figure
FIGSIZE_WIDE = (14, 5)  # full-width paper figure (figure* in LaTeX)
FIGSIZE_TALL = (7, 7)  # tall single-column (e.g. feature importance)
FIGSIZE_TABLE = (8, 2)  # table-style figure


def apply_style() -> None:
    """Apply unified matplotlib + seaborn style."""
    plt.rcParams.update(RCPARAMS)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    # Re-apply after seaborn overrides
    plt.rcParams.update(RCPARAMS)


def latex_safe(s: str) -> str:
    """Escape characters that are special in LaTeX."""
    return re.sub(r"([_&%$#{}])", r"\\\1", str(s))


def add_bar_labels(
        ax: plt.Axes,
        bars,
        fmt: str = "{:.0f}",
        fontsize: int = 8,
        rotation: int = 0,
) -> None:
    """Add value labels on top of bar chart bars."""
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )
