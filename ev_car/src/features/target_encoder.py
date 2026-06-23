"""
Leave-one-out target encoder with smoothing.

Fit on training data only, then transform train + test.
Uses smoothing to regularize rare categories:
    encoded = (count * group_mean + m * global_mean) / (count + m)
"""

import numpy as np
import pandas as pd


class TargetEncoder:
    """Leave-one-out target encoder with Bayesian smoothing."""

    def __init__(self, smoothing: int = 10):
        self.smoothing = smoothing
        self.global_mean_: float = 0.0
        self.encoding_map_: dict[str, float] = {}

    def fit(self, series: pd.Series, target: pd.Series) -> "TargetEncoder":
        """Fit encoder on training data."""
        self.global_mean_ = target.mean()
        df = pd.DataFrame({"cat": series.astype(str), "target": target})
        stats = df.groupby("cat")["target"].agg(["mean", "count"])
        m = self.smoothing
        stats["smoothed"] = (stats["count"] * stats["mean"] + m * self.global_mean_) / (stats["count"] + m)
        self.encoding_map_ = stats["smoothed"].to_dict()
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        """Transform using fitted encoding. Unseen categories get global mean."""
        return series.astype(str).map(self.encoding_map_).fillna(self.global_mean_)

    def fit_transform_loo(
            self, series: pd.Series, target: pd.Series
    ) -> pd.Series:
        """
        Leave-one-out transform for training data.
        Each row's encoding excludes itself to prevent leakage.
        """
        self.fit(series, target)
        df = pd.DataFrame({"cat": series.astype(str), "target": target})
        group_sum = df.groupby("cat")["target"].transform("sum")
        group_count = df.groupby("cat")["target"].transform("count")

        # LOO: exclude current row from group stats
        loo_mean = (group_sum - target) / (group_count - 1)
        # For groups with count=1, fall back to global mean
        loo_mean = loo_mean.fillna(self.global_mean_)

        # Apply smoothing
        m = self.smoothing
        loo_count = group_count - 1
        smoothed = (loo_count * loo_mean + m * self.global_mean_) / (
                loo_count + m
        )
        return smoothed
