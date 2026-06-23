"""
Build EV specs lookup table from otodien.vn raw data.

Extracts battery_kWh, range_km, power_hp per base_model and saves
as a small lookup CSV that can be joined to all records.
"""

import re

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_OTODIEN = ROOT / "data" / "raw" / "data_xe_dien_web_otodien.csv"
OUTPUT = ROOT / "data" / "interim" / "ev_specs_lookup.csv"


def extract_base_model(name: str) -> str:
    """Extract base_model from otodien car name like 'VinFast VF8 2023 Eco'."""
    if pd.isna(name):
        return ""
    # Remove brand prefix
    name = re.sub(
        r"(?i)^(vinfast|byd|wuling|porsche|bmw|mercedes[- ]benz|audi|volvo|ford|mg|hyundai|kia|geely|bestune|hongqi|lexus|tesla)\s*",
        "", name.strip())
    # Known base_model patterns
    models = [
        "VF9", "VF8", "VF7", "VF6", "VF5", "VF3", "VF e34",
        "Limo Green", "Herio Green",
        "Atto 3", "Dolphin", "Seal", "Han", "M6",
        "Taycan", "EQS", "EQE", "EQB", "EQA",
        "iX", "i4", "i5", "i7",
        "Mustang Mach-E", "Model 3", "Model Y", "Model S",
        "Ioniq 5", "Ioniq 6", "EV6", "EV9",
    ]
    for m in models:
        if m.lower() in name.lower():
            return m
    # Fallback: take first 2 words
    parts = name.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else name


def build_ev_specs_lookup() -> pd.DataFrame:
    """Build and save EV specs lookup table."""
    oto = pd.read_csv(RAW_OTODIEN)

    col_map = {
        "Dung lượng pin (kWh)": "battery_kwh",
        "Tầm hoạt động (km)": "range_km",
        "Công suất tốt đa(hp)": "power_hp",
    }

    # Extract base_model
    oto["base_model"] = oto["Tên"].apply(extract_base_model)

    # Parse numeric columns
    for vn_col, en_col in col_map.items():
        if vn_col in oto.columns:
            oto[en_col] = pd.to_numeric(oto[vn_col], errors="coerce")

    # Group by base_model → median specs
    spec_cols = list(col_map.values())
    lookup = (
        oto.groupby("base_model")[spec_cols]
        .median()
        .dropna(how="all")
        .reset_index()
    )

    # Remove empty base_model
    lookup = lookup[lookup["base_model"].str.len() > 0]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(OUTPUT, index=False)
    print(f"EV specs lookup saved: {OUTPUT} ({len(lookup)} models)")
    return lookup


if __name__ == "__main__":
    build_ev_specs_lookup()
