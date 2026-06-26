"""
Feature engineering pipeline — single reproducible script.

Reads:  data/interim/ev_eda_ready.csv
Writes: data/processed/X_train.csv, X_test.csv, X_train_scaled.csv, X_test_scaled.csv,
        y_train.csv, y_test.csv, y_train_log.csv, y_test_log.csv,
        feature_names.csv, scaler.joblib, sample_weights_train.csv

Run: uv run python -m src.features.build_features
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .target_encoder import TargetEncoder
from .ev_specs import build_ev_specs_lookup, OUTPUT as EV_SPECS_PATH

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
INPUT_FILE = INTERIM / "ev_eda_ready.csv"

RANDOM_STATE = 42

# Vietnamese → English mappings
DRIVETRAIN_MAP = {
    "FWD - Dẫn động cầu trước": "FWD", "Cầu trước": "FWD", "Cầu Trước": "FWD",
    "AWD - 4 bánh toàn thời gian": "AWD", "2 Cầu": "AWD", "4WD - Dẫn động 4 bánh": "AWD",
    "Cầu sau": "RWD", "RFD - Dẫn động cầu sau": "RWD", "Cầu Sau": "RWD",
    "RWD - Dẫn động cầu sau": "RWD",
}

# ── Brand normalization ──────────────────────────────────────────────────────
BRAND_NORMALIZE = {
    "Vinfast": "VinFast",
    "Vinfat": "VinFast",
    "LIMO": "VinFast",
    "Limo Green": "VinFast",
    "Mercedes Benz": "Mercedes-Benz",
    "Mercedes": "Mercedes-Benz",
    "Herio": "VinFast",
    "HERIO GREEN": "VinFast",
    "Ecvan": "VinFast",
    "NERIO": "VinFast",
    "EC": "VinFast",
    "Kim Long": "Other",
    "Kim Long Motor": "Other",
    "/null": "Unknown",
    ":null,": "Unknown",
    "": "Unknown",
}

# ── Base model normalization ─────────────────────────────────────────────────
BASE_MODEL_NORMALIZE = {
    # VFe34 variants → VF e34
    "VFe34": "VF e34", "VFe34 2022": "VF e34", "VFe34 2023": "VF e34",
    "VFE34 22": "VF e34", "VFe34 23": "VF e34",
    "E34": "VF e34", "e34 AT": "VF e34", "e34 Plus": "VF e34",
    "E34 xanh": "VF e34",
    # VF spacing variants
    "VF 5": "VF5", "VF 3": "VF3", "VF 6": "VF6",
    "VF 7": "VF7", "VF 8": "VF8", "VF 9": "VF9",
    "VF 8S": "VF8",
    # MPV variants → MPV 7
    "MPV7": "MPV 7", "MVP7": "MPV 7", "MVP 7": "MPV 7",
    "VF MPV": "MPV 7", "VF MPV7": "MPV 7", "VFM PV": "MPV 7",
    "PV7": "MPV 7", "MPV7 BẢN": "MPV 7", "Limo/MPV7 7": "MPV 7",
    # Limo Green variants
    "LimoGreen": "Limo Green", "Limogreen": "Limo Green",
    "LIMO GREEM": "Limo Green", "VF Limogreen": "Limo Green",
    "Limogreen 2026": "Limo Green", "VF limo": "Limo Green",
    "Limo 7": "Limo Green",
    # Limo standalone → Limo Green (same VinFast product line)
    "Limo": "Limo Green",
    # EC Van variants
    "ECVAN": "EC Van", "ECVAN Nâng": "EC Van",
    "VAN ĐIỆN": "EC Van", "Van": "EC Van",
    # Minio Green variants
    "MinioGreen": "Minio Green", "Minio": "Minio Green",
    # Herio / Nerio variants
    "NERIO Green": "Herio Green", "Nerio": "Herio Green",
    "Herio": "Herio Green",
    # Green standalone
    "Green": "Herio Green", "Green Xanh": "Herio Green",
    "GREEN 2026": "Herio Green",
    # BYD variants
    "Atto3 2025": "Atto 3", "Atto 2": "Atto 3",
    "Bingo Base": "Bingo", "Bingo 333km": "Bingo", "Bingo 333": "Bingo",
    "Bingo 410km": "Bingo", "BINGO Max": "Bingo", "Bingo 2026": "Bingo",
    "Bongo": "Bingo",
    # Wuling
    "Mini EV": "Xiaoma", "Hongguang Mini": "Xiaoma",
    "HongGuang MiniEV": "Xiaoma", "Mini 170Km": "Xiaoma",
    "MiniEV Macaron": "Xiaoma", "MACARON": "Xiaoma", "EV 170": "Xiaoma",
    # Volvo
    "EC40 Ultra": "EC40", "EC40 Recharge": "EC40",
    # Hongqi
    "E-HS9 Premium": "E-HS9", "7X-E Premium": "E-HS9",
    # Geely
    "EX5 Max": "EX5", "EX5 Pro": "EX5", "EX5 EM": "EX5",
    "Ex5": "EX5", "EX2 MAX": "EX2",
    # Bestune
    "M9 Premium": "M9", "M9 MPV": "M9", "M9 Advance": "M9",
    "M9 Advanced": "M9",
    # Noise entries → Unknown
    ".": "Unknown", "/null": "Unknown", ":": "Unknown",
    "": "Unknown", "All model": "Unknown", "Full Option": "Unknown",
    "Ô tô": "Unknown",
}


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load EDA-ready data, apply Phase 1 cleaning."""
    df = pd.read_csv(path)
    n_start = len(df)
    print(f"Loaded {n_start:,} records from {path.name}")

    # Drop rows with missing price
    df = df.dropna(subset=["price_vnd"])
    print(f"  After dropping null prices: {len(df):,}")

    # Scale prices under 100,000 (which are in Millions, e.g. 785.0 -> 785,000,000.0)
    million_mask = df["price_vnd"] < 100_000
    n_million = million_mask.sum()
    if n_million > 0:
        df.loc[million_mask, "price_vnd"] = df.loc[million_mask, "price_vnd"] * 1_000_000
        print(f"  Scaled {n_million} prices from Millions to raw VND")

    # Fix chotot.com systematic 10x price inflation
    # chotot.com prices are consistently ~10x higher than other websites
    # (likely a unit mismatch in scraping/LLM extraction pipeline)
    if "website" in df.columns:
        chotot_mask = df["website"] == "chotot.com"
        n_chotot = chotot_mask.sum()
        df.loc[chotot_mask, "price_vnd"] = df.loc[chotot_mask, "price_vnd"] / 10
        print(f"  Fixed chotot.com prices (/10): {n_chotot:,} records")

    # Filter out price outliers (< 150M VND or > 3B VND)
    n_before_filter = len(df)
    df = df[(df["price_vnd"] >= 150_000_000) & (df["price_vnd"] <= 3_000_000_000)]
    print(f"  Filtered out {n_before_filter - len(df):,} price outliers (price < 150M or > 3B). Remaining: {len(df):,}")

    # ── Normalize brand names BEFORE dedup ────────────────────────────────
    df["brand"] = df["brand"].fillna("").astype(str).str.strip()
    df["brand"] = df["brand"].replace(BRAND_NORMALIZE)
    print(f"  Brand normalization applied ({len(BRAND_NORMALIZE)} mappings)")

    # ── Normalize base_model names BEFORE dedup ───────────────────────────
    df["base_model"] = df["base_model"].fillna("").astype(str).str.strip()
    df["base_model"] = df["base_model"].replace(BASE_MODEL_NORMALIZE)
    print(f"  Base model normalization applied ({len(BASE_MODEL_NORMALIZE)} mappings)")

    # Phase 1a: Deduplicate
    dedup_cols = ["brand", "base_model", "model_mode", "year", "price_vnd", "condition", "mileage_km"]
    n_before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    print(f"  After dedup: {len(df):,} (removed {n_before - len(df):,} duplicates)")

    # Phase 1c: Remove BYD (systematic pricing error)
    n_byd = (df["brand"] == "BYD").sum()
    df = df[df["brand"] != "BYD"]
    print(f"  After removing BYD: {len(df):,} (removed {n_byd} BYD records)")

    # Drop leakage/non-feature columns
    drop_cols = ["id", "seller_name", "post_date", "website"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values."""
    # Base model: fill NaN with "Unknown" to prevent stratification errors
    df["base_model"] = df["base_model"].replace({"": "Unknown"}).fillna("Unknown")

    # Brand: lookup from base_model
    if df["brand"].isna().any() or (df["brand"] == "Unknown").any():
        brand_lookup = (
            df[~df["brand"].isin(["Unknown", ""])]
            .groupby("base_model")["brand"]
            .first()
            .to_dict()
        )
        unknown_mask = df["brand"].isin(["Unknown", ""]) | df["brand"].isna()
        df.loc[unknown_mask, "brand"] = df.loc[unknown_mask, "base_model"].map(brand_lookup)
        df["brand"] = df["brand"].fillna("Other")

    # Year: median per base_model
    if df["year"].isna().any():
        year_median = df.groupby("base_model")["year"].transform("median")
        df["year"] = df["year"].fillna(year_median).fillna(df["year"].median())

    # Condition: smart rule
    if df["condition"].isna().any():
        mask_new = (df["mileage_km"].fillna(0) == 0) & (df["year"].fillna(0) >= 2025)
        df.loc[df["condition"].isna() & mask_new, "condition"] = "New"
        df.loc[df["condition"].isna(), "condition"] = "Used"

    # Mileage: 0 for new, median per brand for used
    if df["mileage_km"].isna().any():
        df.loc[(df["mileage_km"].isna()) & (df["condition"] == "New"), "mileage_km"] = 0
        mileage_median = df[df["condition"] == "Used"].groupby("brand")["mileage_km"].transform("median")
        used_mask = (df["mileage_km"].isna()) & (df["condition"] == "Used")
        df.loc[used_mask, "mileage_km"] = mileage_median
        df["mileage_km"] = df["mileage_km"].fillna(df["mileage_km"].median())

    # Cap mileage at 300k
    df.loc[df["mileage_km"] > 300_000, "mileage_km"] = np.nan
    df["mileage_km"] = df["mileage_km"].fillna(df["mileage_km"].median())

    # Seats: impute missing with median per base_model, then global median
    if "seats" in df.columns:
        if df["seats"].isna().any():
            seats_median = df.groupby("base_model")["seats"].transform("median")
            df["seats"] = df["seats"].fillna(seats_median).fillna(df["seats"].median())
        # Clip unrealistic values
        df["seats"] = df["seats"].clip(2, 9)

    # Exterior color: Vietnamese → English, then GROUP into 4 categories
    COLOR_MAP = {
        "Trắng": "White", "Đỏ": "Red", "Đen": "Black", "Xám": "Gray",
        "Xanh": "Blue", "Xanh lá": "Green", "Xanh dương": "Blue",
        "Vàng": "Yellow", "Bạc": "Silver", "Nâu": "Brown", "Hồng": "Pink",
        "Cam": "Orange", "Tím": "Purple", "Be": "Beige",
    }
    df["exterior_color"] = df["exterior_color"].map(COLOR_MAP).fillna("Other")

    # Group colors into 4 categories to reduce sparsity (was 11 one-hot columns)
    COLOR_GROUP = {
        "White": "Neutral", "Silver": "Neutral", "Beige": "Neutral",
        "Black": "Dark", "Gray": "Dark", "Brown": "Dark",
        "Red": "Vibrant", "Blue": "Vibrant", "Green": "Vibrant",
        "Yellow": "Vibrant", "Orange": "Vibrant", "Pink": "Vibrant",
        "Purple": "Vibrant",
    }
    df["color_group"] = df["exterior_color"].map(COLOR_GROUP).fillna("Other")

    # Drivetrain: Vietnamese → English
    df["drivetrain"] = df["drivetrain"].map(DRIVETRAIN_MAP).fillna("Other")

    # Origin: Vietnamese → English
    ORIGIN_MAP = {
        "Việt Nam": "Domestic", "Lắp ráp trong nước": "Domestic",
        "Nhập khẩu": "Imported", "Nước khác": "Imported",
        "Đang cập nhật": "Unknown",
    }
    df["origin"] = df["origin"].map(ORIGIN_MAP).fillna("Unknown")

    # City: Vietnamese → English (group to regions)
    CITY_MAP = {
        "Hà Nội": "Hanoi", "Hồ Chí Minh": "HCMC", "Đà Nẵng": "DaNang",
        "Hải Phòng": "HaiPhong", "Cần Thơ": "CanTho", "Đồng Nai": "DongNai",
    }
    df["city"] = df["city"].map(CITY_MAP).fillna("Other")

    return df


def engineer_features(
        df: pd.DataFrame,
        ev_specs: pd.DataFrame,
) -> pd.DataFrame:
    """Create derived features."""
    # ── Core numeric features ────────────────────────────────────────────
    df["car_age"] = 2026 - df["year"]
    df["is_new"] = (df["condition"] == "New").astype(int)
    df["log_mileage"] = np.log1p(df["mileage_km"])

    # ── Join EV specs lookup ─────────────────────────────────────────────
    df = df.merge(ev_specs, on="base_model", how="left")
    for col in ["battery_kwh", "range_km", "power_hp"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # ── NEW: Interaction features ────────────────────────────────────────

    # Depreciation interaction: mileage × age
    df["mileage_x_age"] = df["log_mileage"] * df["car_age"]

    # EV value ratios
    df["range_per_kwh"] = df["range_km"] / df["battery_kwh"].clip(lower=1)
    df["hp_per_kwh"] = df["power_hp"] / df["battery_kwh"].clip(lower=1)

    # Luxury brand indicator
    df["is_luxury_brand"] = df["brand"].isin(
        ["Porsche", "Mercedes-Benz", "BMW", "Audi", "Volvo", "Lexus", "Jaguar"]
    ).astype(int)

    # Battery size tier (ordinal: 0=small, 1=medium, 2=large)
    try:
        df["battery_tier"] = pd.qcut(
            df["battery_kwh"], q=3, labels=[0, 1, 2], duplicates="drop"
        ).astype(float).fillna(1).astype(int)
    except ValueError:
        # Fallback if too few unique values for qcut
        df["battery_tier"] = 1

    # ── Categorical interaction (for target encoding) ────────────────────
    df["brand_body"] = df["brand"] + "_" + df["body_type"].fillna("Other")
    df["model_year"] = df["base_model"] + "_" + df["year"].astype(int).astype(str)

    return df


def encode_features(
        train: pd.DataFrame,
        test: pd.DataFrame,
        y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode categorical features.
    - High-cardinality: target encoding (model_mode, model_year, base_model, brand_body)
    - Low-cardinality: one-hot (brand, body_type, drivetrain, origin, city, color_group)
    """
    # --- Target encode high-cardinality ---
    target_enc_cols = ["model_mode", "model_year", "base_model", "brand_body"]
    for col in target_enc_cols:
        if col not in train.columns:
            continue
        enc = TargetEncoder(smoothing=10)
        train[f"{col}_enc"] = enc.fit_transform_loo(train[col], y_train)
        test[f"{col}_enc"] = enc.transform(test[col])
        train = train.drop(columns=[col])
        test = test.drop(columns=[col])

    # --- One-hot encode low-cardinality ---
    # Reduced set: brand, body_type, drivetrain, origin, city, color_group
    # (was: brand, base_model, body_type, drivetrain, origin, city, exterior_color)
    onehot_cols = ["brand", "body_type", "drivetrain", "origin", "city", "color_group"]
    # Group rare categories first
    for col in onehot_cols:
        if col not in train.columns:
            continue
        counts = train[col].value_counts()
        rare = counts[counts < 10].index
        train[col] = train[col].where(~train[col].isin(rare), "Other")
        test[col] = test[col].where(~test[col].isin(rare), "Other")
        # Handle test-only categories
        known_cats = set(train[col].unique())
        test[col] = test[col].where(test[col].isin(known_cats), "Other")

    train = pd.get_dummies(train, columns=onehot_cols, drop_first=True, dtype=int)
    test = pd.get_dummies(test, columns=onehot_cols, drop_first=True, dtype=int)

    # Align columns (train may have cols test doesn't and vice versa)
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    for col in train_cols - test_cols:
        test[col] = 0
    for col in test_cols - train_cols:
        train[col] = 0
    # Same column order
    test = test[train.columns]

    return train, test


def main():
    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE (v2 — improved)")
    print("=" * 70)

    # --- Load and clean ---
    df = load_and_clean(INPUT_FILE)

    # --- Impute ---
    print("\n--- Imputation ---")
    df = impute_missing(df)
    print(f"  Remaining nulls: {df.isna().sum().sum()}")

    # --- Build EV specs lookup ---
    print("\n--- EV Specs Lookup ---")
    ev_specs = build_ev_specs_lookup()

    # --- Engineer features ---
    print("\n--- Feature Engineering ---")
    df = engineer_features(df, ev_specs)

    # --- Drop non-feature columns ---
    # Keep: seats (important for price)
    # Drop: condition (replaced by is_new), year (replaced by car_age),
    #        mileage_km (replaced by log_mileage), doors (low signal),
    #        exterior_color (replaced by color_group),
    #        has_aftermarket_mods (very low signal)
    drop_cols = ["condition", "year", "mileage_km", "doors",
                 "exterior_color", "has_aftermarket_mods"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # --- Train/Test split (stratified on base_model) ---
    print("\n--- Train/Test Split ---")
    target = df["price_vnd"]
    features = df.drop(columns=["price_vnd"])

    # Stratify on base_model (more meaningful than brand for 93.6% VinFast)
    strat_col = features["base_model"] if "base_model" in features.columns else None

    # For stratification, group rare base_models
    if strat_col is not None:
        strat = strat_col.copy()
        rare_mask = strat.map(strat.value_counts()) < 5
        strat[rare_mask] = "_rare_"
    else:
        strat = None

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE,
        stratify=strat,
    )

    print(f"  Train: {X_train.shape[0]:,} rows")
    print(f"  Test:  {X_test.shape[0]:,} rows")

    # --- Encode features ---
    print("\n--- Feature Encoding ---")
    X_train, X_test = encode_features(X_train, X_test, y_train)

    # Remove price_vnd if it leaked into features (it shouldn't but guard)
    for col in ["price_vnd"]:
        if col in X_train.columns:
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])

    # Ensure all columns are numeric
    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  WARNING: dropping non-numeric columns: {non_numeric}")
        X_train = X_train.drop(columns=non_numeric)
        X_test = X_test.drop(columns=non_numeric)

    # Fill any remaining NaN
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    print(f"  Features after encoding: {X_train.shape[1]}")
    print(f"  Feature names: {list(X_train.columns)}")

    # Skip feature pruning — with target encoding + one-hot,
    # all features carry signal and we have enough data
    print(f"\n--- Keeping all {X_train.shape[1]} features ---")

    # --- Scale for LR/SVR ---
    print("\n--- Scaling ---")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # --- Log target for LR/SVR ---
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # --- Sample weights (inverse brand frequency) ---
    # Use base_model_enc bins as proxy for weighting
    sw = np.ones(len(y_train))
    if "base_model_enc" in X_train.columns:
        enc_vals = X_train["base_model_enc"]
        # Group into price tiers
        bins = pd.qcut(enc_vals, q=5, labels=False, duplicates="drop")
        for b in bins.unique():
            mask = bins == b
            count = mask.sum()
            sw[mask.values] = len(y_train) / (5 * count) if count > 0 else 1.0
    sw = sw / sw.mean()  # Normalize to mean=1

    # --- Save ---
    print("\n--- Saving outputs ---")
    PROCESSED.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED / "X_test.csv", index=False)
    X_train_scaled.to_csv(PROCESSED / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(PROCESSED / "X_test_scaled.csv", index=False)
    y_train.to_csv(PROCESSED / "y_train.csv", index=False, header=["price_vnd"])
    y_test.to_csv(PROCESSED / "y_test.csv", index=False, header=["price_vnd"])
    y_train_log.to_csv(PROCESSED / "y_train_log.csv", index=False, header=["log_price_vnd"])
    y_test_log.to_csv(PROCESSED / "y_test_log.csv", index=False, header=["log_price_vnd"])
    pd.DataFrame({"feature": X_train.columns}).to_csv(PROCESSED / "feature_names.csv", index=False)
    joblib.dump(scaler, PROCESSED / "scaler.joblib")
    pd.DataFrame({"weight": sw}).to_csv(PROCESSED / "sample_weights_train.csv", index=False)

    print(f"\n  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Features: {list(X_train.columns)}")
    print(f"  y_train range: {y_train.min() / 1e9:.3f}B - {y_train.max() / 1e9:.3f}B")
    print(f"  y_test range:  {y_test.min() / 1e9:.3f}B - {y_test.max() / 1e9:.3f}B")

    print("\n" + "=" * 70)
    print("DONE — All outputs saved to data/processed/")
    print("=" * 70)


if __name__ == "__main__":
    main()
