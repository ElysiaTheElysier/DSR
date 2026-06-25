"""
Feature engineering pipeline for Two-Wheelers (Bicycles & Motorbikes).
Fits all preprocessing on the training set only to prevent data leakage.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger

from src.features.target_encoder import TargetEncoder

def get_root():
    current = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path("..").resolve()

ROOT = get_root()
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed" / "two_wheelers"
INPUT_FILE = INTERIM / "two_wheelers_eda_ready.csv"

# Specifications dictionary for two-wheelers
SPECS_MAP = {
    # VinFast motorbikes
    'evo200': {'battery_kwh': 3.5, 'range_km': 203.0, 'power_hp': 3.35},
    'feliz': {'battery_kwh': 3.5, 'range_km': 160.0, 'power_hp': 4.02},
    'klara': {'battery_kwh': 3.5, 'range_km': 194.0, 'power_hp': 4.02},
    'vento': {'battery_kwh': 3.5, 'range_km': 160.0, 'power_hp': 7.00},
    'theon': {'battery_kwh': 3.5, 'range_km': 150.0, 'power_hp': 9.92},
    'impes': {'battery_kwh': 1.2, 'range_km': 70.0, 'power_hp': 1.61},
    'ludo': {'battery_kwh': 1.2, 'range_km': 70.0, 'power_hp': 1.61},
    
    # Dat Bike motorbikes
    'quantum': {'battery_kwh': 5.6, 'range_km': 270.0, 'power_hp': 9.38},
    'weaver': {'battery_kwh': 3.0, 'range_km': 200.0, 'power_hp': 8.04},
    
    # Yadea motorbikes
    'voltguard': {'battery_kwh': 2.7, 'range_km': 100.0, 'power_hp': 4.02},
    'odora': {'battery_kwh': 1.5, 'range_km': 80.0, 'power_hp': 2.01},
    'orla': {'battery_kwh': 1.5, 'range_km': 80.0, 'power_hp': 2.01},
    'buye': {'battery_kwh': 1.5, 'range_km': 80.0, 'power_hp': 2.01},
    
    # Pega motorbikes
    'aura': {'battery_kwh': 1.5, 'range_km': 80.0, 'power_hp': 2.01},
    
    # Fallback default values
    'bicycle_default': {'battery_kwh': 0.6, 'range_km': 50.0, 'power_hp': 0.47},
    'motorbike_default': {'battery_kwh': 1.8, 'range_km': 90.0, 'power_hp': 2.50}
}

def map_specifications(row):
    model_name = str(row['model']).lower()
    vehicle_type = str(row['vehicle_type'])
    
    # Check if any model key matches
    for key, spec in SPECS_MAP.items():
        if key in model_name:
            return spec['battery_kwh'], spec['range_km'], spec['power_hp']
            
    # Default fallbacks
    if vehicle_type == "xe_may_dien":
        default = SPECS_MAP['motorbike_default']
    else:
        default = SPECS_MAP['bicycle_default']
    return default['battery_kwh'], default['range_km'], default['power_hp']

def main():
    logger.info("=== STEP 2: TWO-WHEELERS FEATURE ENGINEERING ===")
    
    if not INPUT_FILE.exists():
        logger.error(f"Input file {INPUT_FILE} does not exist!")
        return
        
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"Loaded {len(df)} records from {INPUT_FILE.name}")
    
    # Drop rows without price
    df = df.dropna(subset=["price_vnd"])
    
    # Scale prices under 100,000 (which are in Millions, e.g. 24.5 -> 24,500,000.0)
    million_mask = df["price_vnd"] < 100_000
    n_million = million_mask.sum()
    if n_million > 0:
        df.loc[million_mask, "price_vnd"] = df.loc[million_mask, "price_vnd"] * 1_000_000
        logger.info(f"Scaled {n_million} prices from Millions to raw VND")

    # Compute baseline medians for models using valid prices (1M to 80M)
    valid_mask = (df["price_vnd"] >= 1_000_000) & (df["price_vnd"] < 80_000_000)
    model_medians = df[valid_mask].groupby("model")["price_vnd"].median().to_dict()
    brand_medians = df[valid_mask].groupby("brand")["price_vnd"].median().to_dict()
    overall_median = df[valid_mask]["price_vnd"].median() if valid_mask.any() else 15_000_000.0

    # Scale both low-priced outliers (< 1M) and high-priced outliers (>= 80M) using ratio-to-median
    n_scaled_low = 0
    n_scaled_high = 0

    for idx in df.index:
        p = df.loc[idx, "price_vnd"]
        model = df.loc[idx, "model"]
        brand = df.loc[idx, "brand"]
        
        ref = model_medians.get(model, brand_medians.get(brand, overall_median))
        if pd.isna(ref) or ref == 0:
            ref = overall_median
            
        ratio = p / ref
        
        # Low price outliers (between 10k and 1M VND)
        if p < 1_000_000:
            if 0.07 <= ratio <= 0.13:
                df.loc[idx, "price_vnd"] = p * 10
                n_scaled_low += 1
            elif 0.007 <= ratio <= 0.013:
                df.loc[idx, "price_vnd"] = p * 100
                n_scaled_low += 1
                
        # High price outliers (>= 80M VND)
        elif p >= 80_000_000:
            if 7.0 <= ratio <= 13.0:
                df.loc[idx, "price_vnd"] = p / 10
                n_scaled_high += 1
            elif 70.0 <= ratio <= 130.0:
                df.loc[idx, "price_vnd"] = p / 100
                n_scaled_high += 1

    if n_scaled_low > 0:
        logger.info(f"Rescaled {n_scaled_low} low-priced two-wheelers (< 1M) using ratio-to-median")
    if n_scaled_high > 0:
        logger.info(f"Rescaled {n_scaled_high} high-priced two-wheelers (>= 80M) using ratio-to-median")

    # Filter price outliers (keep only prices between 1M and 80M VND)
    initial_len = len(df)
    df = df[(df["price_vnd"] >= 1_000_000) & (df["price_vnd"] <= 80_000_000)]
    price_filtered = initial_len - len(df)
    if price_filtered > 0:
        logger.info(f"Filtered out {price_filtered} price outliers (price < 1M or > 80M). Remaining: {len(df)}")
        
    # Handle invalid years (year < 2010 is treated as NaN to be imputed by median)
    invalid_years = (df["year"] < 2010) & (df["year"].notna())
    num_invalid_years = invalid_years.sum()
    if num_invalid_years > 0:
        logger.info(f"Treating {num_invalid_years} invalid years (< 2010) as NaN")
        df.loc[invalid_years, "year"] = np.nan
        
    # Handle extreme mileages (clip/cap mileage at 100,000 km)
    extreme_mileage = df["mileage_km"] > 100_000
    num_extreme_mileage = extreme_mileage.sum()
    if num_extreme_mileage > 0:
        logger.info(f"Capping {num_extreme_mileage} extreme mileages (> 100,000 km) at 100,000 km")
        df.loc[extreme_mileage, "mileage_km"] = 100_000.0
    
    # Drop non-feature columns
    drop_cols = ["link", "post_date", "seller_name", "title", "description", "origin"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Map specs (battery, range, power) based on model name
    specs = df.apply(map_specifications, axis=1)
    df['battery_kwh'] = [s[0] for s in specs]
    df['range_km'] = [s[1] for s in specs]
    df['power_hp'] = [s[2] for s in specs]
    
    # Separate features and target
    target = df["price_vnd"]
    features = df.drop(columns=["price_vnd", "id"])
    
    # Train-test split FIRST to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    logger.info(f"Split data into Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    
    # Impute numeric NaNs using training median (prevent leakage)
    year_median = X_train['year'].median()
    if pd.isna(year_median):
        year_median = 2023
    X_train['year'] = X_train['year'].fillna(year_median)
    X_test['year'] = X_test['year'].fillna(year_median)
    
    mileage_median = X_train['mileage_km'].median()
    if pd.isna(mileage_median):
        mileage_median = 5000.0
    X_train['mileage_km'] = X_train['mileage_km'].fillna(mileage_median)
    X_test['mileage_km'] = X_test['mileage_km'].fillna(mileage_median)
    
    # Create derived features
    for dataset in [X_train, X_test]:
        dataset['car_age'] = 2026 - dataset['year']
        dataset['is_new'] = (dataset['condition'] == 'New').astype(int)
        dataset['log_mileage'] = np.log1p(dataset['mileage_km'])
        dataset.drop(columns=['year', 'condition', 'mileage_km'], inplace=True, errors='ignore')
        
    # Target encoding for high cardinality feature 'model' (fit LOO on train, transform test)
    te = TargetEncoder(smoothing=10)
    X_train['model_enc'] = te.fit_transform_loo(X_train['model'], y_train)
    X_test['model_enc'] = te.transform(X_test['model'])
    X_train.drop(columns=['model'], inplace=True)
    X_test.drop(columns=['model'], inplace=True)
    
    # One-hot encode low-cardinality features: brand, vehicle_type, city
    onehot_cols = ['brand', 'vehicle_type', 'city']
    
    # Extract training categories to align test set later
    X_train_encoded = pd.get_dummies(X_train, columns=onehot_cols, drop_first=False, dtype=int)
    X_test_encoded = pd.get_dummies(X_test, columns=onehot_cols, drop_first=False, dtype=int)
    
    # Align columns to prevent shape mismatch and leakage
    for col in set(X_train_encoded.columns) - set(X_test_encoded.columns):
        X_test_encoded[col] = 0
    for col in set(X_test_encoded.columns) - set(X_train_encoded.columns):
        X_train_encoded[col] = 0
    X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    # Ensure all columns are numeric
    non_numeric = X_train_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning(f"Dropping non-numeric columns: {non_numeric}")
        X_train_encoded.drop(columns=non_numeric, inplace=True)
        X_test_encoded.drop(columns=non_numeric, inplace=True)
        
    # Scale features (fit on train, transform test)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_encoded), 
        columns=X_train_encoded.columns, 
        index=X_train_encoded.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_encoded), 
        columns=X_test_encoded.columns, 
        index=X_test_encoded.index
    )
    
    # Save outputs
    PROCESSED.mkdir(parents=True, exist_ok=True)
    
    X_train_encoded.to_csv(PROCESSED / "X_train.csv", index=False)
    X_test_encoded.to_csv(PROCESSED / "X_test.csv", index=False)
    X_train_scaled.to_csv(PROCESSED / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(PROCESSED / "X_test_scaled.csv", index=False)
    
    y_train.to_csv(PROCESSED / "y_train.csv", index=False, header=["price_vnd"])
    y_test.to_csv(PROCESSED / "y_test.csv", index=False, header=["price_vnd"])
    
    # Log targets
    np.log1p(y_train).to_csv(PROCESSED / "y_train_log.csv", index=False, header=["log_price_vnd"])
    np.log1p(y_test).to_csv(PROCESSED / "y_test_log.csv", index=False, header=["log_price_vnd"])
    
    logger.success(f"Feature engineering completed! Train size: {X_train_scaled.shape}, Test size: {X_test_scaled.shape}")
    logger.info(f"Outputs written to {PROCESSED}")

if __name__ == "__main__":
    main()
