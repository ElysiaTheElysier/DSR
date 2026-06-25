"""
Two-Wheelers (Bicycles & Motorbikes) Data Pipeline.
Handles raw merging, cleaning, and final alignment with LLM extraction.
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw" / "chotot"

CLEANED_PATH = INTERIM / "two_wheelers_cleaned.csv"
EXTRACTED_PATH = INTERIM / "two_wheelers_extracted.csv"
OUTPUT_PATH = INTERIM / "two_wheelers_eda_ready.csv"

def parse_city(val):
    if pd.isna(val):
        return "Other"
    val_str = str(val).strip()
    cities = [
        "Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ", 
        "Đồng Nai", "Bình Dương", "Lâm Đồng", "Quảng Ninh", "Khánh Hòa", 
        "Ninh Bình", "Vũng Tàu", "Bình Thuận", "Long An", "Tây Ninh", 
        "Bình Phước", "Quảng Nam", "Huế", "Tiền Giang", "Bến Tre",
        "Vĩnh Long", "Cà Mau", "Kiên Giang", "An Giang", "Sóc Trăng"
    ]
    for city in cities:
        if city.lower() in val_str.lower():
            return city
    # If not in our predefined list, take the last part (often province/city)
    parts = val_str.split(',')
    return parts[-1].strip() if parts else "Other"

def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove multiple spaces, newlines, tabs
    return re.sub(r'\s+', ' ', str(text)).strip()

def load_and_clean_raw():
    logger.info("=== STEP 1: LOAD & CLEAN RAW TWO-WHEELERS ===")
    
    bicycles_file = RAW / "bicycles.json"
    motorbikes_file = RAW / "motorbikes.json"
    
    if not bicycles_file.exists() or not motorbikes_file.exists():
        logger.error("Raw files bicycles.json and/or motorbikes.json not found!")
        return False
        
    with open(bicycles_file, 'r', encoding='utf-8-sig') as f:
        df_bike = pd.json_normalize(json.load(f))
        
    with open(motorbikes_file, 'r', encoding='utf-8-sig') as f:
        df_moto = pd.json_normalize(json.load(f))
        
    logger.info(f"Loaded {len(df_bike)} bicycles, {len(df_moto)} motorbikes")
    
    # Harmonize column names
    df_bike['sub_type'] = df_bike['ad_params.bicycletype.value'].fillna("Xe đạp điện")
    df_bike['origin'] = df_bike['ad_params.bicycleorigin.value'].fillna("Unknown")
    df_bike['year'] = np.nan
    df_bike['mileage_raw'] = np.nan
    
    df_moto['sub_type'] = df_moto['ad_params.motorbiketype.value'].fillna("Xe máy điện")
    df_moto['origin'] = "Unknown"  # Not present in raw motorbikes json
    df_moto['year'] = df_moto['ad_params.mfdate.value']
    df_moto['mileage_raw'] = df_moto['ad_params.mileage_v2.value']
    
    # Select common features
    cols = {
        'url': 'link',
        'exact_date_posted': 'post_date',
        'vehicle_type': 'vehicle_type',
        'ad.subject': 'title',
        'ad.price': 'price_raw',
        'ad.account_name': 'seller_name',
        'ad.body': 'description',
        'ad_params.address.value': 'address',
        'ad_params.condition_ad.value': 'condition_raw',
        'sub_type': 'sub_type',
        'origin': 'origin',
        'year': 'year_raw',
        'mileage_raw': 'mileage_raw'
    }
    
    df_bike_clean = df_bike[list(cols.keys())].rename(columns=cols)
    df_moto_clean = df_moto[list(cols.keys())].rename(columns=cols)
    
    # Combine
    df = pd.concat([df_bike_clean, df_moto_clean], ignore_index=True)
    df.index.name = 'id'
    df = df.reset_index()
    
    # Clean text columns
    df['title'] = df['title'].apply(clean_text)
    df['description'] = df['description'].apply(clean_text)
    
    # Parse prices
    df['price_vnd'] = pd.to_numeric(df['price_raw'], errors='coerce')
    # Filter out invalid prices (below 500k or above 200M VND)
    valid_mask = (df['price_vnd'] >= 500_000) & (df['price_vnd'] <= 200_000_000)
    logger.info(f"Filtered out {(~valid_mask).sum()} records with invalid prices")
    df = df[valid_mask].copy()
    
    # Standardize condition
    cond_map = {
        'Mới': 'New', 
        'Đã sử dụng': 'Used', 
        'Mới 100%': 'New', 
        'Đã qua sử dụng': 'Used'
    }
    df['condition'] = df['condition_raw'].map(cond_map).fillna('Used')
    
    # Parse city
    df['city'] = df['address'].apply(parse_city)
    
    # Parse numeric year and mileage
    df['year'] = pd.to_numeric(df['year_raw'], errors='coerce')
    df['mileage_km'] = pd.to_numeric(df['mileage_raw'], errors='coerce')
    
    # Deduplicate
    dedup_cols = ['title', 'price_vnd', 'condition', 'vehicle_type']
    n_before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep='first')
    logger.info(f"Deduplicated: {n_before} -> {len(df)} records (removed {n_before - len(df)} duplicates)")
    
    # Save cleaned
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False, encoding='utf-8-sig')
    logger.success(f"Saved cleaned raw listings to {CLEANED_PATH.name} ({len(df)} records)")
    return True

def merge_with_extracted():
    logger.info("=== STEP 3: MERGE CLEANED & EXTRACTED DATA ===")
    
    if not CLEANED_PATH.exists():
        logger.error(f"Cleaned file {CLEANED_PATH} does not exist!")
        return False
        
    if not EXTRACTED_PATH.exists():
        logger.error(f"Extracted file {EXTRACTED_PATH} does not exist! Run the extraction first.")
        return False
        
    df_cleaned = pd.read_csv(CLEANED_PATH)
    df_extracted = pd.read_csv(EXTRACTED_PATH)
    
    logger.info(f"Loaded {len(df_cleaned)} cleaned records, {len(df_extracted)} extracted records")
    
    # Merge on id
    merged = pd.merge(df_cleaned, df_extracted, on='id', how='inner')
    logger.info(f"Merged size: {len(merged)}")
    
    # Finalize columns
    ready = pd.DataFrame()
    ready['id'] = merged['id']
    ready['vehicle_type'] = merged['vehicle_type']
    ready['sub_type'] = merged['sub_type']
    ready['brand'] = merged['brand'].fillna("Unknown")
    
    # Clean model names to be consistent
    def clean_model(val):
        if pd.isna(val):
            return "Unknown"
        return str(val).strip()
        
    ready['model'] = merged['model'].apply(clean_model)
    
    # Impute missing year & mileage from LLM
    ready['year'] = merged['imputed_year'].fillna(merged['year']).fillna(2023) # Fallback to 2023 median
    ready['mileage_km'] = merged['imputed_mileage_km'].fillna(merged['mileage_km'])
    # For new vehicles, ODO is 0
    ready['condition'] = merged['condition']
    ready.loc[ready['condition'] == 'New', 'mileage_km'] = 0
    # For used vehicles with missing ODO, fill with median/NaN for later imputation
    
    ready['has_aftermarket_mods'] = merged['has_aftermarket_mods'].fillna(False).astype(int)
    ready['origin'] = merged['origin']
    ready['city'] = merged['city']
    ready['seller_name'] = merged['seller_name']
    ready['post_date'] = merged['post_date']
    ready['link'] = merged['link']
    ready['price_vnd'] = merged['price_vnd']
    ready['title'] = merged['title']
    ready['description'] = merged['description']
    
    # Save output
    ready.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    logger.success(f"Saved EDA-Ready two-wheelers dataset to {OUTPUT_PATH.name} ({len(ready)} records)")
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--merge-only":
        merge_with_extracted()
    else:
        load_and_clean_raw()
