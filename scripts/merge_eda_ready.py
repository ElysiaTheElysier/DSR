"""
Merge EDA-Ready Dataset Pipeline.

Combines rule-based cleaned dataset (ev_cleaned_rule_based.csv)
with LLM-extracted features (ev_extracted_gpt5nano.csv or ev_extracted_qwen2.5_3b.csv)
to construct the final analysis-ready dataset (ev_eda_ready.csv).
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

def extract_base_model(name: str) -> str:
    if pd.isna(name):
        return ""
    name = re.sub(
        r"(?i)^(vinfast|byd|wuling|porsche|bmw|mercedes[- ]benz|audi|volvo|ford|mg|hyundai|kia|geely|bestune|hongqi|lexus|tesla)\s*",
        "", name.strip())
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
    parts = name.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else name

def parse_mileage(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    if 'vạn' in val_str:
        try:
            num = float(re.sub(r'[^\d.]', '', val_str.split('vạn')[0]))
            return num * 10000
        except:
            pass
    digits = re.sub(r'[^\d]', '', val_str)
    if digits:
        return float(digits)
    return np.nan

def parse_number(val):
    if pd.isna(val):
        return np.nan
    digits = re.sub(r'[^\d]', '', str(val))
    return float(digits) if digits else np.nan

def parse_city(val):
    if pd.isna(val):
        return "Other"
    val_str = str(val).strip()
    cities = ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Đồng Nai", "Bình Dương", "Lâm Đồng", "Quảng Ninh", "Khánh Hòa", "Ninh Bình"]
    for city in cities:
        if city.lower() in val_str.lower():
            return city
    parts = val_str.split(',')
    return parts[-1].strip()

def main():
    ROOT = Path(__file__).resolve().parent.parent
    INTERIM = ROOT / "data" / "interim"
    
    CLEANED_PATH = INTERIM / "ev_cleaned_rule_based.csv"
    EXTRACTED_PATH = INTERIM / "ev_extracted_gpt5nano.csv"
    OUTPUT_PATH = INTERIM / "ev_eda_ready.csv"
    
    logger.info("=== MERGE EDA-READY PIPELINE ===")
    
    if not CLEANED_PATH.exists():
        logger.error(f"Missing rule-based cleaned file: {CLEANED_PATH}")
        return
        
    if not EXTRACTED_PATH.exists():
        logger.warning(f"LLM extracted file {EXTRACTED_PATH} not found. Trying local Qwen...")
        EXTRACTED_PATH = INTERIM / "ev_extracted_qwen2.5_3b.csv"
        if not EXTRACTED_PATH.exists():
            logger.error("No LLM extraction files found in data/interim!")
            return

    logger.info(f"Loading cleaned data from {CLEANED_PATH.name}...")
    df_cleaned = pd.read_csv(CLEANED_PATH)
    
    logger.info(f"Loading extracted attributes from {EXTRACTED_PATH.name}...")
    df_extracted = pd.read_csv(EXTRACTED_PATH)
    
    # Sync IDs
    df_cleaned.index.name = 'id'
    df_cleaned = df_cleaned.reset_index()
    
    logger.info("Merging datasets on 'id'...")
    merged = pd.merge(df_cleaned, df_extracted, on='id', how='inner')
    logger.info(f"Merged shape: {merged.shape}")
    
    # Construct final dataframe
    ready = pd.DataFrame()
    ready['id'] = merged['id']
    ready['brand'] = merged['brand']
    ready['base_model'] = merged['car_model'].apply(extract_base_model)
    ready['model_mode'] = merged['car_model']
    ready['year'] = merged['imputed_year'].fillna(merged['Năm sản xuất'])
    
    cond_map = {
        'Mới 100%': 'New',
        'Xe mới': 'New',
        'Mới': 'New',
        'Đã qua sử dụng': 'Used',
        'Xe đã dùng': 'Used',
        'Đã sử dụng': 'Used',
        'Mới 99%': 'Used'
    }
    ready['condition'] = merged['Tình trạng'].map(cond_map).fillna(merged['imputed_condition'].map(cond_map))
    
    cleaned_mileage = merged['Số Km đã đi'].apply(parse_mileage)
    ready['mileage_km'] = merged['imputed_mileage_km'].fillna(cleaned_mileage)
    ready['has_aftermarket_mods'] = merged['has_aftermarket_mods']
    
    BODY_TYPE_MAP = {
        'Coupe (2 cửa)': 'Coupe', 
        'Crossover': 'SUV', 
        'Hatbach': 'Hatchback', 
        'Hatback': 'Hatchback', 
        'Hatchback': 'Hatchback', 
        'Kiểu dáng khác': 'Other', 
        'MPV': 'MPV', 
        'Mini': 'Hatchback', 
        'Mini Car': 'Hatchback', 
        'Minivan (MPV)': 'MPV', 
        'SUV': 'SUV', 
        'SUV / Cross over': 'SUV', 
        'SUV-B': 'SUV', 
        'SUV/Crossover': 'SUV', 
        'Sedan': 'Sedan', 
        'Van': 'MPV', 
        'Van/Minivan': 'MPV', 
        'Wagon': 'Wagon'
    }
    ready['body_type'] = merged['Kiểu dáng'].map(BODY_TYPE_MAP).fillna('Other')
    
    ready['seats'] = merged['Số chỗ ngồi'].apply(parse_number)
    ready['doors'] = merged['Số cửa'].apply(parse_number)
    ready['drivetrain'] = merged['Dẫn động']
    ready['origin'] = merged['Xuất xứ']
    ready['exterior_color'] = merged['Màu ngoại thất']
    ready['city'] = merged['Địa chỉ'].apply(parse_city)
    ready['seller_name'] = merged['Tên người bán']
    ready['post_date'] = merged['Ngày đăng']
    ready['website'] = merged['Website']
    ready['price_vnd'] = merged['Giá_VND']
    
    # Save output
    ready.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    logger.success(f"Successfully saved EDA-Ready dataset to {OUTPUT_PATH} ({len(ready)} records)")

if __name__ == "__main__":
    main()