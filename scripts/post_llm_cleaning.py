import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import yaml

def load_config():
    with open("../configs/local_llm_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def clean_data():
    config = load_config()
    model_name = config['pipeline']['default_model'].replace(':', '_')
    raw_file = Path("../data/interim/ev_cleaned_rule_based.csv")
    llm_file = Path(f"../data/interim/ev_extracted_{model_name}.csv")
    output_file = Path("../data/processed/ev_zero_nulls.csv")
    
    if not llm_file.exists():
        logger.error(f"Cannot find LLM file: {llm_file}")
        return

    logger.info(f"Loading raw data from {raw_file}")
    df_raw = pd.read_csv(raw_file, encoding='utf-8-sig')
    
    logger.info(f"Loading LLM data from {llm_file}")
    # skip bad lines in case there are still corrupted rows
    df_llm = pd.read_csv(llm_file, encoding='utf-8-sig', on_bad_lines='skip')
    
    # Merge the dataframes on index (since LLM id is the row index of raw_file)
    df_llm.rename(columns={'id': 'raw_index'}, inplace=True)
    df_raw['raw_index'] = df_raw.index
    
    df = pd.merge(df_raw, df_llm, on='raw_index', how='inner')
    
    initial_len = len(df)

    # 1. Drop columns with too many nulls that aren't critical
    cols_to_drop = ["Dẫn động", "Màu nội thất", "raw_index"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 2. Impute Year (Năm sản xuất)
    df['Năm sản xuất'] = pd.to_numeric(df['Năm sản xuất'], errors='coerce')
    df['imputed_year'] = pd.to_numeric(df['imputed_year'], errors='coerce')
    df['Năm sản xuất'] = df['Năm sản xuất'].fillna(df['imputed_year']).fillna(2024)

    # 3. Impute Condition (Tình trạng)
    df['Tình trạng'] = df['Tình trạng'].fillna(df['imputed_condition']).fillna("Đã qua sử dụng")
    
    # Standardize Condition strings
    df.loc[df['Tình trạng'].str.contains("mới 99", case=False, na=False), 'Tình trạng'] = "Đã qua sử dụng"
    df.loc[df['Tình trạng'].str.contains("dùng|sử dụng", case=False, na=False), 'Tình trạng'] = "Đã qua sử dụng"
    df.loc[df['Tình trạng'].str.contains("mới", case=False, na=False) & (df['Tình trạng'] != "Đã qua sử dụng"), 'Tình trạng'] = "Mới"

    # 4. Impute Mileage (Số Km đã đi)
    df['Số Km đã đi'] = pd.to_numeric(df['Số Km đã đi'], errors='coerce')
    df['imputed_mileage_km'] = pd.to_numeric(df['imputed_mileage_km'], errors='coerce')
    df['Số Km đã đi'] = df['Số Km đã đi'].fillna(df['imputed_mileage_km'])
    
    # Guardrails for Mileage vs Condition
    # Rule 1: If mileage > 100, it MUST be a used car
    df.loc[df['Số Km đã đi'] > 100, 'Tình trạng'] = "Đã qua sử dụng"
    
    # Rule 2: If it's a new car, mileage MUST be 0
    df.loc[df['Tình trạng'] == "Mới", 'Số Km đã đi'] = 0
    
    # Rule 3: If it's a used car but mileage is <= 100 or NaN, set to median
    median_used_mileage = df[(df['Tình trạng'] == "Đã qua sử dụng") & (df['Số Km đã đi'] > 100)]['Số Km đã đi'].median()
    if pd.isna(median_used_mileage):
        median_used_mileage = 15000
        
    df.loc[(df['Tình trạng'] == "Đã qua sử dụng") & ((df['Số Km đã đi'] <= 100) | df['Số Km đã đi'].isna()), 'Số Km đã đi'] = median_used_mileage

    # 5. Impute Doors and Seats based on vehicle_type
    # Rename vehicle_type_y from LLM to just vehicle_type if it exists, but use vehicle_type_x (raw data) as truth
    if 'vehicle_type_x' in df.columns:
        df.rename(columns={'vehicle_type_x': 'vehicle_type'}, inplace=True)
        if 'vehicle_type_y' in df.columns:
            df.drop(columns=['vehicle_type_y'], inplace=True)
            
    if 'vehicle_type' in df.columns:
        is_car = df['vehicle_type'] == 'oto_dien'
        df.loc[is_car & df['Số cửa'].isna(), 'Số cửa'] = '4'
        df.loc[is_car & df['Số chỗ ngồi'].isna(), 'Số chỗ ngồi'] = '5'
        
        is_bike = df['vehicle_type'].isin(['xe_may_dien', 'xe_dap_dien'])
        df.loc[is_bike, 'Số cửa'] = '0'
        df.loc[is_bike, 'Số chỗ ngồi'] = '2'
    
    # 6. Fill categorical defaults
    if 'Xuất xứ' in df.columns:
        df['Xuất xứ'] = df['Xuất xứ'].fillna("Việt Nam")
    if 'Màu ngoại thất' in df.columns:
        df['Màu ngoại thất'] = df['Màu ngoại thất'].fillna("Đang cập nhật")
    if 'Kiểu dáng' in df.columns:
        df.loc[df['vehicle_type'] == 'xe_may_dien', 'Kiểu dáng'] = 'Xe máy điện'
        df.loc[df['vehicle_type'] == 'xe_dap_dien', 'Kiểu dáng'] = 'Xe đạp điện'
        df['Kiểu dáng'] = df['Kiểu dáng'].fillna("Đang cập nhật")

    # 7. Final Drops
    df['Giá_VND'] = pd.to_numeric(df['Giá_VND'], errors='coerce')
    df = df.dropna(subset=['Giá_VND'])
    df = df[df['Giá_VND'] > 0]

    if 'brand' in df.columns:
        df['brand'] = df['brand'].fillna("Khác")
        
    if 'car_model' in df.columns:
        df['car_model'] = df['car_model'].fillna("Khác")
        
    if 'battery_status' in df.columns:
        df['battery_status'] = df['battery_status'].fillna("Không rõ")
        
    if 'is_accident_free' in df.columns:
        df['is_accident_free'] = df['is_accident_free'].fillna(True)
        
    if 'has_aftermarket_mods' in df.columns:
        df['has_aftermarket_mods'] = df['has_aftermarket_mods'].fillna(False)
        
    if 'Địa chỉ' in df.columns:
        df['Địa chỉ'] = df['Địa chỉ'].fillna("Đang cập nhật")

    # Clean up redundant intermediate columns
    redundant_cols = ['imputed_year', 'imputed_mileage_km', 'imputed_condition', 'reasoning', 'raw_index']
    df = df.drop(columns=[c for c in redundant_cols if c in df.columns])
    
    # 8. Deduplicate based on exact matching Descriptions (spam listings)
    if 'Mô tả' in df.columns:
        df = df.drop_duplicates(subset=['Mô tả'])

    # 9. Assert Zero Nulls
    critical_cols = [
        "Giá_VND", "Năm sản xuất", "Tình trạng", "Số Km đã đi", "Xuất xứ", 
        "Kiểu dáng", "Số chỗ ngồi", "Số cửa", "vehicle_type", "brand", 
        "car_model", "battery_status", "Địa chỉ"
    ]
    critical_cols = [c for c in critical_cols if c in df.columns]
    
    null_counts = df[critical_cols].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Still have nulls in critical columns:\n{null_counts[null_counts > 0]}")
        df = df.dropna(subset=critical_cols)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"Cleaned data saved to {output_file}")
    logger.info(f"Initial raw rows: {len(df_raw)} -> Processed by LLM: {initial_len} -> Final clean rows: {len(df)}")

if __name__ == "__main__":
    clean_data()
