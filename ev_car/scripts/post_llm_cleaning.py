import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

def clean_data(prefix: str):
    logger.info(f"--- Starting post-LLM cleaning for {prefix} ---")
    raw_file = Path(f"../data/interim/ev_cleaned_{prefix}.csv")
    llm_file = Path(f"../data/interim/ev_extracted_{prefix}.csv")
    output_file = Path(f"../data/processed/ev_final_{prefix}.csv")
    
    if not raw_file.exists() or not llm_file.exists():
        logger.error(f"Cannot find required files for {prefix}. Ensure raw and llm files exist.")
        return

    logger.info(f"Loading raw data from {raw_file}")
    df_raw = pd.read_csv(raw_file, encoding='utf-8-sig')
    
    logger.info(f"Loading LLM data from {llm_file}")
    # skip bad lines in case there are corrupted rows
    df_llm = pd.read_csv(llm_file, encoding='utf-8-sig', on_bad_lines='skip')
    
    # Merge on index
    df_llm.rename(columns={'id': 'raw_index'}, inplace=True)
    df_raw['raw_index'] = df_raw.index
    
    df = pd.merge(df_raw, df_llm, on='raw_index', how='inner')
    
    initial_len = len(df)

    # 1. Drop redundant columns
    cols_to_drop = ["Dẫn động", "Màu nội thất", "raw_index"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 2. Impute Year
    df['Năm sản xuất'] = pd.to_numeric(df['Năm sản xuất'], errors='coerce')
    if 'imputed_year' in df.columns:
        df['imputed_year'] = pd.to_numeric(df['imputed_year'], errors='coerce')
        df['Năm sản xuất'] = df['Năm sản xuất'].fillna(df['imputed_year'])
    df['Năm sản xuất'] = df['Năm sản xuất'].fillna(2024)

    # 3. Impute Condition
    if 'imputed_condition' in df.columns:
        df['Tình trạng'] = df['Tình trạng'].fillna(df['imputed_condition'])
    df['Tình trạng'] = df['Tình trạng'].fillna("Đã qua sử dụng")
    
    df.loc[df['Tình trạng'].str.contains("mới 99", case=False, na=False), 'Tình trạng'] = "Đã qua sử dụng"
    df.loc[df['Tình trạng'].str.contains("dùng|sử dụng", case=False, na=False), 'Tình trạng'] = "Đã qua sử dụng"
    df.loc[df['Tình trạng'].str.contains("mới", case=False, na=False) & (df['Tình trạng'] != "Đã qua sử dụng"), 'Tình trạng'] = "Mới 100%"

    # 4. Impute Mileage
    df['Số Km đã đi'] = pd.to_numeric(df['Số Km đã đi'], errors='coerce')
    if 'imputed_mileage_km' in df.columns:
        df['imputed_mileage_km'] = pd.to_numeric(df['imputed_mileage_km'], errors='coerce')
        df['Số Km đã đi'] = df['Số Km đã đi'].fillna(df['imputed_mileage_km'])
    
    # Guardrails
    df.loc[df['Số Km đã đi'] > 100, 'Tình trạng'] = "Đã qua sử dụng"
    df.loc[df['Tình trạng'] == "Mới 100%", 'Số Km đã đi'] = 0
    
    median_used_mileage = df[(df['Tình trạng'] == "Đã qua sử dụng") & (df['Số Km đã đi'] > 100)]['Số Km đã đi'].median()
    if pd.isna(median_used_mileage):
        median_used_mileage = 15000
        
    df.loc[(df['Tình trạng'] == "Đã qua sử dụng") & ((df['Số Km đã đi'] <= 100) | df['Số Km đã đi'].isna()), 'Số Km đã đi'] = median_used_mileage

    # 5. Impute Doors and Seats
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
        if prefix == 'bike':
            df.loc[df['vehicle_type'] == 'xe_may_dien', 'Kiểu dáng'] = 'Xe máy điện'
            df.loc[df['vehicle_type'] == 'xe_dap_dien', 'Kiểu dáng'] = 'Xe đạp điện'
        df['Kiểu dáng'] = df['Kiểu dáng'].fillna("Đang cập nhật")

    # 7. Final Drops & Defaults
    df['Giá_VND'] = pd.to_numeric(df['Giá_VND'], errors='coerce')
    df = df.dropna(subset=['Giá_VND'])
    df = df[df['Giá_VND'] > 0]

    for col in ['brand', 'car_model']:
        if col in df.columns:
            df[col] = df[col].fillna("Khác")
            
    if 'battery_status' in df.columns:
        df['battery_status'] = df['battery_status'].fillna("Không rõ")
        
    if 'is_accident_free' in df.columns:
        df['is_accident_free'] = df['is_accident_free'].fillna(True)
        
    if 'has_aftermarket_mods' in df.columns:
        df['has_aftermarket_mods'] = df['has_aftermarket_mods'].fillna(False)
        
    if 'Địa chỉ' in df.columns:
        df['Địa chỉ'] = df['Địa chỉ'].fillna("Đang cập nhật")

    # Clean intermediate columns
    redundant_cols = ['imputed_year', 'imputed_mileage_km', 'imputed_condition', 'reasoning']
    df = df.drop(columns=[c for c in redundant_cols if c in df.columns])
    
    if 'Mô tả' in df.columns:
        df = df.drop_duplicates(subset=['Mô tả'])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"Cleaned data saved to {output_file}")
    logger.info(f"Initial raw rows: {len(df_raw)} -> Processed by LLM: {initial_len} -> Final clean rows: {len(df)}\n")

if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    clean_data("oto")
    clean_data("bike")
