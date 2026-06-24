import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

def merge_data(original_file: Path, extracted_file: Path, output_file: Path):
    if not original_file.exists() or not extracted_file.exists():
        logger.warning(f"Files not found for merging: {original_file.name} or {extracted_file.name}")
        return

    logger.info(f"Merging {original_file.name} with {extracted_file.name}...")
    
    df_orig = pd.read_csv(original_file)
    df_ext = pd.read_csv(extracted_file)
    
    if 'error' in df_ext.columns:
        df_ext = df_ext[df_ext['error'].isna()].copy()
        
    mapping = {
        'imputed_body_style': 'Kiểu dáng',
        'imputed_seats': 'Số chỗ ngồi',
        'imputed_exterior_color': 'Màu ngoại thất',
        'imputed_location': 'Địa chỉ',
        'imputed_year': 'Năm sản xuất',
        'imputed_mileage_km': 'Số Km đã đi',
        'imputed_condition': 'Tình trạng'
    }
    
    # Merge on Link
    # df_ext might have duplicates if ran multiple times, but let's assume it's unique by Link
    df_ext = df_ext.drop_duplicates(subset=['Link'])
    
    df_merged = df_orig.merge(df_ext, on='Link', how='left')
    
    updates_count = {}
    
    for imp_col, orig_col in mapping.items():
        if imp_col in df_merged.columns and orig_col in df_merged.columns:
            # Count how many we are going to fill
            mask = df_merged[orig_col].isna() & df_merged[imp_col].notna()
            updates_count[orig_col] = mask.sum()
            
            # Fill original with imputed
            df_merged[orig_col] = np.where(mask, df_merged[imp_col], df_merged[orig_col])
            
    # Drop imputed columns
    df_merged = df_merged.drop(columns=[col for col in df_ext.columns if col != 'Link'], errors='ignore')
    
    # Save
    df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    logger.info(f"Saved merged file to {output_file.name}")
    logger.info(f"Updates made: {updates_count}")

def main():
    ROOT = Path(__file__).parent.parent
    data_dir = ROOT / "data" / "interim"
    
    # Oto
    merge_data(
        original_file=data_dir / "ev_cleaned_oto.csv",
        extracted_file=data_dir / "ev_extracted_missing_oto.csv",
        output_file=data_dir / "ev_cleaned_v2_oto.csv"
    )
    
    # Bike
    merge_data(
        original_file=data_dir / "ev_cleaned_bike.csv",
        extracted_file=data_dir / "ev_extracted_missing_bike.csv",
        output_file=data_dir / "ev_cleaned_v2_bike.csv"
    )

if __name__ == "__main__":
    main()
