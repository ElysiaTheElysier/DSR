import pandas as pd
import numpy as np

try:
    df = pd.read_csv('../data/interim/ev_cleaned_rule_based.csv')
    
    print("=== TỔNG QUAN DỮ LIỆU SẠCH (Rule-based) ===")
    print(f"Tổng số bản ghi: {len(df)}")
    print(f"Số lượng theo phân loại xe (vehicle_type):")
    print(df['vehicle_type'].value_counts().to_string())

    print("\n=== THỐNG KÊ GIÁ TRUNG BÌNH THEO LOẠI XE (Triệu VNĐ) ===")
    if 'Giá' in df.columns:
        # Convert numeric and convert to millions
        df['Giá_M'] = pd.to_numeric(df['Giá'], errors='coerce') / 1_000_000
        print(df.groupby('vehicle_type')['Giá_M'].describe()[['count', 'mean', 'min', '50%', 'max']].to_string())

    print("\n=== TOP 5 HÃNG THEO LOẠI XE (Dựa vào Tên xe) ===")
    df['Brand_guess'] = df['Tên xe'].astype(str).apply(lambda x: x.split()[0].upper() if len(x.split()) > 0 else "UNKNOWN")
    for vtype in df['vehicle_type'].unique():
        print(f"\n[{vtype.upper()}]")
        print(df[df['vehicle_type'] == vtype]['Brand_guess'].value_counts().head(5).to_string())

    print("\n=== THỐNG KÊ NĂM SẢN XUẤT ===")
    if 'Năm sản xuất' in df.columns:
        df['Năm_num'] = pd.to_numeric(df['Năm sản xuất'], errors='coerce')
        # Filter realistic years
        valid_years = df[(df['Năm_num'] >= 2000) & (df['Năm_num'] <= 2026)]
        print(valid_years['Năm_num'].value_counts().sort_index(ascending=False).head(10).to_string())

    print("\n=== THỐNG KÊ GIÁ TRỊ THIẾU (NULL/NaN) ===")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing_data, 'Missing_Percentage(%)': missing_percent})
    # Sort by missing percentage descending
    missing_df = missing_df.sort_values(by='Missing_Percentage(%)', ascending=False)
    print(missing_df.to_string())

except Exception as e:
    print(f"Error: {e}")
