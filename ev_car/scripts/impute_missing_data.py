import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys

# Ensure utf-8 output
sys.stdout.reconfigure(encoding='utf-8')

def impute_oto(df):
    df = df.copy()
    
    # 1. Kiểu dáng and Số chỗ ngồi (Group by Tên xe)
    # Use mode for categorical/discrete
    mode_body = df.groupby('Tên xe')['Kiểu dáng'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    mode_seats = df.groupby('Tên xe')['Số chỗ ngồi'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    
    df['Kiểu dáng'] = df['Kiểu dáng'].fillna(df['Tên xe'].map(mode_body))
    df['Số chỗ ngồi'] = df['Số chỗ ngồi'].fillna(df['Tên xe'].map(mode_seats))
    
    # If still missing, fill with overall mode/median
    df['Kiểu dáng'] = df['Kiểu dáng'].fillna('Khác')
    df['Số chỗ ngồi'] = df['Số chỗ ngồi'].fillna(5.0) # Median seats
    
    # 2. Số Km đã đi
    # Condition: Mới 100% -> 0
    df.loc[(df['Tình trạng'] == 'Mới 100%') & df['Số Km đã đi'].isna(), 'Số Km đã đi'] = 0
    # Condition: Đã qua sử dụng -> median of used cars
    median_used_km = df.loc[df['Tình trạng'] == 'Đã qua sử dụng', 'Số Km đã đi'].median()
    df['Số Km đã đi'] = df['Số Km đã đi'].fillna(median_used_km)
    
    # 3. Địa chỉ and Màu ngoại thất
    df['Địa chỉ'] = df['Địa chỉ'].fillna('Không rõ')
    df['Màu ngoại thất'] = df['Màu ngoại thất'].fillna('Không rõ')
    
    # 4. Năm sản xuất
    median_year = df.groupby('Tên xe')['Năm sản xuất'].apply(lambda x: x.median())
    df['Năm sản xuất'] = df['Năm sản xuất'].fillna(df['Tên xe'].map(median_year))
    df['Năm sản xuất'] = df['Năm sản xuất'].fillna(df['Năm sản xuất'].median()) # Fallback
    
    # 5. Tên người bán
    if 'Tên người bán' in df.columns:
        df['Tên người bán'] = df['Tên người bán'].fillna('Ẩn danh')
        
    return df

def impute_bike(df):
    df = df.copy()
    
    # 1. Năm sản xuất
    if 'Năm sản xuất' in df.columns:
        median_year = df.groupby('Tên xe')['Năm sản xuất'].apply(lambda x: x.median())
        df['Năm sản xuất'] = df['Năm sản xuất'].fillna(df['Tên xe'].map(median_year))
        df['Năm sản xuất'] = df['Năm sản xuất'].fillna(df['Năm sản xuất'].median())
    
    # 2. Số Km đã đi
    if 'Số Km đã đi' in df.columns:
        df.loc[(df['Tình trạng'] == 'Mới') & df['Số Km đã đi'].isna(), 'Số Km đã đi'] = 0
        median_used_km = df.loc[df['Tình trạng'] == 'Đã sử dụng', 'Số Km đã đi'].median()
        df['Số Km đã đi'] = df['Số Km đã đi'].fillna(median_used_km)
        
    # Other potential missing
    if 'Tên người bán' in df.columns:
        df['Tên người bán'] = df['Tên người bán'].fillna('Ẩn danh')
    if 'Địa chỉ' in df.columns:
        df['Địa chỉ'] = df['Địa chỉ'].fillna('Không rõ')
        
    return df

def main():
    ROOT = Path(__file__).parent.parent
    interim_dir = ROOT / "data" / "interim"
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Oto
    oto_path = interim_dir / "ev_cleaned_v2_oto.csv"
    if oto_path.exists():
        df_oto = pd.read_csv(oto_path)
        df_oto_imp = impute_oto(df_oto)
        out_oto = processed_dir / "ev_final_oto.csv"
        df_oto_imp.to_csv(out_oto, index=False, encoding='utf-8-sig')
        logger.info(f"Oto missing after imputation: \n{df_oto_imp.isnull().sum()}")
        logger.info(f"Saved {out_oto}")
        
    # Bike
    bike_path = interim_dir / "ev_cleaned_v2_bike.csv"
    if bike_path.exists():
        df_bike = pd.read_csv(bike_path)
        df_bike_imp = impute_bike(df_bike)
        out_bike = processed_dir / "ev_final_bike.csv"
        df_bike_imp.to_csv(out_bike, index=False, encoding='utf-8-sig')
        logger.info(f"Bike missing after imputation: \n{df_bike_imp.isnull().sum()}")
        logger.info(f"Saved {out_bike}")

if __name__ == "__main__":
    main()
