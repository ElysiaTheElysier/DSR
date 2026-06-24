import pandas as pd
from pathlib import Path
from loguru import logger
import sys

sys.stdout.reconfigure(encoding='utf-8')

def normalize_types(df):
    # Categorical -> string (char)
    cat_cols = ['Website', 'Link', 'Tên xe', 'Ngày đăng', 'Tên người bán', 'Địa chỉ', 
                'Tình trạng', 'Kiểu dáng', 'Màu ngoại thất', 'Mô tả', 'vehicle_type']
    
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    # Numerical -> int
    num_cols = ['Năm sản xuất', 'Số Km đã đi', 'Giá_VND']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
            
    # Đặc biệt Số chỗ ngồi đang là chuỗi "5 chỗ", cần tách số ra và ép thành int
    if 'Số chỗ ngồi' in df.columns:
        df['Số chỗ ngồi'] = df['Số chỗ ngồi'].astype(str).str.extract(r'(\d+)').fillna(5).astype(int)
        
    return df

def main():
    ROOT = Path(__file__).parent.parent
    data_dir = ROOT / "data" / "processed"
    
    # Oto
    oto_path = data_dir / "ev_model_ready_oto.csv"
    if oto_path.exists():
        logger.info("Normalizing types for Oto...")
        df_oto = pd.read_csv(oto_path)
        df_oto = normalize_types(df_oto)
        df_oto.to_csv(oto_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved {oto_path}")
        
    # Bike
    bike_path = data_dir / "ev_model_ready_bike.csv"
    if bike_path.exists():
        logger.info("Normalizing types for Bike...")
        df_bike = pd.read_csv(bike_path)
        df_bike = normalize_types(df_bike)
        df_bike.to_csv(bike_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved {bike_path}")

if __name__ == "__main__":
    main()
