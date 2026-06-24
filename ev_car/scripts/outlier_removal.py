import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys

sys.stdout.reconfigure(encoding='utf-8')

def remove_outliers_and_transform(df, is_oto=True):
    initial_len = len(df)
    
    # 1. Year > 2010
    df = df[df['Năm sản xuất'] >= 2010]
    
    # 2. Mileage <= 500,000
    df = df[df['Số Km đã đi'] <= 500000]
    
    # 3. Price Capping
    if is_oto:
        df = df[df['Giá_VND'] <= 20e9] # 20 Billion VND
    else:
        df = df[df['Giá_VND'] <= 1e8]  # 100 Million VND
        
    final_len = len(df)
    logger.info(f"Dropped {initial_len - final_len} outlier rows.")
    
    # 4. Log Transformation
    df['Log_Giá_VND'] = np.log1p(df['Giá_VND'])
    df['Log_Số_Km'] = np.log1p(df['Số Km đã đi'])
    
    return df

def main():
    ROOT = Path(__file__).parent.parent
    data_dir = ROOT / "data" / "processed"
    
    # Oto
    oto_path = data_dir / "ev_final_oto.csv"
    if oto_path.exists():
        logger.info("Processing Oto...")
        df_oto = pd.read_csv(oto_path)
        df_oto_ready = remove_outliers_and_transform(df_oto, is_oto=True)
        out_oto = data_dir / "ev_model_ready_oto.csv"
        df_oto_ready.to_csv(out_oto, index=False, encoding='utf-8-sig')
        logger.info(f"Saved {out_oto}")
        
    # Bike
    bike_path = data_dir / "ev_final_bike.csv"
    if bike_path.exists():
        logger.info("Processing Bike...")
        df_bike = pd.read_csv(bike_path)
        df_bike_ready = remove_outliers_and_transform(df_bike, is_oto=False)
        out_bike = data_dir / "ev_model_ready_bike.csv"
        df_bike_ready.to_csv(out_bike, index=False, encoding='utf-8-sig')
        logger.info(f"Saved {out_bike}")

if __name__ == "__main__":
    main()
