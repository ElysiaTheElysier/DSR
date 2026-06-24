import pandas as pd
import numpy as np
import re
from pathlib import Path
from loguru import logger
import sys

sys.stdout.reconfigure(encoding='utf-8')

def extract_battery_status(description):
    if not isinstance(description, str):
        return 'Không rõ'
        
    desc_lower = description.lower()
    
    # Keywords for Bao pin / Mua đứt (Battery Included)
    bao_pin_keywords = ['bao pin', 'mua đứt', 'mua pin', 'đã mua pin', 'kèm pin', 'có pin']
    if any(k in desc_lower for k in bao_pin_keywords):
        return 'Bao pin'
        
    # Keywords for Thuê pin (Battery Leased)
    if 'thuê pin' in desc_lower:
        return 'Thuê pin'
        
    return 'Không rõ'

def process_file(filepath, out_path):
    df = pd.read_csv(filepath)
    if 'Mô tả' in df.columns:
        df['Battery_Status'] = df['Mô tả'].apply(extract_battery_status)
    else:
        df['Battery_Status'] = 'Không rõ'
        
    logger.info(f"{filepath.name} Battery Status Distribution:")
    logger.info(f"\n{df['Battery_Status'].value_counts()}")
    
    df.to_csv(out_path, index=False, encoding='utf-8-sig')

def main():
    ROOT = Path(__file__).parent.parent
    data_dir = ROOT / "data" / "processed"
    
    oto_path = data_dir / "ev_model_ready_oto.csv"
    oto_out_path = data_dir / "ev_model_ready_oto_v2.csv"
    if oto_path.exists():
        process_file(oto_path, oto_out_path)
        
    bike_path = data_dir / "ev_model_ready_bike.csv"
    bike_out_path = data_dir / "ev_model_ready_bike_v2.csv"
    if bike_path.exists():
        process_file(bike_path, bike_out_path)

if __name__ == "__main__":
    main()
