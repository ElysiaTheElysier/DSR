import re
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from typing import Union
import emoji


class DataPreprocessor:
    """
    Executes rule-based cleaning on the harmonized vehicle dataset.
    Cleans messy text formats, purges non-EVs, and parses prices.
    """

    def __init__(self, input_path: Union[str, Path], output_dir: Union[str, Path]):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)

    def clean_text_formatting(self, text: str) -> str:
        """Removes messy line breaks, tabs, multiple spaces, and emojis from scraped text."""
        if pd.isna(text):
            return ""
        # Remove emojis
        text = emoji.replace_emoji(str(text), replace='')
        # Remove newlines, carriage returns, tabs, and multiple spaces
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def parse_vietnamese_price(self, price_str: str) -> float:
        """
        - 1. Handle Chotot's pre-formatted raw integers
        - 2. Handle obfuscated prices (e.g., "26x triệu" -> "260 triệu")
        - 3. Extract 'Tỷ' (Billions)
        - 4. Extract 'Triệu' (Millions)
        - 5. Fallback: Strip all non-numeric characters
        """
        if pd.isna(price_str):
            return np.nan

        p_str = str(price_str).lower().strip()

        if p_str.isdigit():
            return float(p_str)

        p_str = p_str.replace('x', '0')

        total_vnd = 0.0

        if 'tỷ' in p_str:
            parts = p_str.split('tỷ')
            try:
                ty_val = float(parts[0].strip().replace(',', '.'))
                total_vnd += ty_val * 1_000_000_000
            except ValueError:
                pass
            p_str = parts[1] if len(parts) > 1 else ""

        if 'triệu' in p_str:
            tr_part = p_str.split('triệu')[0].strip()
            tr_part = tr_part.replace('.', '')
            try:
                total_vnd += float(tr_part) * 1_000_000
            except ValueError:
                pass

        if total_vnd > 0:
            return total_vnd

        digits = re.sub(r'[^\d]', '', str(price_str))
        return float(digits) if digits else np.nan

    def run_pipeline(self) -> None:
        """
        - 1. Fix Poor Formatting in Text Columns
        - 2. The ICE Purge - Remove internal combustion engine vehicles based on:
        - 3. Price Standardization
        - 4. Drop Redundant & Zero-Variance Columns
        - 5. Standardize date
        """
        logger.info(f"Loading dataset from {self.input_path}")
        df = pd.read_csv(self.input_path)
        initial_len = len(df)

        logger.info("Cleaning whitespace, line breaks, and emojis from text columns...")
        text_cols = ['Tên xe', 'Tên người bán', 'Địa chỉ', 'Mô tả', 'Kiểu dáng', 'Màu ngoại thất', 'Tình trạng']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text_formatting)

        # --- Phase 1: Rule-Based Deduplication & Cleaning ---
        initial_count = len(df)
        
        # Filter out ICE car models by name (since Động cơ was removed)
        ice_car_models = ['fadil', 'lux a', 'lux sa', 'president']
        ice_pattern = '|'.join(ice_car_models)
        is_ice_model = df['Tên xe'].str.lower().str.contains(ice_pattern, na=False)

        df_ev = df[~is_ice_model].copy()
        logger.info(f"Purged {initial_count - len(df_ev)} ICE vehicles by name. {len(df_ev)} EV records remain.")

        df_ev['Giá_VND'] = df_ev['Giá'].apply(self.parse_vietnamese_price)

        # Otodien prices are in millions (e.g. 286 instead of 286,000,000)
        is_otodien = df_ev['Website'] == 'otodien.vn'
        # Multiply by 1M if it's less than 100,000 (meaning it was parsed as raw integer rather than VNĐ)
        df_ev.loc[is_otodien & (df_ev['Giá_VND'] < 100000), 'Giá_VND'] *= 1_000_000

        # Clean 'Số Km đã đi' to numeric
        def clean_mileage(val):
            if pd.isna(val):
                return np.nan
            val = str(val).lower().replace('km', '').replace(',', '').replace('.', '').strip()
            try:
                return float(val)
            except:
                return np.nan
                
        df_ev['Số Km đã đi'] = df_ev['Số Km đã đi'].apply(clean_mileage)
        
        # Fill missing mileage with 0 for new cars
        is_new = df_ev['Tình trạng'].astype(str).str.lower().isin(['mới', 'xe mới'])
        df_ev.loc[is_new & df_ev['Số Km đã đi'].isna(), 'Số Km đã đi'] = 0

        # Purge unrealistic prices (< 10M VND or null)
        initial_price_len = len(df_ev)
        df_ev = df_ev[df_ev['Giá_VND'] >= 10_000_000].copy()

        cols_to_drop = ['Giá', 'Động cơ', 'Hộp số']
        df_ev = df_ev.drop(columns=[col for col in cols_to_drop if col in df_ev.columns])

        null_prices = df_ev['Giá_VND'].isnull().sum()
        logger.info(f"Price parsing complete. Found {null_prices} unparsable/null prices.")

        df_ev['Ngày đăng'] = pd.to_datetime(df_ev['Ngày đăng'], format='mixed', dayfirst=True).dt.strftime('%Y-%m-%d')

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split datasets
        df_oto = df_ev[df_ev['vehicle_type'] == 'oto_dien'].copy()
        df_bike = df_ev[df_ev['vehicle_type'].isin(['xe_may_dien', 'xe_dap_dien'])].copy()
        
        out_oto = self.output_dir / "ev_cleaned_oto.csv"
        out_bike = self.output_dir / "ev_cleaned_bike.csv"
        
        # Remove 'Số chỗ ngồi', 'Màu ngoại thất', and 'Kiểu dáng' for bikes as requested
        for col in ['Số chỗ ngồi', 'Màu ngoại thất', 'Kiểu dáng']:
            if col in df_bike.columns:
                df_bike = df_bike.drop(columns=[col])

        # Save to interim
        df_oto.to_csv(out_oto, index=False, encoding='utf-8-sig')
        df_bike.to_csv(out_bike, index=False, encoding='utf-8-sig')
        
        logger.info(f"Rule-based preprocessing complete.")
        logger.info(f"Saved {len(df_oto)} cars to {out_oto}")
        logger.info(f"Saved {len(df_bike)} bikes to {out_bike}")


if __name__ == "__main__":
    ROOT_PATH = Path(__file__).parent.parent
    INPUT_FILE = ROOT_PATH / "data" / "interim" / "merged_raw_listings.csv"
    OUTPUT_DIR = ROOT_PATH / "data" / "interim"

    preprocessor = DataPreprocessor(input_path=INPUT_FILE, output_dir=OUTPUT_DIR)
    preprocessor.run_pipeline()
