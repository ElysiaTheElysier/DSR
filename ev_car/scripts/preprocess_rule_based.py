import re
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from typing import Union


class DataPreprocessor:
    """
    Executes rule-based cleaning on the harmonized vehicle dataset.
    Cleans messy text formats, purges non-EVs, and parses prices.
    """

    def __init__(self, input_path: Union[str, Path], output_dir: Union[str, Path]):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)

    def clean_text_formatting(self, text: str) -> str:
        """Removes messy line breaks, tabs, and multiple spaces from scraped text."""
        if pd.isna(text):
            return ""
        return re.sub(r'\s+', ' ', str(text)).strip()

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

        logger.info("Cleaning whitespace and line breaks from 'Tên xe'...")
        df['Tên xe'] = df['Tên xe'].apply(self.clean_text_formatting)

        # ICE model patterns for both cars and motorbikes
        ice_car_models = ['fadil', 'lux a', 'lux sa', 'president']
        ice_pattern = '|'.join(ice_car_models)

        is_combustion_engine = df['Động cơ'].astype(str).str.lower().isin(['xăng', 'dầu', 'xăng/dầu'])

        is_ice_model = df['Tên xe'].str.lower().str.contains(ice_pattern, na=False)

        df_ev = df[~is_combustion_engine & ~is_ice_model].copy()
        logger.info(f"Purged {initial_len - len(df_ev)} ICE vehicles. {len(df_ev)} EV records remain.")

        df_ev['Giá_VND'] = df_ev['Giá'].apply(self.parse_vietnamese_price)

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
