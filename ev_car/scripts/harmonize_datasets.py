import json
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
from typing import Dict, List, Union


class DataHarmonizer:
    """
    A robust ETL pipeline to extract heterogeneous vehicle datasets,
    transform them into a unified 20-feature schema (including Website), 
    and load them into a single merged CSV.
    """

    FINAL_COLUMNS = [
        "Website",
        "Link",
        "Tên xe",
        "Giá",
        "Ngày đăng",
        "Tên người bán",
        "Địa chỉ",
        "Năm sản xuất",
        "Tình trạng",
        "Số Km đã đi",
        "Kiểu dáng",
        "Màu ngoại thất",
        "Số chỗ ngồi",
        "Mô tả",
        "vehicle_type"
    ]

    def __init__(self, raw_data_dir: Union[str, Path], output_path: Union[str, Path]):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_path = Path(output_path)

    def _enforce_schema(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Ensures the DataFrame matches the final schema, adding NaNs where missing."""
        for col in self.FINAL_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df_out = df[self.FINAL_COLUMNS].copy()
        logger.info(f"[{source_name}] Schema enforced. Output shape: {df_out.shape}")
        return df_out

    def process_standard_csv(self, filename: str) -> pd.DataFrame:
        """Processes CSV files that already have standard Vietnamese columns (Bonbanh, VinfastLuot)."""
        filepath = self.raw_data_dir / filename
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            if 'Website' not in df.columns:
                df['Website'] = 'bonbanh.com' if 'bonbanh' in filename else 'xevinfastluot.vn'
            if 'vehicle_type' not in df.columns:
                df['vehicle_type'] = 'oto_dien'
            if 'Động cơ' in df.columns:
                df = df[~df['Động cơ'].astype(str).str.lower().isin(['xăng', 'dầu', 'xăng/dầu', 'hybrid'])]
            return self._enforce_schema(df, filename)
        except Exception:
            logger.exception(f"Failed to process {filename}")
            return pd.DataFrame(columns=self.FINAL_COLUMNS)

    def process_otodien(self, filename: str) -> pd.DataFrame:
        """Processes Otodien by mapping available columns and handling implicit EV data.

        Otodien is a 100% EV platform, so we can safely impute:
        - 'Hộp số' → 'Số tự động'
        - 'Động cơ' → 'Điện'
        - 'Xuất xứ' → extracted from title/description or 'Lắp ráp trong nước' (default for VinFast)
        
        We also extract 'Năm sản xuất' from the title (pattern: "VinFast VF8 2023 Plus")
        and impute 'Tình trạng' / 'Số Km đã đi' based on description context.
        """
        import re
        filepath = self.raw_data_dir / filename
        try:
            df = pd.read_csv(filepath)

            column_mapping = {
                "Ngày đăng": "Ngày đăng",
                "Tên": "Tên xe",
                "Tiền (VNĐ)": "Giá",
                "Người dùng": "Tên người bán",
                "Vị trí": "Địa chỉ",
                "Năm sản xuất": "Năm sản xuất_raw",
                "Tình trạng": "Tình trạng_raw",
                "Số Km đã đi": "Số Km đã đi_raw",
                "Kiểu dáng": "Kiểu dáng",
                "Màu bên ngoài": "Màu ngoại thất",
                "Số chỗ ngồi": "Số chỗ ngồi",
                "Thông tin mô tả": "Mô tả",
                "ID": "Link"
            }
            df = df.rename(columns=column_mapping)

            # --- Extract Năm sản xuất from title ---
            def extract_year(row):
                raw_year = row.get("Năm sản xuất_raw")
                if pd.notna(raw_year) and str(raw_year).strip() != "":
                    try:
                        return int(float(raw_year))
                    except:
                        pass
                
                title = row.get("Tên xe")
                if pd.isna(title):
                    return np.nan
                match = re.search(r'\b(20[1-3]\d)\b', str(title))
                return int(match.group(1)) if match else np.nan
            df["Năm sản xuất"] = df.apply(extract_year, axis=1)

            # --- Impute Tình trạng from Mô tả ---
            def extract_condition(row):
                raw_cond = str(row.get("Tình trạng_raw", ""))
                if "mới" in raw_cond.lower() or "đã sử dụng" in raw_cond.lower():
                    return raw_cond.strip()
                    
                desc = str(row.get("Mô tả", "")).lower()
                title = str(row.get("Tên xe", "")).lower()
                combined = title + " " + desc
                used_keywords = ["đã qua sử dụng", "secondhand", "cũ", "đã đi", "odo", "km đã đi", "lướt"]
                new_keywords = ["xe mới", "mới 100%", "chưa lăn bánh", "giao ngay", "mới giao", "new"]
                for kw in used_keywords:
                    if kw in combined:
                        return "Đã sử dụng"
                for kw in new_keywords:
                    if kw in combined:
                        return "Xe mới"
                return np.nan
            df["Tình trạng"] = df.apply(extract_condition, axis=1)

            # --- Impute Số Km đã đi from Mô tả ---
            def extract_mileage(row):
                raw_km = str(row.get("Số Km đã đi_raw", ""))
                if pd.notna(raw_km) and raw_km.strip() not in ["", "nan", "None"]:
                    return raw_km.strip()
                    
                desc = str(row.get("Mô tả", ""))
                title = str(row.get("Tên xe", ""))
                combined = title + " " + desc
                # Try to find patterns like "odo 12,000km", "1000 km", etc.
                match = re.search(r'(?:odo|ODO|km đã đi|đã đi)\s*[:\-]?\s*([\d,.]+)\s*(?:km|Km|KM)?', combined)
                if match:
                    km_str = match.group(1).replace(',', '').replace('.', '')
                    try:
                        return f"{int(km_str):,} Km"
                    except:
                        pass
                match2 = re.search(r'([\d,.]+)\s*(?:km|Km|KM)\s*(?:đã đi|đã chạy)', combined)
                if match2:
                    km_str = match2.group(1).replace(',', '').replace('.', '')
                    try:
                        return f"{int(km_str):,} Km"
                    except:
                        pass
                if row.get("Tình trạng") == "Xe mới":
                    return "0 Km"
                return np.nan
            df["Số Km đã đi"] = df.apply(extract_mileage, axis=1)

            # --- Static imputation for EV-only platform ---
            df["Website"] = "otodien.vn"
            df["vehicle_type"] = "oto_dien"

            return self._enforce_schema(df, filename)

        except Exception:
            logger.exception(f"Failed to process {filename}")
            return pd.DataFrame(columns=self.FINAL_COLUMNS)

    def process_chotot_json(self, filename: str) -> pd.DataFrame:
        """Flattens the JSON and maps the deeply nested Chotot features."""
        filepath = self.raw_data_dir / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            df = pd.json_normalize(data)

            column_mapping = {
                "exact_date_posted": "Ngày đăng",
                "ad.subject": "Tên xe",
                "ad.price": "Giá",
                "ad.account_name": "Tên người bán",
                "ad_params.address.value": "Địa chỉ",
                "ad_params.mfdate.value": "Năm sản xuất",
                "ad_params.condition_ad.value": "Tình trạng",
                "ad_params.mileage_v2.value": "Số Km đã đi",
                "ad_params.carorigin.value": "Xuất xứ",
                "ad_params.cartype.value": "Kiểu dáng",
                "ad_params.gearbox.value": "Hộp số",
                "ad_params.fuel.value": "Động cơ",
                "ad_params.carcolor.value": "Màu ngoại thất",
                "ad_params.carseats.value": "Số chỗ ngồi",
                "ad.body": "Mô tả",
                "url": "Link"
            }

            df = df.rename(columns=column_mapping)
            df["Website"] = "chotot.com"
            df["vehicle_type"] = "oto_dien"

            return self._enforce_schema(df, filename)

        except Exception:
            logger.exception(f"Failed to process {filename}")
            return pd.DataFrame(columns=self.FINAL_COLUMNS)

    def process_chotot_ev_json(self, filename: str, vehicle_type: str) -> pd.DataFrame:
        """Processes Chotot motorbike/bicycle JSON from Gateway API scraper.

        Uses the same column mapping as the car spider output.
        """
        filepath = self.raw_data_dir / filename
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)

            df = pd.json_normalize(data)

            column_mapping = {
                "exact_date_posted": "Ngày đăng",
                "ad.subject": "Tên xe",
                "ad.price": "Giá",
                "ad.account_name": "Tên người bán",
                "ad_params.address.value": "Địa chỉ",
                "ad_params.mfdate.value": "Năm sản xuất",
                "ad_params.condition_ad.value": "Tình trạng",
                "ad_params.mileage_v2.value": "Số Km đã đi",
                "ad_params.fuel.value": "Động cơ",
                "ad_params.motorbiketype.value": "Kiểu dáng",
                "ad.body": "Mô tả",
                "url": "Link"
            }

            df = df.rename(columns=column_mapping)
            df["Website"] = "chotot.com"
            df["vehicle_type"] = vehicle_type

            return self._enforce_schema(df, filename)

        except FileNotFoundError:
            logger.warning(f"File not found: {filename}. Skipping.")
            return pd.DataFrame(columns=self.FINAL_COLUMNS)
        except Exception:
            logger.exception(f"Failed to process {filename}")
            return pd.DataFrame(columns=self.FINAL_COLUMNS)

    def run_pipeline(self) -> None:
        """Executes the extraction, mapping, and merging process."""
        logger.info("Starting Data Harmonization Pipeline.")

        df_bonbanh = self.process_standard_csv("bonbanh.csv")
        df_vinfast = self.process_standard_csv("vfluot/xevinfastluot_full.csv")
        df_otodien = self.process_otodien("otodien/data_xe_dien.csv")
        df_chotot = self.process_chotot_json("chotot/cars.json")

        # New EV sources: motorbikes & bicycles from Chotot
        df_motorbike = self.process_chotot_ev_json("chotot/motorbikes.json", "xe_may_dien")
        df_bicycle = self.process_chotot_ev_json("chotot/bicycles.json", "xe_dap_dien")

        logger.info("Concatenating datasets.")
        all_dfs = [df_bonbanh, df_vinfast, df_otodien, df_chotot, df_motorbike, df_bicycle]
        merged_df = pd.concat(all_dfs, ignore_index=True)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(self.output_path, index=False, encoding='utf-8-sig')

        logger.info(f"Successfully exported harmonized dataset to {self.output_path}")
        logger.info(f"Total records: {len(merged_df)}")
        logger.info(f"By vehicle_type: {merged_df['vehicle_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    ROOT_PATH = Path(__file__).resolve().parent.parent
    RAWD_PATH = ROOT_PATH / "data" / "raw"
    PROC_PATH = ROOT_PATH / "data" / "interim" / "merged_raw_listings.csv"

    harmonizer = DataHarmonizer(raw_data_dir=RAWD_PATH, output_path=PROC_PATH)
    harmonizer.run_pipeline()
