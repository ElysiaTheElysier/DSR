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

    CORE_FEATURES = [
        "Ngày đăng",
        "Tên xe",
        "Giá",
        "Tên người bán",
        "Địa chỉ",
        "Năm sản xuất",
        "Tình trạng",
        "Số Km đã đi",
        "Xuất xứ",
        "Kiểu dáng",
        "Hộp số",
        "Động cơ",
        "Màu ngoại thất",
        "Màu nội thất",
        "Số chỗ ngồi",
        "Số cửa",
        "Dẫn động",
        "Mô tả",
        "Link",
        "Website",
        "vehicle_type"
    ]

    def __init__(self, raw_data_dir: Union[str, Path], output_path: Union[str, Path]):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_path = Path(output_path)

    def _enforce_schema(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Ensures the dataframe has exactly the CORE_FEATURES.
        Missing columns are filled with NaN. Extra columns are dropped.
        """
        for col in self.CORE_FEATURES:
            if col not in df.columns:
                df[col] = np.nan

        logger.info(f"[{source_name}] Schema enforced. Output shape: {df[self.CORE_FEATURES].shape}")
        return df[self.CORE_FEATURES]

    def process_standard_csv(self, filename: str) -> pd.DataFrame:
        """Processes datasets that already perfectly match the core features.

        Add a 'Website' column based on filename heuristics to identify the source platform.
        """
        filepath = self.raw_data_dir / filename
        try:
            df = pd.read_csv(filepath, encoding="utf-8-sig")

            if "bonbanh" in filename.lower():
                df["Website"] = "bonbanh.com"
            elif "vinfastluot" in filename.lower():
                df["Website"] = "xevinfastluot.com"
            else:
                df["Website"] = "unknown"

            df["vehicle_type"] = "oto_dien"
            return self._enforce_schema(df, filename)
        except Exception:
            logger.exception(f"Failed to process {filename}")
            return pd.DataFrame(columns=self.CORE_FEATURES)

    def process_otodien(self, filename: str) -> pd.DataFrame:
        """Processes Otodien by mapping available columns and handling implicit EV data.

        Otodien's is 100% EV platform, so we can safely impute 'Hộp số' to 'Số tự động' and 'Động cơ' to 'Điện'
        for all records.
        """
        filepath = self.raw_data_dir / filename
        try:
            df = pd.read_csv(filepath)

            column_mapping = {
                "Ngày đăng": "Ngày đăng",
                "Tên": "Tên xe",
                "Tiền (VNĐ)": "Giá",
                "Người dùng": "Tên người bán",
                "Vị trí": "Địa chỉ",
                "Kiểu dáng": "Kiểu dáng",
                "Màu bên ngoài": "Màu ngoại thất",
                "Số chỗ ngồi": "Số chỗ ngồi",
                "Thông tin mô tả": "Mô tả",
                "ID": "Link"
            }
            df = df.rename(columns=column_mapping)

            df["Hộp số"] = "Số tự động"
            df["Động cơ"] = "Điện"
            df["Website"] = "otodien.vn"
            df["vehicle_type"] = "oto_dien"

            return self._enforce_schema(df, filename)

        except Exception:
            logger.exception(f"Failed to process {filename}")
            return pd.DataFrame(columns=self.CORE_FEATURES)

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
            return pd.DataFrame(columns=self.CORE_FEATURES)

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
            return pd.DataFrame(columns=self.CORE_FEATURES)
        except Exception:
            logger.exception(f"Failed to process {filename}")
            return pd.DataFrame(columns=self.CORE_FEATURES)

    def run_pipeline(self) -> None:
        """Executes the extraction, mapping, and merging process."""
        logger.info("Starting Data Harmonization Pipeline.")

        df_bonbanh = self.process_standard_csv("bonbanh.csv")
        df_vinfast = self.process_standard_csv("xevinfastluot_full.csv")
        df_otodien = self.process_otodien("data_xe_dien_web_otodien.csv")
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
    ROOT_PATH = Path.cwd().parent
    RAWD_PATH = ROOT_PATH / "data" / "raw"
    PROC_PATH = ROOT_PATH / "data" / "interim" / "merged_raw_listings.csv"

    harmonizer = DataHarmonizer(raw_data_dir=RAWD_PATH, output_path=PROC_PATH)
    harmonizer.run_pipeline()
