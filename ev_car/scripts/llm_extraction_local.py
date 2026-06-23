"""
Synchronous Local LLM Extraction Pipeline with Auto-Retry & Guardrails.

Optimized for 4GB VRAM.
Features:
- Pure synchronous execution for maximum GPU utilization.
- Few-Shot Prompting: Dramatically improves SLM entity extraction.
- Python Guardrails: Hard-codes logic to fix SLM hallucinations
  (e.g., forcing condition to 'Used' if mileage > 0).
TODO: Improve this:
- Slow
- Not accurate
- Missing brand/model parsing logic (currently relies on SLM to do this, which is often wrong).
"""

import argparse
import time
import json
import re
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import yaml
from loguru import logger
from ollama import Client
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm


class VehicleExtraction(BaseModel):
    id: int
    reasoning: Optional[str] = Field(default="")
    brand: Optional[str] = Field(default=None)
    car_model: Optional[str] = Field(default=None)
    imputed_year: Optional[int] = Field(default=None)
    imputed_mileage_km: Optional[int] = Field(default=None)
    imputed_condition: Optional[str] = Field(default=None)
    battery_status: Optional[str] = Field(default=None)
    is_accident_free: Optional[bool] = Field(default=None)
    has_aftermarket_mods: Optional[bool] = Field(default=False)
    vehicle_type: Optional[str] = Field(default=None)


SYSTEM_PROMPT = """
You are a precise Data Extraction Pipeline. Extract vehicle data into a strict JSON object.

EXAMPLE 1 (Car):
ID: 99
Name: Xe VinFast VF8 Plus 2023
Desc: Bán xe cũ chạy 1 vạn, zin keng, đã lên cam 360, mua đứt pin.

OUTPUT:
{"id": 99, "reasoning": "chạy 1 vạn, zin keng, lên cam 360", "brand": "VinFast", "car_model": "VF8 Plus", "imputed_year": 2023, "imputed_mileage_km": 10000, "imputed_condition": "Đã qua sử dụng", "battery_status": "Mua pin", "is_accident_free": true, "has_aftermarket_mods": true, "vehicle_type": "oto_dien"}

EXAMPLE 2 (Electric Motorbike):
ID: 200
Name: VinFast Feliz S 2024 Trắng
Desc: Bán xe điện Feliz S, odo 2000km, xe mới 98%, sạc đầy đi 100km.

OUTPUT:
{"id": 200, "reasoning": "odo 2000km, mới 98%, sạc đầy 100km", "brand": "VinFast", "car_model": "Feliz S", "imputed_year": 2024, "imputed_mileage_km": 2000, "imputed_condition": "Đã qua sử dụng", "battery_status": null, "is_accident_free": null, "has_aftermarket_mods": false, "vehicle_type": "xe_may_dien"}

EXAMPLE 3 (Electric Bicycle):
ID: 300
Name: Xe đạp điện Yadea M6L 2023
Desc: Xe đạp điện mới 90%, bình ắc quy mới thay, chạy 50km/lần sạc.

OUTPUT:
{"id": 300, "reasoning": "mới 90%, bình ắc quy mới thay", "brand": "Yadea", "car_model": "M6L", "imputed_year": 2023, "imputed_mileage_km": null, "imputed_condition": "Đã qua sử dụng", "battery_status": null, "is_accident_free": null, "has_aftermarket_mods": false, "vehicle_type": "xe_dap_dien"}

RULES:
- vehicle_type: "oto_dien" for cars, "xe_may_dien" for motorbikes/scooters, "xe_dap_dien" for e-bikes.
- 1 vạn = 10000 km. 
- If absent from text, return null. DO NOT guess.
- Return ONLY valid JSON, no extra text.

OUTPUT TEMPLATE:
{"id": <ID>, "reasoning": "<keywords>", "brand": null, "car_model": null, "imputed_year": null, "imputed_mileage_km": null, "imputed_condition": null, "battery_status": null, "is_accident_free": null, "has_aftermarket_mods": false, "vehicle_type": null}
"""


class SyncLocalPipeline:
    def __init__(self, config_path: Path, model_override: Optional[str] = None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.root_dir = config_path.parent.parent
        self.input_file = self.root_dir / self.config["pipeline"]["input_file"]
        self.model_name = model_override or self.config["pipeline"]["default_model"]

        output_filename = f"ev_extracted_{self.model_name.replace(':', '_')}.csv"
        self.output_file = self.root_dir / self.config["pipeline"]["output_dir"] / output_filename

        self.ollama_options = self.config.get("ollama_options", {"temperature": 0.0, "num_ctx": 1024})
        self.client = Client()

        logger.remove()
        logger.add(self.root_dir / f"logs/ollama_{self.model_name.replace(':', '_')}.log", rotation="50 MB")
        logger.add(lambda msg: print(msg, end=""), level="INFO")

    def clean_text(self, text: str) -> str:
        if pd.isna(text): return ""
        clean = str(text).replace('"', "'")
        return "".join(char for char in clean if char.isprintable()).strip()

    def extract_json_block(self, raw_text: str) -> str:
        text = raw_text.strip()
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end + 1]
        return text

    def enforce_logic_guardrails(self, data: dict) -> dict:
        """Fixes SLM logical hallucinations before saving to CSV."""

        mileage = data.get('imputed_mileage_km')
        if mileage is not None and mileage > 100:
            data['imputed_condition'] = "Đã qua sử dụng"
        elif data.get('imputed_condition') == "Mới 100%":
            data['imputed_mileage_km'] = 0

        model_str = data.get('car_model', '')
        if model_str and isinstance(model_str, str):
            if "vinfast" in model_str.lower():
                data['brand'] = "VinFast"

            model_str = re.sub(r'(?i)^(xe\s+)?(vinfast\s+)?', '', model_str)
            model_str = re.sub(r'\b202\d\b', '', model_str)
            data['car_model'] = model_str.strip()

        brand_str = data.get('brand', '')
        if brand_str and isinstance(brand_str, str):
            if brand_str.lower() == "vinfast":
                data['brand'] = "VinFast"
            elif brand_str.lower() == "wuling":
                data['brand'] = "Wuling"

        return data

    def run(self) -> None:
        start_time = time.time()
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.input_file)

        if self.output_file.exists():
            existing_df = pd.read_csv(self.output_file)
            processed_ids = existing_df['id'].unique()
            df = df[~df.index.isin(processed_ids)]
            logger.info(f"Resuming: {len(processed_ids)} processed. {len(df)} remaining.")
        else:
            logger.info(f"Starting {self.model_name} pipeline. Records: {len(df)}")

        if df.empty:
            logger.success("All records processed!")
            return

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=self.model_name):
            name = self.clean_text(row['Tên xe'])
            desc = self.clean_text(row['Mô tả'])
            payload = f"ID: {idx}\nName: {name}\nDesc: {desc[:800]}\n\nJSON Output:\n"

            max_retries = 3
            success = False

            for attempt in range(max_retries):
                try:
                    response = self.client.chat(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": payload}
                        ],
                        format="json",
                        options=self.ollama_options
                    )

                    raw_output = response['message']['content']
                    clean_string = self.extract_json_block(raw_output)
                    raw_dict = json.loads(clean_string, strict=False)

                    parsed_data = VehicleExtraction(**raw_dict)
                    data_dict = parsed_data.model_dump()

                    cleaned_data_dict = self.enforce_logic_guardrails(data_dict)

                    res_df = pd.DataFrame([cleaned_data_dict])
                    res_df.to_csv(
                        self.output_file,
                        mode='a',
                        index=False,
                        header=not self.output_file.exists(),
                        encoding='utf-8-sig'
                    )

                    success = True
                    break

                except json.JSONDecodeError as je:
                    if attempt == max_retries - 1:
                        logger.error(f"ID {idx} Skipped: Broken JSON Syntax -> {je}")
                except ValidationError as ve:
                    if attempt == max_retries - 1:
                        errors = "; ".join([f"{e['loc'][0]}: {e['msg']}" for e in ve.errors() if e.get('loc')])
                        logger.error(f"ID {idx} Skipped: Schema Mismatch -> {errors}")
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"ID {idx} Skipped: {e}")

            if not success:
                logger.warning(f"ID {idx} completely failed after {max_retries} attempts.")

        duration = time.time() - start_time
        logger.success(f"Pipeline complete in {duration:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local EV extraction.")
    parser.add_argument("--model", type=str, help="Override default model (e.g., qwen2.5:3b)")
    args = parser.parse_args()

    ROOT = Path(__file__).parent.parent
    CONFIG_PATH = ROOT / "configs" / "local_llm_config.yaml"

    pipeline = SyncLocalPipeline(config_path=CONFIG_PATH, model_override=args.model)
    pipeline.run()
