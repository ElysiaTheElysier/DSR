import os
import time
import pandas as pd
import anthropic
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEYS = ["AGOP-DA2E-D3BD-59CF", "AGOP-BAE4-B148-1E86"]
BASE_URL = "https://api.nkq.vn"
MODEL = "claude-3-5-sonnet-20241022"

SYSTEM_PROMPT = """
You are a precise Data Extraction Pipeline. Extract vehicle data into a strict JSON object.

RULES:
1. ABSENCE IS NOT EVIDENCE: If a feature is not explicitly in the text, return null. DO NOT ASSUME.
2. NEW CAR LOGIC: If 'imputed_condition' is 'Mới 100%', set 'imputed_mileage_km' to 0.
3. MODS: True only for user additions (e.g., wrap, extra seats). False for factory specs.
4. REASONING: Must mention the specific phrase that led to your conclusion.
5. vehicle_type: Must be either "oto_dien", "xe_may_dien", or "xe_dap_dien" based on the data.
6. 1 vạn = 10000 km.

OUTPUT TEMPLATE (Return ONLY this JSON structure without markdown formatting):
{
    "id": <ID>, 
    "reasoning": "<keywords>", 
    "brand": "<string or null>", 
    "car_model": "<string or null>", 
    "imputed_year": <integer or null>, 
    "imputed_mileage_km": <integer or null>, 
    "imputed_condition": "<Mới 100% or Đã qua sử dụng or null>", 
    "battery_status": "<Mua pin or Thuê pin or null>", 
    "is_accident_free": <true/false/null>, 
    "has_aftermarket_mods": <true/false>, 
    "vehicle_type": "<string>"
}
"""

class AnthropicExtractionPipeline:
    def __init__(self, input_file: Path, output_file: Path):
        self.input_file = input_file
        self.output_file = output_file
        # Create multiple clients for load balancing across keys
        self.clients = [anthropic.Anthropic(api_key=key, base_url=BASE_URL) for key in API_KEYS]
        
        logger.remove()
        logger.add("logs/anthropic_extraction.log", rotation="50 MB")
        logger.add(lambda msg: print(msg, end=""), level="INFO")

    def sanitize_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        clean = str(text).replace('"', "'")
        clean = "".join(char for char in clean if char.isprintable())
        return clean.strip()
        
    def extract_json_block(self, raw_text: str) -> str:
        text = raw_text.strip()
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end + 1]
        return text

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(7)
    )
    def _call_api(self, payload: str):
        # Randomly pick a client to distribute load across keys
        client = random.choice(self.clients)
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": payload}
            ]
        )
        return response.content[0].text

    def process_row(self, row, idx):
        clean_name = self.sanitize_text(row['Tên xe'])
        clean_desc = self.sanitize_text(row.get('Mô tả', ''))
        payload = f"ID: {idx}\nName: {clean_name}\nDesc: {clean_desc[:2000]}\n\nJSON Output:\n"
        
        # NO try-except block here. Let it crash so we know exactly what is wrong.
        raw_output = self._call_api(payload)
        clean_json = self.extract_json_block(raw_output)
        data_dict = json.loads(clean_json)
        # Ensure ID is correct
        data_dict['id'] = idx
        return data_dict

    def run(self):
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(self.input_file)
        
        if self.output_file.exists():
            existing_df = pd.read_csv(self.output_file)
            processed_ids = set(existing_df['id'].unique())
            df_to_process = df[~df.index.isin(processed_ids)]
            logger.info(f"Resuming {self.input_file.name}: {len(processed_ids)} already processed. {len(df_to_process)} records remaining.")
        else:
            df_to_process = df
            logger.info(f"Starting {self.input_file.name}. Processing {len(df_to_process)} records.")

        if df_to_process.empty:
            logger.success(f"All records for {self.input_file.name} have been processed!")
            return

        tasks = [(row, idx) for idx, row in df_to_process.iterrows()]
        
        batch_size = 50 
        results_buffer = []
        
        write_header = not self.output_file.exists()
        
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = {executor.submit(self.process_row, task[0], task[1]): task[1] for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting {self.input_file.name}"):
                # NO try-except on future.result(). If it fails, the script crashes.
                res = future.result()
                results_buffer.append(res)
                    
                if len(results_buffer) >= batch_size:
                    res_df = pd.DataFrame(results_buffer)
                    res_df.to_csv(self.output_file, mode='a', index=False, header=write_header, encoding='utf-8-sig')
                    write_header = False
                    results_buffer = []
                    
        # Write any remaining
        if results_buffer:
            res_df = pd.DataFrame(results_buffer)
            res_df.to_csv(self.output_file, mode='a', index=False, header=write_header, encoding='utf-8-sig')

        logger.success(f"Pipeline finished for {self.input_file.name}")

if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    
    INPUT_OTO = ROOT / "data" / "interim" / "ev_cleaned_oto.csv"
    OUTPUT_OTO = ROOT / "data" / "interim" / "ev_extracted_oto.csv"
    if INPUT_OTO.exists():
        pipeline_oto = AnthropicExtractionPipeline(INPUT_OTO, OUTPUT_OTO)
        pipeline_oto.run()
        
    INPUT_BIKE = ROOT / "data" / "interim" / "ev_cleaned_bike.csv"
    OUTPUT_BIKE = ROOT / "data" / "interim" / "ev_extracted_bike.csv"
    if INPUT_BIKE.exists():
        pipeline_bike = AnthropicExtractionPipeline(INPUT_BIKE, OUTPUT_BIKE)
        pipeline_bike.run()
