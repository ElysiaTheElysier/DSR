import os
import time
import pandas as pd
from openai import OpenAI
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "YOUR_API_KEY"
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """
You are a precise Data Extraction Pipeline. Read the vehicle description and extract missing data into a strict JSON object.

RULES:
1. ABSENCE IS NOT EVIDENCE: If a feature is not explicitly in the text, return null. DO NOT ASSUME.
2. 1 vạn = 10000 km.

OUTPUT TEMPLATE (Return ONLY this JSON structure without markdown formatting):
{
    "Link": "<string>", 
    "imputed_body_style": "<string or null, e.g. SUV, Sedan, Hatchback>",
    "imputed_seats": <integer or null>,
    "imputed_exterior_color": "<string or null>",
    "imputed_location": "<string or null>",
    "imputed_year": <integer or null>, 
    "imputed_mileage_km": <integer or null>, 
    "imputed_condition": "<Mới 100% or Đã qua sử dụng or null>"
}
"""

class OpenAIExtractionPipeline:
    def __init__(self, input_file: Path, output_file: Path):
        self.input_file = input_file
        self.output_file = output_file
        self.client = OpenAI(api_key=API_KEY)
        
    def sanitize_text(self, text):
        if pd.isna(text):
            return ""
        clean = str(text).replace('"', "'").replace('\n', ' ')
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
        stop=stop_after_attempt(3)
    )
    def _call_api(self, payload: str):
        response = self.client.chat.completions.create(
            model=MODEL,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload}
            ]
        )
        return response.choices[0].message.content

    def process_row(self, row):
        clean_name = self.sanitize_text(row['Tên xe'])
        clean_desc = self.sanitize_text(row.get('Mô tả', ''))
        link = row['Link']
        payload = f"Link: {link}\nName: {clean_name}\nDesc: {clean_desc[:2000]}\n\nJSON Output:\n"
        
        try:
            raw_output = self._call_api(payload)
            clean_json = self.extract_json_block(raw_output)
            data_dict = json.loads(clean_json)
        except Exception as e:
            # Fallback
            data_dict = {
                "Link": link,
                "error": str(e)
            }
            
        data_dict['Link'] = link
        return data_dict

    def run(self, max_workers=5):
        logger.info(f"Loading data from {self.input_file}")
        df = pd.read_csv(self.input_file)
        
        # Determine missing rows
        target_cols = ['Kiểu dáng', 'Số Km đã đi', 'Số chỗ ngồi', 'Màu ngoại thất', 'Địa chỉ', 'Năm sản xuất']
        check_cols = [c for c in target_cols if c in df.columns]
        
        df_missing = df[df[check_cols].isna().any(axis=1)].copy()
        logger.info(f"Found {len(df_missing)} rows with missing data out of {len(df)}")
        
        if len(df_missing) == 0:
            logger.info("No missing data to process.")
            return
            
        records = df_missing.to_dict('records')
        results = []
        
        logger.info(f"Starting LLM extraction with model {MODEL} using OpenAI API")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {executor.submit(self.process_row, row): row for row in records}
            for future in tqdm(as_completed(future_to_row), total=len(records), desc="Extracting"):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    logger.error(f"Error processing row: {e}")
                    
        df_results = pd.DataFrame(results)
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        logger.info(f"Saved extracted results to {self.output_file}")

def main():
    ROOT = Path(__file__).parent.parent
    
    # Oto
    input_oto = ROOT / "data" / "interim" / "ev_cleaned_oto.csv"
    output_oto = ROOT / "data" / "interim" / "ev_extracted_missing_oto.csv"
    if input_oto.exists():
        pipeline_oto = OpenAIExtractionPipeline(input_oto, output_oto)
        pipeline_oto.run(max_workers=10)
        
    # Bike
    input_bike = ROOT / "data" / "interim" / "ev_cleaned_bike.csv"
    output_bike = ROOT / "data" / "interim" / "ev_extracted_missing_bike.csv"
    if input_bike.exists():
        pipeline_bike = OpenAIExtractionPipeline(input_bike, output_bike)
        pipeline_bike.run(max_workers=10)

if __name__ == "__main__":
    main()
