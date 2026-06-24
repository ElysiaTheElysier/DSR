import os
import pandas as pd
import openai
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "YOUR_API_KEY"
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are an advanced Data Extraction AI specializing in the Vietnamese EV market.
Your task is to extract highly accurate vehicle specifications from raw classified descriptions into a strict JSON object.

CRITICAL RULES:
1. ABSENCE IS NOT EVIDENCE: If a feature is not explicitly stated, return null. DO NOT ASSUME.
2. CONDITION & MILEAGE (Odo) LOGIC: 
   - A car is ONLY "Mới 100%" if described explicitly as 'chưa lăn bánh', 'xe mới 100%', or if it is a clear dealership promotional post without any Odo history.
   - If ANY Odo > 0 is mentioned, or if Vietnamese slang like 'xe lướt', 'kính koong', 'leng keng', 'mới keng', 'như mới', 'nguyên zin', 'ken' is used, it MUST be "Đã qua sử dụng".
   - DO NOT confuse battery condition with vehicle condition (e.g., 'bình mới thay', 'bình mới 100%' means the battery is new, NOT the vehicle).
   - DO NOT confuse RANGE with ODO (e.g., 'đi được 60km/lần sạc', 'quãng đường 1200km' is the battery range, NOT the mileage!).
   - Odo calculation: '1 vạn' = 10000 km, '1,4 vạn' = 14000 km.
3. MODS (has_aftermarket_mods):
   - Set true ONLY if user explicitly added accessories (e.g., lót sàn, dán phim, camera hành trình, phủ taplo, độ cốp, lên đồ chơi).
   - Set false if described as 'nguyên bản' and absolutely no accessories are listed. Otherwise return null.
4. ACCIDENT FREE (is_accident_free):
   - Set true if terms like 'không đâm đụng', 'không ngập nước', 'không thủy kích', 'phát hiện lỗi tặng xe', 'keo chỉ nguyên zin' are present.
5. BATTERY (battery_status):
   - "Mua pin" if terms like 'mua pin', 'sẵn pin', 'bao gồm pin', 'pin SDI', 'pin CATL' are present.
   - "Thuê pin" if terms like 'thuê pin', 'cọc pin' are present.
6. vehicle_type: strictly "oto_dien", "xe_may_dien", or "xe_dap_dien".
7. REASONING: Explain your extraction logic briefly in Vietnamese.

OUTPUT TEMPLATE:
{
    "id": 0, 
    "reasoning": "<keywords>", 
    "brand": "<string>", 
    "car_model": "<string>", 
    "imputed_year": <int>, 
    "imputed_mileage_km": <int>, 
    "imputed_condition": "<string>", 
    "battery_status": "<string>", 
    "is_accident_free": <bool>, 
    "has_aftermarket_mods": <bool>, 
    "vehicle_type": "<string>"
}"""

class OpenAIExtractionPipeline:
    def __init__(self, input_file: Path, output_file: Path):
        self.input_file = input_file
        self.output_file = output_file
        self.client = openai.OpenAI(api_key=API_KEY, max_retries=0)
        
        logger.remove()
        logger.add("logs/openai_extraction.log", rotation="50 MB")
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
        response = self.client.chat.completions.create(
            model=MODEL,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload}
            ],
            timeout=30.0
        )
        return response.choices[0].message.content

    def process_row(self, row, idx):
        clean_name = self.sanitize_text(row['Tên xe'])
        clean_desc = self.sanitize_text(row.get('Mô tả', ''))
        payload = f"ID: {idx}\nName: {clean_name}\nDesc: {clean_desc[:2000]}\n\nJSON Output:\n"
        
        raw_output = self._call_api(payload)
        clean_json = self.extract_json_block(raw_output)
        data_dict = json.loads(clean_json)
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
        
        # 7 workers for stable throughput without deadlocking API
        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = {executor.submit(self.process_row, task[0], task[1]): task[1] for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting {self.input_file.name}"):
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
        pipeline_oto = OpenAIExtractionPipeline(INPUT_OTO, OUTPUT_OTO)
        pipeline_oto.run()
        
    INPUT_BIKE = ROOT / "data" / "interim" / "ev_cleaned_bike.csv"
    OUTPUT_BIKE = ROOT / "data" / "interim" / "ev_extracted_bike.csv"
    if INPUT_BIKE.exists():
        pipeline_bike = OpenAIExtractionPipeline(INPUT_BIKE, OUTPUT_BIKE)
        pipeline_bike.run()
