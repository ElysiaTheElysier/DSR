"""
Two-Wheelers Attribute Extraction.
Uses ONLY OpenAI API (gpt-5-nano) to extract brands, models, production years,
and ODO mileages. Increased timeout to 120s and retries to 3 to accommodate slow responses.
Does NOT fall back to gpt-4o-mini.
"""

import os
import re
import sys
import json
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, List
from openai import AsyncOpenAI
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
INPUT_FILE = INTERIM / "two_wheelers_cleaned.csv"
OUTPUT_FILE = INTERIM / "two_wheelers_extracted.csv"

# Load environment variables
load_dotenv(dotenv_path=ROOT / ".env")

class TwoWheelerExtraction(BaseModel):
    id: int = Field(description="The unique ID of the listing.")
    reasoning: str = Field(description="Brief explanation of findings.")
    brand: Optional[str] = Field(description="Standardized brand name (e.g. VinFast, Dat Bike, Yadea, Pega, Giant, Asama, Dibao, Sonsu, Honda, Yamaha). Null if unknown.")
    model: Optional[str] = Field(description="Clean model name (e.g. Feliz, Evo200, Klara, Vento, Theon, Quantum, Weaver, Cap-A, Odora). Null if unknown.")
    imputed_year: Optional[int] = Field(description="Production year (4 digits, e.g. 2023). Null if not mentioned.")
    imputed_mileage_km: Optional[int] = Field(description="ODO mileage in km. '1.5 vạn' = 15000. Null if not mentioned.")
    has_aftermarket_mods: bool = Field(description="True if owner added parts (giỏ, cốp, đèn, phuộc). False if stock.")

class TwoWheelerBatch(BaseModel):
    results: List[TwoWheelerExtraction]

SYSTEM_PROMPT = """
You are a professional Data Scientist. Extract electric bicycle and electric motorbike listing features into JSON.
STRICT RULES:
1. ABSENCE IS NOT EVIDENCE: If a feature (year, mileage) is not mentioned in the title/desc, set it to null.
2. MODS: Set has_aftermarket_mods to true ONLY if owner added items (e.g. extra box, custom rack, custom seats).
3. BRAND/MODEL: Map to standard brands (VinFast, Dat Bike, Yadea, Pega, Honda, Giant, Asama, Dibao).
"""

def extract_regex_fallback(row) -> dict:
    title = str(row['title']).lower()
    desc = str(row['description']).lower()
    full_text = f"{title} {desc}"
    
    # 1. Brand
    brand = None
    brands_map = {
        'VinFast': ['vinfast', 'vin fast', 'vf', 'evo200', 'feliz', 'klara', 'vento', 'theon', 'impes', 'ludo'],
        'Dat Bike': ['datbike', 'dat bike', 'quantum', 'weaver'],
        'Yadea': ['yadea', 'odora', 'orla', 'buye', 'voltguard'],
        'Pega': ['pega', 'cap-a', 'crazy bull', 'hkbike', 'hk bike'],
        'Honda': ['honda', 'benly', 'u-go', 'mono'],
        'Peugeot': ['peugeot'],
        'Yamaha': ['yamaha'],
        'Giant': ['giant'],
        'Asama': ['asama'],
        'Dibao': ['dibao', 'pansy', 'creer', 'gogo'],
        'Sonsu': ['sonsu'],
        'DK Bike': ['dkbike', 'dk bike', 'dk']
    }
    for b, keywords in brands_map.items():
        if any(kw in full_text for kw in keywords):
            brand = b
            break
    if not brand:
        brand = "Unknown"
        
    # 2. Model
    model = None
    models_list = [
        'evo200', 'feliz', 'klara', 'vento', 'theon', 'impes', 'ludo',
        'quantum', 'weaver', 'odora', 'orla', 'buye', 'voltguard', 'cap-a', 'gogo',
        'pansy', 'creer', 'aura'
    ]
    for m in models_list:
        if m in full_text:
            model = m.title()
            break
            
    if not model:
        # Heuristic: first word after brand in title
        words = str(row['title']).split()
        if len(words) > 1:
            model = words[1]
        else:
            model = "Unknown"
            
    # 3. Year
    imputed_year = None
    year_match = re.search(r'\b(201[5-9]|202[0-6])\b', full_text)
    if year_match:
        imputed_year = int(year_match.group(1))
    else:
        doi_match = re.search(r'đời\s*(\d{2})\b', full_text)
        if doi_match:
            imputed_year = 2000 + int(doi_match.group(1))
            
    # 4. Mileage
    imputed_mileage_km = None
    van_match = re.search(r'(\d+(?:\.\d+)?)\s*vạn', full_text)
    if van_match:
        imputed_mileage_km = int(float(van_match.group(1)) * 10000)
    else:
        km_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:km|odo|chạy)', full_text)
        if km_match:
            val = float(km_match.group(1))
            if val >= 500: # range check
                imputed_mileage_km = int(val)
                
    # 5. Mods
    has_aftermarket_mods = False
    mods_keywords = ['độ', 'lên đồ', 'phụ kiện', 'giỏ', 'cốp', 'chống trộm', 'yên da', 'đồ chơi', 'thay sạc']
    if any(kw in full_text for kw in mods_keywords):
        has_aftermarket_mods = True
        
    return {
        'id': row['id'],
        'reasoning': 'Regex fallback parser',
        'brand': brand,
        'model': model,
        'imputed_year': imputed_year,
        'imputed_mileage_km': imputed_mileage_km,
        'has_aftermarket_mods': has_aftermarket_mods
    }

class TwoWheelerLlmPipeline:
    def __init__(self, batch_size: int = 5, max_concurrency: int = 4):
        self.batch_size = batch_size
        self.client = None
        self.use_api = True
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.file_lock = asyncio.Lock()

    async def _call_api(self, payload: str):
        # Strictly use gpt-5-nano as requested by the user, with a longer timeout
        return await self.client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload}
            ],
            response_format=TwoWheelerBatch,
            timeout=120.0
        )

    async def process_batch(self, batch_df: pd.DataFrame, pbar: tqdm) -> None:
        async with self.semaphore:
            if not self.use_api:
                results = [extract_regex_fallback(row) for _, row in batch_df.iterrows()]
            else:
                payload = "Extract two-wheeler features for these records:\n"
                for _, row in batch_df.iterrows():
                    payload += f"ID: {row['id']} | Title: {row['title']} | Desc: {str(row['description'])[:500]}\n"
                
                try:
                    response = await self._call_api(payload)
                    results = [res.model_dump() for res in response.choices[0].message.parsed.results]
                except Exception as e:
                    logger.error(f"gpt-5-nano API Batch error: {e}. Falling back to Regex for this batch.")
                    results = [extract_regex_fallback(row) for _, row in batch_df.iterrows()]
            
            # Write to CSV under lock
            async with self.file_lock:
                res_df = pd.DataFrame(results)
                write_header = not OUTPUT_FILE.exists()
                res_df.to_csv(OUTPUT_FILE, mode='a', index=False, header=write_header, encoding='utf-8-sig')
            
            pbar.update(len(batch_df))

    async def run(self) -> None:
        if not INPUT_FILE.exists():
            logger.error(f"Input file {INPUT_FILE} does not exist!")
            return
            
        df = pd.read_csv(INPUT_FILE)
        logger.info(f"Loaded {len(df)} records for attribute extraction.")
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No OPENAI_API_KEY found! Using Regex-only extractor.")
            self.use_api = False
        else:
            # Set max_retries = 3 for robustness with gpt-5-nano
            self.client = AsyncOpenAI(api_key=api_key, max_retries=3)
            self.use_api = True
            
        # Resume check
        processed_ids = set()
        if OUTPUT_FILE.exists():
            try:
                existing = pd.read_csv(OUTPUT_FILE)
                processed_ids = set(existing['id'].unique())
                logger.info(f"Found existing output. Skipping {len(processed_ids)} already processed records.")
            except Exception:
                OUTPUT_FILE.unlink()
                
        df = df[~df['id'].isin(processed_ids)].copy()
        
        if df.empty:
            logger.success("All records already processed!")
            return
            
        tasks = []
        with tqdm(total=len(df), desc="Extracting") as pbar:
            for i in range(0, len(df), self.batch_size):
                batch = df.iloc[i:i+self.batch_size]
                tasks.append(self.process_batch(batch, pbar))
                
            await asyncio.gather(*tasks)
                
        logger.success(f"Extraction completed! Results saved to {OUTPUT_FILE.name}")

if __name__ == "__main__":
    pipeline = TwoWheelerLlmPipeline()
    asyncio.run(pipeline.run())
