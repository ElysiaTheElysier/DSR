"""
LLM Extraction Pipeline for Electric Vehicle (EV) Listings.

This module utilizes OpenAI's GPT-5-Nano to asynchronously extract and impute 
structured data from unstructured vehicle listing descriptions.

- Auto-Resumption: Skips already processed IDs.
- Incremental Checkpointing: Appends to CSV immediately after each batch.
- Sanitization: Cleans illegal characters to prevent API 400 Bad Request errors.
"""

import os
import time
import asyncio
import pandas as pd
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
import openai
from openai import AsyncOpenAI
from loguru import logger
from tqdm.asyncio import tqdm
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()


class VehicleExtraction(BaseModel):
    id: int = Field(description="The unique numeric ID provided in the prompt.")
    reasoning: str = Field(description="Internal logic check. "
                                       "If data is missing, state 'Not explicitly mentioned, returning null'.")
    brand: str = Field(description="Standardized brand name (e.g., 'VinFast', 'Wuling').")
    car_model: str = Field(description="Clean model name (e.g., 'VF3', 'VF8', 'Bingo').")
    imputed_year: Optional[int] = Field(description="Production year. Null if not stated.")
    imputed_mileage_km: Optional[int] = Field(description="Integer km. 1 vạn = 10000. If 'Mới 100%', set to 0.")
    imputed_condition: Optional[Literal["Mới 100%", "Đã qua sử dụng"]] = Field(
        description="State based on ODO or keywords like 'zin keng', 'đã lăn bánh'.")
    battery_status: Optional[Literal["Mua pin", "Thuê pin"]] = Field(
        description="Status: 'đã gồm pin'=Mua pin, 'thuê pin'=Thuê pin. Null if unknown.")
    is_accident_free: Optional[bool] = Field(
        description="True if 'zin keng', 'không lỗi'. False if 'va quệt'. Null if unknown.")
    has_aftermarket_mods: bool = Field(
        description="True ONLY if owner added parts (3 thanh giằng, bệ tỳ tay). False for factory safety specs.")


class VehicleBatch(BaseModel):
    results: List[VehicleExtraction]


SYSTEM_PROMPT = """
You are a professional Data Scientist. Extract EV listing data into JSON.
STRICT RULES:
1. ABSENCE IS NOT EVIDENCE: If a feature is not explicitly in the text, return null. DO NOT ASSUME.
2. NEW CAR LOGIC: If 'imputed_condition' is 'Mới 100%', set 'imputed_mileage_km' to 0.
3. MODS: True only for user additions (e.g., wrap, extra seats). False for factory specs (e.g., ABS, LFP battery).
4. REASONING: Must mention the specific phrase that led to your conclusion.
"""


class EVExtractionPipeline:
    def __init__(self, input_file: Path, output_file: Path, batch_size: int = 15):
        self.input_file = input_file
        self.output_file = output_file
        self.batch_size = batch_size
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        self.start_time = 0
        self.total_cost = 0.0
        self.input_tokens = 0
        self.output_tokens = 0

        self.semaphore = asyncio.Semaphore(3)
        self.file_lock = asyncio.Lock()

        logger.remove()
        logger.add("logs/extraction.log", rotation="50 MB")
        logger.add(lambda msg: print(msg, end=""), level="INFO")

    def _update_cost(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.input_tokens += prompt_tokens
        self.output_tokens += completion_tokens
        self.total_cost += (prompt_tokens * 0.05 / 1e6) + (completion_tokens * 0.40 / 1e6)

    def sanitize_text(self, text: str) -> str:
        """Removes hidden characters and escapes quotes to prevent 400 Bad Request JSON errors."""
        if pd.isna(text):
            return ""
        clean = str(text).replace('"', "'")
        clean = "".join(char for char in clean if char.isprintable())
        return clean.strip()

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(7),
        retry=retry_if_exception_type(openai.RateLimitError),
        before_sleep=lambda retry_state: logger.warning(
            f"Rate limit hit. Retrying in {retry_state.next_action.sleep}s...")
    )
    async def _call_api(self, payload: str):
        return await self.client.beta.chat.completions.parse(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload}
            ],
            response_format=VehicleBatch,
            seed=42
        )

    async def process_batch(self, batch_df: pd.DataFrame, pbar: tqdm) -> None:
        payload = "Extract details for these records:\n"
        for idx, row in batch_df.iterrows():
            clean_name = self.sanitize_text(row['Tên xe'])
            clean_desc = self.sanitize_text(row['Mô tả'])
            payload += f"ID: {idx} | Name: {clean_name} | Desc: {clean_desc[:2500]}\n"

        async with self.semaphore:
            try:
                response = await self._call_api(payload)
                self._update_cost(response.usage.prompt_tokens, response.usage.completion_tokens)

                results = [res.model_dump() for res in response.choices[0].message.parsed.results]

                if results:
                    res_df = pd.DataFrame(results)
                    async with self.file_lock:
                        write_header = not self.output_file.exists()
                        res_df.to_csv(self.output_file, mode='a', index=False, header=write_header,
                                      encoding='utf-8-sig')

            except Exception as e:
                logger.error(f"Fatal batch processing error: {e}")
            finally:
                pbar.update(len(batch_df))

    async def run(self) -> None:
        self.start_time = time.time()
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.input_file)

        if self.output_file.exists():
            existing_df = pd.read_csv(self.output_file)
            processed_ids = existing_df['id'].unique()
            df = df[~df.index.isin(processed_ids)]
            logger.info(f"Resuming: {len(processed_ids)} already processed. {len(df)} records remaining.")
        else:
            logger.info(f"Starting gpt-5-nano pipeline. Processing {len(df)} records.")

        if df.empty:
            logger.success("All records have already been processed! Pipeline complete.")
            return

        tasks = []
        with tqdm(total=len(df), desc="Extracting") as pbar:
            for i in range(0, len(df), self.batch_size):
                batch = df.iloc[i: i + self.batch_size]
                tasks.append(self.process_batch(batch, pbar))

            await asyncio.gather(*tasks)

        duration = time.time() - self.start_time
        logger.success(f"Pipeline finished in {duration:.2f}s")
        logger.info(f"Tokens Used - Input: {self.input_tokens} | Output: {self.output_tokens}")
        logger.info(f"Total Exact Cost (gpt-5-nano): ${self.total_cost:.4f} USD")


if __name__ == "__main__":
    import nest_asyncio

    nest_asyncio.apply()
    ROOT = Path(__file__).parent.parent
    INPUT = ROOT / "data" / "interim" / "ev_cleaned_rule_based.csv"
    OUTPUT = ROOT / "data" / "interim" / "ev_extracted_gpt5nano.csv"

    pipeline = EVExtractionPipeline(INPUT, OUTPUT)
    asyncio.run(pipeline.run())
