import pandas as pd
import openai
import json

key = 'YOUR_API_KEY'
client = openai.OpenAI(api_key=key)

df = pd.read_csv('data/interim/ev_cleaned_oto.csv')
valid_rows = df[df['Tên xe'].notna()]
if not valid_rows.empty:
    sample_row = valid_rows.iloc[100]
    name = str(sample_row['Tên xe'])
    desc = str(sample_row.get('Mô tả', ''))[:500]
    prompt = f'Name: {name}\nDesc: {desc}'
    
    sys_prompt = """You are a precise Data Extraction Pipeline. Extract vehicle data into a strict JSON object.
RULES:
1. ABSENCE IS NOT EVIDENCE: If a feature is not explicitly in the text, return null. DO NOT ASSUME.
2. NEW CAR LOGIC: If imputed_condition is Mới 100%, set imputed_mileage_km to 0.
3. MODS: True only for user additions.
4. vehicle_type: Must be either oto_dien, xe_may_dien, or xe_dap_dien.
5. 1 vạn = 10000 km.
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
    
    res = client.chat.completions.create(
        model='gpt-4o-mini',
        response_format={'type':'json_object'},
        messages=[
            {'role':'system', 'content': sys_prompt},
            {'role':'user', 'content': prompt}
        ]
    )
    
    out = '=== RAW DATA ===\n'
    out += f'TÊN XE: {name}\n'
    out += f'MÔ TẢ: {desc}\n\n'
    out += '=== LLM JSON ===\n'
    out += res.choices[0].message.content
    
    with open('sample_llm.txt', 'w', encoding='utf-8') as f:
        f.write(out)
