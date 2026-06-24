import pandas as pd
import random

def sample_desc(file_path, n=15):
    try:
        df = pd.read_csv(file_path)
        valid = df[df['Mô tả'].notna() & (df['Mô tả'] != '')]
        if len(valid) == 0:
            return []
        n = min(n, len(valid))
        sampled = valid.sample(n=n, random_state=42)
        return sampled['Mô tả'].tolist()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

oto_desc = sample_desc('data/interim/ev_cleaned_oto.csv', 20)
bike_desc = sample_desc('data/interim/ev_cleaned_bike.csv', 10)

all_desc = ["=== OTO DESCRIPTIONS ==="] + oto_desc + ["\n=== BIKE DESCRIPTIONS ==="] + bike_desc

with open('sample_descriptions.txt', 'w', encoding='utf-8') as f:
    for i, d in enumerate(all_desc):
        if d.startswith("==="):
            f.write(f"{d}\n")
        else:
            clean_d = str(d).replace('\n', ' ')
            f.write(f"- {clean_d[:500]}...\n")
