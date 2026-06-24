import pandas as pd
import json

df_raw = pd.read_csv('data/interim/ev_cleaned_oto.csv')
df_ext = pd.read_csv('data/interim/ev_extracted_oto.csv')

valid = df_ext[df_ext['brand'].notna()]
if valid.empty:
    print('Still no valid rows')
else:
    sample = valid.head(3)
    for _, row in sample.iterrows():
        idx = int(row['id'])
        raw_row = df_raw.iloc[idx]
        print(f'=== ID: {idx} ===')
        print(f'TÊN XE: {raw_row["Tên xe"]}')
        desc = str(raw_row.get("Mô tả", ""))[:500].replace("\n", " ")
        print(f'MÔ TẢ: {desc}...')
        print('JSON LLM TRẢ VỀ:')
        # Keep only the columns we care about to avoid clutter
        d = row.to_dict()
        print(json.dumps(d, ensure_ascii=False, indent=2))
        print('\n')
