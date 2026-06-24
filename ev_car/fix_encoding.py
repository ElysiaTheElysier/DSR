import pandas as pd

def fix_mojibake(text):
    if not isinstance(text, str):
        return text
    try:
        # If it contains double-encoded utf-8, this will fix it
        return text.encode('latin1').decode('utf-8')
    except:
        return text

# 1. Fix otodien raw
oto_raw = 'data/raw/otodien/data_xe_dien.csv'
df_oto = pd.read_csv(oto_raw, encoding='utf-8')
df_oto = df_oto.map(fix_mojibake)
df_oto.to_csv(oto_raw, index=False, encoding='utf-8-sig')

print("Fixed otodien raw CSV encoding.")
