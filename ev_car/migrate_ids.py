import pandas as pd

try:
    old_cleaned = pd.read_csv('data/interim/old_ev_cleaned_oto.csv', encoding='utf-16')
except:
    try:
        old_cleaned = pd.read_csv('data/interim/old_ev_cleaned_oto.csv', encoding='utf-8-sig')
    except:
        old_cleaned = pd.read_csv('data/interim/old_ev_cleaned_oto.csv', encoding='latin1')

new_cleaned = pd.read_csv('data/interim/ev_cleaned_oto.csv', encoding='utf-8')
extracted = pd.read_csv('data/interim/ev_extracted_oto.csv', encoding='utf-8')

print("Old cleaned:", len(old_cleaned))
print("New cleaned:", len(new_cleaned))
print("Extracted:", len(extracted))

idx_to_link = {idx: row['Link'] for idx, row in old_cleaned.iterrows()}
extracted['Link'] = extracted['id'].map(idx_to_link)
extracted = extracted.drop(columns=['id'])

extracted.to_csv('data/interim/ev_extracted_oto.csv', index=False, encoding='utf-8-sig')
print("Successfully migrated ev_extracted_oto.csv")
