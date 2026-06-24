"""
Script: fix_bonbanh_csv.py
Sửa triệt để mọi vấn đề data quality trong file bonbanh.csv:
  1. Clean cột "Tên xe" bị rác HTML (\n\t)
  2. Fill "Số Km đã đi" = "0 Km" cho xe mới (vì web bonbanh không hiển thị trường này cho xe mới)
  3. Fill "Tên người bán" và "Địa chỉ" nếu NaN thì ghi "Không rõ"
  4. Loại bỏ mọi whitespace thừa trong toàn bộ cột string
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import re

INPUT  = r"C:\Ki_5\final\ev_car\data\raw\bonbanh.csv"
OUTPUT = r"C:\Ki_5\final\ev_car\data\raw\bonbanh.csv"

df = pd.read_csv(INPUT)
print(f"[BEFORE] Shape: {df.shape}")
print(f"[BEFORE] NaN per col:\n{df.isna().sum()}\n")

# ========== 1. Clean tất cả cột string: loại bỏ \n \t và whitespace thừa ==========
str_cols = df.select_dtypes(include='object').columns
for col in str_cols:
    df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip() if pd.notna(x) else x)

# ========== 2. Clean cột "Tên xe" - loại bỏ prefix "Xe " nếu bị lặp ==========
df['Tên xe'] = df['Tên xe'].str.replace(r'^Xe\s+', '', regex=True).str.strip()

# ========== 3. Fill "Số Km đã đi" cho xe mới ==========
mask_xe_moi = (df['Số Km đã đi'].isna()) & (df['Tình trạng'] == 'Xe mới')
df.loc[mask_xe_moi, 'Số Km đã đi'] = '0 Km'

# Nếu vẫn còn NaN ở Km (xe cũ mà thiếu) -> ghi "Không rõ"
df['Số Km đã đi'] = df['Số Km đã đi'].fillna('Không rõ')

# ========== 4. Fill các cột còn lại nếu NaN ==========
df['Tên người bán'] = df['Tên người bán'].fillna('Không rõ')
df['Địa chỉ'] = df['Địa chỉ'].fillna('Không rõ')

# ========== 5. Loại bỏ string "N/A" thành actual NaN rồi fill lại ==========
df = df.replace('N/A', pd.NA)
# Fill lại tất cả NaN còn sót
for col in str_cols:
    df[col] = df[col].fillna('Không rõ')

# ========== 6. Save ==========
df.to_csv(OUTPUT, index=False, encoding='utf-8-sig')

print(f"[AFTER] Shape: {df.shape}")
print(f"[AFTER] NaN per col:\n{df.isna().sum()}\n")
print(f"[AFTER] Sample Tên xe:")
for t in df['Tên xe'].head(5).tolist():
    print(f"  {t}")
print(f"\n[AFTER] Sample Km:")
for k in df['Số Km đã đi'].head(10).tolist():
    print(f"  {k}")
print(f"\nĐã lưu file sạch tại: {OUTPUT}")
