import pandas as pd

# Đọc file CSV
df = pd.read_csv("xevinfastluot_full.csv")

# Lọc các dòng có phần động cơ là điện
filtered = df[df["Động cơ"].str.contains("điện", case=False, na=False)]

# Lưu ra file mới
filtered.to_csv("VFL_Final.csv", index=False)
