import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

DATA_FILE = r"C:\Ki_5\final\ev_car\data\interim\merged_raw_listings.csv"
REPORT_DIR = r"C:\Ki_5\final\ev_car\reports\eda\2_after_harmonized"
os.makedirs(REPORT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

def main():
    print(f"Loading harmonized data from {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        print("Data file not found. Run harmonize_datasets.py first.")
        return
        
    df = pd.read_csv(DATA_FILE)
    
    # 1. Missing Data Percentage
    print("Generating Missing Data Plot...")
    missing_perc = df.isna().mean() * 100
    plt.figure(figsize=(10, 8))
    sns.barplot(x=missing_perc.values, y=missing_perc.index, palette="viridis")
    plt.title("Missing Data Percentage - Harmonized Dataset")
    plt.xlabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "missing_data_harmonized.png"))
    plt.close()

    # 2. Source vs Vehicle Type Crosstab Heatmap
    print("Generating Crosstab Heatmap...")
    ct = pd.crosstab(df['Website'], df['vehicle_type'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
    plt.title("Vehicle Count by Source and Type")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "source_vs_type_heatmap.png"))
    plt.close()

    # Copy CSV
    print("Copying CSV to EDA folder...")
    shutil.copy(DATA_FILE, REPORT_DIR)
    
    print(f"Done EDA Phase 2. Reports and data in: {REPORT_DIR}")

if __name__ == "__main__":
    main()
