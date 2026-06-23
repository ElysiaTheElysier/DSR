import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# Config
DATA_DIR = r"C:\Ki_5\final\ev_car\data\raw"
REPORT_DIR = r"C:\Ki_5\final\ev_car\reports\eda\1_before_harmonized"
os.makedirs(REPORT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def load_chotot_json(filepath, vehicle_type):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    
    flat_data = [flatten_dict(item) for item in data]
    df = pd.DataFrame(flat_data)
    df['website'] = 'chotot'
    df['vehicle_category'] = vehicle_type
    return df

def load_csv(filepath, website, vehicle_type):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except:
        df = pd.read_csv(filepath, encoding='utf-8')
    df['website'] = website
    df['vehicle_category'] = vehicle_type
    return df

def main():
    print("Loading raw data...")
    df_ct_bike = load_chotot_json(os.path.join(DATA_DIR, "chotot", "bicycles.json"), "Xe đạp điện")
    df_ct_moto = load_chotot_json(os.path.join(DATA_DIR, "chotot", "motorbikes.json"), "Xe máy điện")
    df_ct_car = load_chotot_json(os.path.join(DATA_DIR, "chotot", "cars.json"), "Ô tô điện")
    
    df_otodien = load_csv(os.path.join(DATA_DIR, "otodien", "data_xe_dien.csv"), "otodien", "Ô tô điện")
    df_vfluot = load_csv(os.path.join(DATA_DIR, "vfluot", "xevinfastluot_full.csv"), "vfluot", "Ô tô điện")
    df_bonbanh = load_csv(os.path.join(DATA_DIR, "bonbanh.csv"), "bonbanh", "Ô tô điện")
    
    datasets = {
        'Chotot_Bicycles': df_ct_bike,
        'Chotot_Motorbikes': df_ct_moto,
        'Chotot_Cars': df_ct_car,
        'Otodien': df_otodien,
        'VFLuot': df_vfluot,
        'Bonbanh': df_bonbanh
    }
    
    datasets = {k: v for k, v in datasets.items() if not v.empty}
    
    # Combined basic info for overall distributions
    summary_list = []
    for name, df in datasets.items():
        for _, row in df.iterrows():
            summary_list.append({
                'Website': row.get('website', 'unknown'),
                'Vehicle_Type': row.get('vehicle_category', 'unknown'),
                'Dataset': name
            })
    df_summary = pd.DataFrame(summary_list)
    
    # Plot missing data per dataset
    print("Generating Missing Data plots...")
    for name, df in datasets.items():
        df_cols = df.drop(columns=['website', 'vehicle_category'], errors='ignore')
        missing_perc = df_cols.isna().mean() * 100
        if df_cols.empty or len(df_cols) == 0:
            continue
            
        plt.figure(figsize=(10, 6))
        top_missing = missing_perc.sort_values(ascending=False).head(20)
        sns.barplot(x=top_missing.values, y=top_missing.index, palette="viridis")
        plt.title(f"Missing Data Percentage - {name} (Top 20 Cols)")
        plt.xlabel("Percentage (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, f"missing_data_{name}.png"))
        plt.close()

    # Distribution by Website
    print("Generating Website Distribution plot...")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df_summary, x='Website', palette="Set2")
    plt.title("Number of Listings per Website")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "distribution_website.png"))
    plt.close()

    # Copy files
    print("Copying raw files to EDA folder...")
    for file in [
        os.path.join(DATA_DIR, "chotot", "bicycles.json"),
        os.path.join(DATA_DIR, "chotot", "motorbikes.json"),
        os.path.join(DATA_DIR, "chotot", "cars.json"),
        os.path.join(DATA_DIR, "otodien", "data_xe_dien.csv"),
        os.path.join(DATA_DIR, "vfluot", "xevinfastluot_full.csv"),
        os.path.join(DATA_DIR, "bonbanh.csv")
    ]:
        if os.path.exists(file):
            shutil.copy(file, REPORT_DIR)
            
    print(f"Done EDA Phase 1. Reports and data in: {REPORT_DIR}")

if __name__ == "__main__":
    main()
