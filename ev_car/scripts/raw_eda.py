import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Config
DATA_DIR = r"C:\Ki_5\final\ev_car\data\raw"
REPORT_DIR = r"C:\Ki_5\final\ev_car\reports\raw_eda"
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
    
    # Flatten JSON
    flat_data = [flatten_dict(item) for item in data]
    df = pd.DataFrame(flat_data)
    df['website'] = 'chotot'
    df['vehicle_category'] = vehicle_type
    return df

def load_csv(filepath, website, vehicle_type):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    try:
        # Try read with normal utf-8-sig
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except:
        df = pd.read_csv(filepath, encoding='utf-8')
    df['website'] = website
    df['vehicle_category'] = vehicle_type
    return df

def main():
    print("Loading data...")
    # Load Chotot
    df_ct_bike = load_chotot_json(os.path.join(DATA_DIR, "chotot", "bicycles.json"), "Xe đạp điện")
    df_ct_moto = load_chotot_json(os.path.join(DATA_DIR, "chotot", "motorbikes.json"), "Xe máy điện")
    df_ct_car = load_chotot_json(os.path.join(DATA_DIR, "chotot", "cars.json"), "Ô tô điện")
    
    # Load Otodien & Vfluot
    df_otodien = load_csv(os.path.join(DATA_DIR, "otodien", "data_xe_dien.csv"), "otodien", "Ô tô điện")
    df_vfluot = load_csv(os.path.join(DATA_DIR, "vfluot", "xevinfastluot_full.csv"), "vfluot", "Ô tô điện")
    
    datasets = {
        'Chotot_Bicycles': df_ct_bike,
        'Chotot_Motorbikes': df_ct_moto,
        'Chotot_Cars': df_ct_car,
        'Otodien': df_otodien,
        'VFLuot': df_vfluot
    }
    
    # Drop empty datasets
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
    
    # 1. Missing Data Plots per dataset
    print("Generating Missing Data plots...")
    for name, df in datasets.items():
        # Drop added metadata cols for missingness
        df_cols = df.drop(columns=['website', 'vehicle_category'], errors='ignore')
        
        # Calculate missing percentage
        missing_perc = df_cols.isna().mean() * 100
        # If no missing data at all, or if dataframe is empty, handle gracefully
        if df_cols.empty or len(df_cols) == 0:
            continue
            
        plt.figure(figsize=(10, 6))
        # Plot top 20 columns with most missing data for readability
        top_missing = missing_perc.sort_values(ascending=False).head(20)
        sns.barplot(x=top_missing.values, y=top_missing.index, palette="viridis")
        plt.title(f"Missing Data Percentage - {name} (Top 20 Cols)")
        plt.xlabel("Percentage (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, f"missing_data_{name}.png"))
        plt.close()

    # 2. Distribution across websites
    print("Generating Website Distribution plot...")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df_summary, x='Website', palette="Set2")
    plt.title("Number of Listings per Website")
    plt.xlabel("Website")
    plt.ylabel("Count")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "distribution_website.png"))
    plt.close()

    # 3. Columns scraped
    print("Generating Columns Scraped plot...")
    col_counts = {name: len(df.columns) - 2 for name, df in datasets.items()} # -2 for our added cols
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=list(col_counts.keys()), y=list(col_counts.values()), palette="magma")
    plt.title("Number of Scraped Columns per Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Number of Columns")
    plt.xticks(rotation=45)
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "columns_scraped_count.png"))
    plt.close()

    # 4. Distribution of vehicle types
    print("Generating Vehicle Type Distribution plot...")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df_summary, x='Vehicle_Type', palette="pastel")
    plt.title("Distribution of Vehicle Types")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Count")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "distribution_vehicle_type.png"))
    plt.close()

    # 5. Distribution of types per website (Stacked)
    print("Generating Stacked Distribution plot...")
    pivot_df = df_summary.groupby(['Website', 'Vehicle_Type']).size().unstack(fill_value=0)
    pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set3')
    plt.title("Distribution of Vehicle Types per Website")
    plt.xlabel("Website")
    plt.ylabel("Count")
    plt.legend(title="Vehicle Type")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "distribution_type_per_website.png"))
    plt.close()
    
    print(f"Done! All plots saved to {REPORT_DIR}")

if __name__ == "__main__":
    main()
