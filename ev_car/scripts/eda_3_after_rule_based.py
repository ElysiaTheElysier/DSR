import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

DATA_FILES = {
    "oto": r"C:\Ki_5\final\ev_car\data\interim\ev_cleaned_oto.csv",
    "bike": r"C:\Ki_5\final\ev_car\data\interim\ev_cleaned_bike.csv"
}
REPORT_DIR = r"C:\Ki_5\final\ev_car\reports\eda\3_after_rule_based"

sns.set_theme(style="whitegrid")

def generate_plots(df, prefix, out_dir):
    print(f"Generating plots for {prefix}...")
    os.makedirs(out_dir, exist_ok=True)
    
    # Clean up old plots if they exist
    for f in os.listdir(out_dir):
        if f.startswith(f"{prefix}_") and f.endswith(".png"):
            os.remove(os.path.join(out_dir, f))
            
    # 1. Price Distribution Plot (Raw Scale)
    plt.figure(figsize=(10, 6))
    valid_prices = df[df['Giá_VND'] > 0]['Giá_VND']
    if not valid_prices.empty:
        sns.histplot(valid_prices, bins=50, kde=True, color='blue')
        plt.title(f"Distribution of Prices (VND) - Raw Scale ({prefix})")
        plt.xlabel("Price in VND")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_price_distribution_raw.png"))
    plt.close()

    # 2. Price Distribution Plot (Log Scale)
    plt.figure(figsize=(10, 6))
    if not valid_prices.empty:
        sns.histplot(valid_prices, bins=50, kde=True, log_scale=True, color='purple')
        plt.title(f"Distribution of Prices (VND) - Log Scale ({prefix})")
        plt.xlabel("Price in VND (Log Scale)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_price_distribution_log.png"))
    plt.close()

    # 3. Distribution Website
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='Website', palette="Set2", hue='Website', legend=False)
    plt.title(f"Distribution by Website ({prefix})")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_distribution_website.png"))
    plt.close()

    # 4. Missing Data
    missing_perc = df.isna().mean() * 100
    missing_perc = missing_perc[missing_perc > 0]
    plt.figure(figsize=(10, 8))
    if len(missing_perc) > 0:
        top_missing = missing_perc.sort_values(ascending=False).head(20)
        sns.barplot(x=top_missing.values, y=top_missing.index, palette="viridis", hue=top_missing.index, legend=False)
    else:
        plt.text(0.5, 0.5, 'No Missing Data', horizontalalignment='center', verticalalignment='center', fontsize=20)
    plt.title(f"Missing Data Percentage - {prefix}")
    plt.xlabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_missing_data.png"))
    plt.close()

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    for prefix, file_path in DATA_FILES.items():
        print(f"Loading cleaned data from {file_path}")
        if not os.path.exists(file_path):
            print(f"Data file {file_path} not found. Run preprocess_rule_based.py first.")
            continue
            
        df = pd.read_csv(file_path)
        generate_plots(df, prefix, REPORT_DIR)
        
        # Copy CSV
        print(f"Copying {prefix} CSV to EDA folder...")
        try:
            shutil.copy(file_path, REPORT_DIR)
        except PermissionError:
            print(f"Warning: Could not copy {file_path} because it is currently open in another program (e.g. Excel).")
        
    print(f"Done EDA Phase 3. Reports and data in: {REPORT_DIR}")

if __name__ == "__main__":
    main()
