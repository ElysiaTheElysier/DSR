import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding='utf-8')

# Ensure Arial/sans-serif for clean English text
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

def format_price_en(x, pos):
    if x >= 1e9:
        return '%1.1f B' % (x * 1e-9) # Billion VND
    if x >= 1e6:
        return '%1.0f M' % (x * 1e-6) # Million VND
    return '%1.0f' % x

translation_dict = {
    'Website': 'Website',
    'Link': 'Link',
    'Tên xe': 'Vehicle Model',
    'Ngày đăng': 'Post Date',
    'Tên người bán': 'Seller Name',
    'Địa chỉ': 'Location',
    'Năm sản xuất': 'Manufacture Year',
    'Tình trạng': 'Condition',
    'Số Km đã đi': 'Mileage (km)',
    'Kiểu dáng': 'Body Style',
    'Màu ngoại thất': 'Exterior Color',
    'Số chỗ ngồi': 'Seats',
    'Mô tả': 'Description',
    'vehicle_type': 'Vehicle Type',
    'Giá_VND': 'Price (VND)',
    'Log_Giá_VND': 'Log Price',
    'Log_Số_Km': 'Log Mileage'
}

def remove_extreme_outliers(df, col):
    Q1 = df[col].quantile(0.01)
    Q3 = df[col].quantile(0.99)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

def plot_numerical_distribution(df, col, out_dir, prefix, use_price_formatter=False):
    # Filter valid data
    if col not in df.columns: return
    data = df[df[col] > 0][col].dropna()
    if len(data) == 0: return
    
    # Remove top 1% extreme outliers for better visualization of the distribution
    data_clean = data[data < data.quantile(0.99)]
    
    mean_val = data_clean.mean()
    median_val = data_clean.median()
    std_val = data_clean.std()

    plt.figure(figsize=(12, 6))
    sns.histplot(data_clean, bins=50, kde=True, color='teal', alpha=0.6)
    
    # Add Statistical Markers
    plt.axvline(mean_val, color='red', linestyle='solid', linewidth=2, label=f'Mean: {mean_val:,.0f}')
    plt.axvline(median_val, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median_val:,.0f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='dotted', linewidth=2, label=f'+1 Std Dev')
    plt.axvline(mean_val - std_val, color='orange', linestyle='dotted', linewidth=2, label=f'-1 Std Dev')
    
    if use_price_formatter:
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_price_en))
        
    plt.title(f'{prefix} - Distribution of {col} (Model Readiness)', fontsize=14, fontweight='bold')
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    clean_col_name = col.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(out_dir / f"{prefix}_Dist_{clean_col_name}.png", dpi=300)
    plt.close()

def plot_boxplots(df, col, out_dir, prefix, use_price_formatter=False):
    if col not in df.columns: return
    data = df[df[col] > 0][col].dropna()
    if len(data) == 0: return
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=data, color='lightblue', fliersize=3)
    
    if use_price_formatter:
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_price_en))
        
    plt.title(f'{prefix} - Quartiles and Outliers for {col}', fontsize=14, fontweight='bold')
    plt.xlabel(col, fontsize=12)
    plt.tight_layout()
    
    clean_col_name = col.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(out_dir / f"{prefix}_Boxplot_{clean_col_name}.png", dpi=300)
    plt.close()

def plot_cardinality(df, out_dir, prefix):
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in ['Website', 'Link', 'Description', 'Post Date', 'Vehicle Type']]
    if not cat_cols: return
    
    cardinality = {col: df[col].nunique() for col in cat_cols}
    card_df = pd.DataFrame(list(cardinality.items()), columns=['Feature', 'Unique_Values']).sort_values('Unique_Values', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=card_df, x='Unique_Values', y='Feature', palette='rocket')
    
    for i, v in enumerate(card_df['Unique_Values']):
        plt.text(v + 1, i, str(v), color='black', va='center', fontweight='bold')
        
    plt.title(f'{prefix} - Categorical Feature Cardinality', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Unique Values', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    # Use log scale if max unique values is very high
    if card_df['Unique_Values'].max() > 1000:
        plt.xscale('log')
        plt.xlabel('Number of Unique Values (Log Scale)', fontsize=12)
        
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_Cardinality.png", dpi=300)
    plt.close()

def process_dataset(df, out_dir, prefix):
    # Translate columns
    df.rename(columns=translation_dict, inplace=True)
    
    # 1. Distributions with stats
    plot_numerical_distribution(df, 'Price (VND)', out_dir, prefix, use_price_formatter=True)
    plot_numerical_distribution(df, 'Mileage (km)', out_dir, prefix, use_price_formatter=False)
    plot_numerical_distribution(df, 'Manufacture Year', out_dir, prefix, use_price_formatter=False)
    
    # 1.5 Distributions for Log Transformed
    plot_numerical_distribution(df, 'Log Price', out_dir, prefix, use_price_formatter=False)
    plot_numerical_distribution(df, 'Log Mileage', out_dir, prefix, use_price_formatter=False)
    
    # 2. Boxplots (Outliers)
    plot_boxplots(df, 'Price (VND)', out_dir, prefix, use_price_formatter=True)
    plot_boxplots(df, 'Mileage (km)', out_dir, prefix, use_price_formatter=False)
    
    # 3. Cardinality
    plot_cardinality(df, out_dir, prefix)

def main():
    ROOT = Path(__file__).parent.parent
    data_dir = ROOT / "data" / "processed"
    out_dir = ROOT / "reports" / "eda" / "6_model_readiness"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Oto
    oto_path = data_dir / "ev_model_ready_oto.csv"
    if oto_path.exists():
        print("Processing Oto Model Readiness...")
        df_oto = pd.read_csv(oto_path)
        process_dataset(df_oto, out_dir, "Oto")
        
    # Bike
    bike_path = data_dir / "ev_model_ready_bike.csv"
    if bike_path.exists():
        print("Processing Bike Model Readiness...")
        df_bike = pd.read_csv(bike_path)
        process_dataset(df_bike, out_dir, "Bike")

if __name__ == "__main__":
    main()
