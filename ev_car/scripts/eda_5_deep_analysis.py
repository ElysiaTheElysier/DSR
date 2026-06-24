import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# Encoding for sys.stdout
sys.stdout.reconfigure(encoding='utf-8')

# Font configuration for Vietnamese
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'Segoe UI', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def format_price(x, pos):
    'The two args are the value and tick position'
    if x >= 1e9:
        return '%1.1f Tỷ' % (x * 1e-9)
    if x >= 1e6:
        return '%1.0f Tr' % (x * 1e-6)
    return '%1.0f' % x

def remove_outliers(df, col):
    Q1 = df[col].quantile(0.05)
    Q3 = df[col].quantile(0.95)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

def plot_univariate(df, out_dir, prefix):
    # 1. Price Distribution
    df_price = remove_outliers(df, 'Giá_VND')
    plt.figure(figsize=(12, 6))
    sns.histplot(df_price['Giá_VND'], bins=50, kde=True, color='skyblue')
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_price))
    plt.title(f'{prefix} - Phân phối Giá Xe (Đã lọc Outlier)')
    plt.xlabel('Giá (VNĐ)')
    plt.ylabel('Số lượng')
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_1_price_dist.png", dpi=300)
    plt.close()

    # 2. Mileage Distribution
    df_km = df[df['Số Km đã đi'] > 0] # Only look at used cars for mileage distribution
    if len(df_km) > 0:
        df_km = remove_outliers(df_km, 'Số Km đã đi')
        plt.figure(figsize=(12, 6))
        sns.histplot(df_km['Số Km đã đi'], bins=50, kde=True, color='salmon')
        plt.title(f'{prefix} - Phân phối Số Km đã đi (Xe cũ)')
        plt.xlabel('Số Km')
        plt.ylabel('Số lượng')
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_1_mileage_dist.png", dpi=300)
        plt.close()

    # 3. Year Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Năm sản xuất', hue='Năm sản xuất', palette='viridis', legend=False)
    plt.title(f'{prefix} - Số lượng xe theo Năm sản xuất')
    plt.xticks(rotation=45)
    plt.xlabel('Năm sản xuất')
    plt.ylabel('Số lượng')
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_1_year_dist.png", dpi=300)
    plt.close()

def plot_categorical(df, out_dir, prefix):
    # 1. Website Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='Website', order=df['Website'].value_counts().index, hue='Website', palette='Set2', legend=False)
    plt.title(f'{prefix} - Phân phối Tin đăng theo Website')
    plt.xlabel('Số lượng')
    plt.ylabel('Website')
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_2_website_dist.png", dpi=300)
    plt.close()

    # 2. Top Models
    top_models = df['Tên xe'].value_counts().nlargest(15)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_models.values, y=top_models.index, hue=top_models.index, palette='crest', legend=False)
    plt.title(f'{prefix} - Top 15 Dòng xe Phổ biến nhất')
    plt.xlabel('Số lượng')
    plt.ylabel('Dòng xe')
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_2_top_models.png", dpi=300)
    plt.close()

    # 3. Colors (Oto only)
    if 'Màu ngoại thất' in df.columns:
        colors = df['Màu ngoại thất'].value_counts().nlargest(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=colors.values, y=colors.index, hue=colors.index, palette='flare', legend=False)
        plt.title(f'{prefix} - Top 10 Màu sắc phổ biến')
        plt.xlabel('Số lượng')
        plt.ylabel('Màu ngoại thất')
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_2_top_colors.png", dpi=300)
        plt.close()

def plot_bivariate(df, out_dir, prefix):
    # Price vs Mileage
    df_clean = remove_outliers(df, 'Giá_VND')
    df_clean = remove_outliers(df_clean, 'Số Km đã đi')
    df_used = df_clean[df_clean['Số Km đã đi'] > 0]
    if len(df_used) > 0:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df_used, x='Số Km đã đi', y='Giá_VND', alpha=0.5, color='purple')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_price))
        plt.title(f'{prefix} - Tương quan Giá xe và Số Km đã đi (Xe cũ)')
        plt.xlabel('Số Km')
        plt.ylabel('Giá (VNĐ)')
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_3_price_vs_mileage.png", dpi=300)
        plt.close()
        
    # Price vs Year (Boxplot)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_clean, x='Năm sản xuất', y='Giá_VND', hue='Năm sản xuất', palette='muted', legend=False)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_price))
    plt.title(f'{prefix} - Biến động Giá theo Năm sản xuất')
    plt.xlabel('Năm sản xuất')
    plt.ylabel('Giá (VNĐ)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_3_price_vs_year.png", dpi=300)
    plt.close()

def plot_geographical(df, out_dir, prefix):
    # Top 10 Locations
    top_loc = df['Địa chỉ'].value_counts().nlargest(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_loc.values, y=top_loc.index, hue=top_loc.index, palette='magma', legend=False)
    plt.title(f'{prefix} - Top 10 Tỉnh thành mua bán sôi động nhất')
    plt.xlabel('Số lượng')
    plt.ylabel('Tỉnh/Thành phố')
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_4_geographical.png", dpi=300)
    plt.close()

def plot_correlation(df, out_dir, prefix):
    num_cols = ['Giá_VND', 'Số Km đã đi', 'Năm sản xuất']
    df_num = df[num_cols].copy()
    
    if 'Số chỗ ngồi' in df.columns:
        df_num['Số chỗ ngồi'] = df['Số chỗ ngồi'].astype(str).str.extract(r'(\d+)').astype(float)
        
    corr = df_num.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title(f'{prefix} - Ma trận tương quan các chỉ số')
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_5_correlation.png", dpi=300)
    plt.close()

def main():
    ROOT = Path(__file__).parent.parent
    data_dir = ROOT / "data" / "processed"
    out_dir = ROOT / "reports" / "eda" / "5_deep_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Oto
    oto_path = data_dir / "ev_final_oto.csv"
    if oto_path.exists():
        print("Processing Oto...")
        df_oto = pd.read_csv(oto_path)
        plot_univariate(df_oto, out_dir, "Oto")
        plot_categorical(df_oto, out_dir, "Oto")
        plot_bivariate(df_oto, out_dir, "Oto")
        plot_geographical(df_oto, out_dir, "Oto")
        plot_correlation(df_oto, out_dir, "Oto")
        
    # Bike
    bike_path = data_dir / "ev_final_bike.csv"
    if bike_path.exists():
        print("Processing Bike...")
        df_bike = pd.read_csv(bike_path)
        plot_univariate(df_bike, out_dir, "Bike")
        plot_categorical(df_bike, out_dir, "Bike")
        plot_bivariate(df_bike, out_dir, "Bike")
        plot_geographical(df_bike, out_dir, "Bike")
        plot_correlation(df_bike, out_dir, "Bike")

if __name__ == "__main__":
    main()
