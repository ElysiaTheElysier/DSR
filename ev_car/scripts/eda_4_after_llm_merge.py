import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil

def plot_missing_data(df: pd.DataFrame, title: str, output_path: Path):
    plt.figure(figsize=(12, 8))
    missing = df.isnull().sum()
    missing_perc = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_perc})
    missing_df = missing_df.sort_values(by='Percentage', ascending=False)
    
    # Filter out columns that have 0 missing data
    missing_df = missing_df[missing_df['Percentage'] > 0]
    
    if len(missing_df) == 0:
        plt.text(0.5, 0.5, '0% Missing Data!', fontsize=20, ha='center')
        plt.title(title)
        plt.axis('off')
    else:
        sns.barplot(x=missing_df['Percentage'], y=missing_df.index, hue=missing_df.index, legend=False, palette='Reds_r')
        plt.title(title)
        plt.xlabel('Percentage Missing (%)')
        plt.ylabel('Columns')
        for i, v in enumerate(missing_df['Percentage']):
            plt.text(v + 0.5, i, f'{v:.1f}% ({int(missing_df.iloc[i]["Missing"])} rows)', va='center')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    ROOT = Path(__file__).parent.parent
    data_dir = ROOT / "data" / "interim"
    reports_dir = ROOT / "reports" / "eda" / "4_after_llm_merge"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Oto
    oto_path = data_dir / "ev_cleaned_v2_oto.csv"
    if oto_path.exists():
        df_oto = pd.read_csv(oto_path)
        plot_missing_data(df_oto, "Missing Data Post LLM Merge - Oto", reports_dir / "oto_missing_data_v2.png")
        try:
            shutil.copy(oto_path, reports_dir / "ev_cleaned_v2_oto.csv")
        except PermissionError:
            print("Cannot copy oto file due to PermissionError")
            
    # Bike
    bike_path = data_dir / "ev_cleaned_v2_bike.csv"
    if bike_path.exists():
        df_bike = pd.read_csv(bike_path)
        plot_missing_data(df_bike, "Missing Data Post LLM Merge - Bike", reports_dir / "bike_missing_data_v2.png")
        try:
            shutil.copy(bike_path, reports_dir / "ev_cleaned_v2_bike.csv")
        except PermissionError:
            print("Cannot copy bike file due to PermissionError")

if __name__ == "__main__":
    main()
