import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')
import sys

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.stdout.reconfigure(encoding='utf-8')

def evaluate_model(y_true_log, y_pred_log):
    r2 = r2_score(y_true_log, y_pred_log)
    
    # Convert back to VND for MAE and RMSE
    y_true_vnd = np.expm1(y_true_log)
    y_pred_vnd = np.expm1(y_pred_log)
    
    mae = mean_absolute_error(y_true_vnd, y_pred_vnd)
    rmse = np.sqrt(mean_squared_error(y_true_vnd, y_pred_vnd))
    
    return r2, mae, rmse

def train_and_evaluate(df, name):
    logger.info(f"--- Training models for {name} ---")
    
    # Define features
    target = 'Log_Giá_VND'
    
    # Drop irrelevant columns
    drop_cols = ['Giá_VND', target, 'Website', 'Link', 'Ngày đăng', 'Tên người bán', 'Mô tả', 'vehicle_type']
    
    if name == 'Oto':
        num_features = ['Năm sản xuất', 'Log_Số_Km', 'Số chỗ ngồi']
        high_card_cat = ['Tên xe', 'Địa chỉ', 'Màu ngoại thất']
        low_card_cat = ['Tình trạng', 'Kiểu dáng', 'Battery_Status']
    else:
        num_features = ['Năm sản xuất', 'Log_Số_Km']
        high_card_cat = ['Tên xe', 'Địa chỉ']
        low_card_cat = ['Tình trạng', 'Battery_Status']
        
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target]
    
    # Make sure we only keep the defined features
    all_features = num_features + high_card_cat + low_card_cat
    X = X[[c for c in all_features if c in X.columns]]
    
    # Fill any str NaN with 'Unknown' just in case
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Unknown').astype(str)
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [c for c in num_features if c in X.columns]),
            ('low_cat', OneHotEncoder(handle_unknown='ignore'), [c for c in low_card_cat if c in X.columns]),
            ('high_cat', TargetEncoder(smoothing=10), [c for c in high_card_cat if c in X.columns])
        ])
        
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    }
    
    results = []
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        r2, mae, rmse = evaluate_model(y_test, y_pred)
        results.append({
            'Model': model_name,
            'Dataset': name,
            'R2': r2,
            'MAE (VND)': mae,
            'RMSE (VND)': rmse
        })
        
    return pd.DataFrame(results)

def main():
    ROOT = Path(__file__).parent.parent
    data_dir = ROOT / "data" / "processed"
    reports_dir = ROOT / "reports" / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Oto
    oto_path = data_dir / "ev_model_ready_oto_v2.csv"
    if oto_path.exists():
        df_oto = pd.read_csv(oto_path)
        res_oto = train_and_evaluate(df_oto, 'Oto')
        all_results.append(res_oto)
        
    # Bike
    bike_path = data_dir / "ev_model_ready_bike_v2.csv"
    if bike_path.exists():
        df_bike = pd.read_csv(bike_path)
        res_bike = train_and_evaluate(df_bike, 'Bike')
        all_results.append(res_bike)
        
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        out_csv = reports_dir / "model_evaluation_metrics.csv"
        final_df.to_csv(out_csv, index=False)
        logger.info(f"\n{final_df}")
        logger.info(f"Metrics saved to {out_csv}")

if __name__ == "__main__":
    main()
