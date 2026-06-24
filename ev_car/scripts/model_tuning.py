import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')
import sys

from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
from scipy.stats import loguniform, randint, uniform

sys.stdout.reconfigure(encoding='utf-8')

def evaluate_model(y_true_log, y_pred_log):
    r2 = r2_score(y_true_log, y_pred_log)
    
    # Convert back to VND for MAE and RMSE
    y_true_vnd = np.expm1(y_true_log)
    y_pred_vnd = np.expm1(y_pred_log)
    
    mae = mean_absolute_error(y_true_vnd, y_pred_vnd)
    rmse = np.sqrt(mean_squared_error(y_true_vnd, y_pred_vnd))
    
    return r2, mae, rmse

def get_param_grid(model_name):
    """Return the hyperparameter search space for a given model."""
    if model_name == 'Ridge Regression':
        return {
            'regressor__alpha': loguniform(0.1, 100.0)
        }
    elif model_name == 'SVR':
        return {
            'regressor__C': loguniform(0.1, 100.0),
            'regressor__gamma': ['scale', 'auto', 0.1, 0.01],
            'regressor__epsilon': uniform(0.01, 0.99) # uniform(loc, scale) -> [0.01, 1.0]
        }
    elif model_name == 'Random Forest':
        return {
            'regressor__max_depth': randint(3, 11),
            'regressor__min_samples_split': randint(5, 21),
            'regressor__min_samples_leaf': randint(5, 21)
        }
    elif model_name == 'XGBoost':
        return {
            'regressor__max_depth': randint(3, 8),
            'regressor__learning_rate': loguniform(0.01, 0.2),
            'regressor__min_child_weight': randint(5, 21),
            'regressor__subsample': uniform(0.6, 0.3), # [0.6, 0.9]
            'regressor__colsample_bytree': uniform(0.6, 0.3)
        }
    elif model_name == 'LightGBM':
        return {
            'regressor__max_depth': randint(3, 8),
            'regressor__num_leaves': randint(10, 32),
            'regressor__learning_rate': loguniform(0.01, 0.2),
            'regressor__min_child_samples': randint(20, 51),
            'regressor__subsample': uniform(0.6, 0.3)
        }
    return {}

def train_and_tune(df, name):
    logger.info(f"--- Hyperparameter Tuning for {name} ---")
    
    # Check actual column names
    col_mapping = {
        'Người bán': 'Tên người bán',
        'Giá_VNĐ': 'Giá_VND',
        'Log_Giá_VNĐ': 'Log_Giá_VND'
    }
    for old, new in col_mapping.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
    
    target = 'Log_Giá_VND'
    
    # 1. Feature Selection based on user rules
    drop_cols = ['Website', 'Link', 'Ngày đăng', 'Tên người bán', 'Người bán', 
                 'Mô tả', 'vehicle_type', 'Giá_VND', 'Giá_VNĐ', 'Số Km đã đi', 'Battery_Status', target]
                 
    if name == 'Oto':
        num_features = ['Năm sản xuất', 'Log_Số_Km', 'Số chỗ ngồi']
        low_card_cat = ['Tình trạng', 'Kiểu dáng']
        high_card_cat = ['Tên xe', 'Địa chỉ', 'Màu ngoại thất']
    else:
        num_features = ['Năm sản xuất', 'Log_Số_Km']
        low_card_cat = ['Tình trạng']
        high_card_cat = ['Tên xe', 'Địa chỉ']
        
    # Drop irrelevant columns from X
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target]
    
    # Ensure strict adherence to defined features
    all_features = num_features + high_card_cat + low_card_cat
    X = X[[c for c in all_features if c in X.columns]]
    
    # Fill NaN for categorical
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Unknown').astype(str)
            
    # Train test split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [c for c in num_features if c in X.columns]),
            ('low_cat', OneHotEncoder(handle_unknown='ignore'), [c for c in low_card_cat if c in X.columns]),
            ('high_cat', TargetEncoder(smoothing=10), [c for c in high_card_cat if c in X.columns])
        ])
        
    models = {
        'Ridge Regression': Ridge(random_state=42),
        'SVR': SVR(),
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(random_state=42, n_jobs=-1),
        'LightGBM': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    }
    
    results = []
    best_pipelines = {}
    
    for model_name, model in models.items():
        logger.info(f"Tuning {model_name}...")
        
        # Pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        param_grid = get_param_grid(model_name)
        
        # RandomizedSearchCV
        search = RandomizedSearchCV(
            pipeline, 
            param_distributions=param_grid,
            n_iter=15, # 15 iterations per model to save time but find good params
            cv=3,
            scoring='r2',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        
        best_pipeline = search.best_estimator_
        best_pipelines[model_name] = best_pipeline
        
        logger.info(f"Best params for {model_name}: {search.best_params_}")
        
        # Evaluate on Test Set
        y_pred = best_pipeline.predict(X_test)
        r2, mae, rmse = evaluate_model(y_test, y_pred)
        
        results.append({
            'Model': model_name,
            'Dataset': name,
            'R2': r2,
            'MAE (VND)': mae,
            'RMSE (VND)': rmse
        })
        
    # Return sorted by R2 descending
    res_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    return res_df

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
        res_oto = train_and_tune(df_oto, 'Oto')
        all_results.append(res_oto)
        
    # Bike
    bike_path = data_dir / "ev_model_ready_bike_v2.csv"
    if bike_path.exists():
        df_bike = pd.read_csv(bike_path)
        res_bike = train_and_tune(df_bike, 'Bike')
        all_results.append(res_bike)
        
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        out_csv = reports_dir / "model_tuning_metrics.csv"
        final_df.to_csv(out_csv, index=False)
        logger.info(f"\n{final_df}")
        logger.info(f"Tuned Metrics saved to {out_csv}")

if __name__ == "__main__":
    main()
