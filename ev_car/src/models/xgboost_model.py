"""
XGBoost Regression model.

Uses unscaled features + raw VND target (tree-based models are scale-invariant).
GridSearchCV tunes n_estimators, max_depth, learning_rate, subsample, colsample_bytree.
"""

from xgboost import XGBRegressor

from .base_model import BaseModel
from .config import register_model, RANDOM_STATE


@register_model("xgboost")
class XGBoostModel(BaseModel):
    name = "xgboost"
    model_type = "unscaled_raw"

    def create_model(self):
        return XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
