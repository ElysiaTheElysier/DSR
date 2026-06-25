"""
LightGBM Regression model.

Uses unscaled features + raw VND target (tree-based models are scale-invariant).
GridSearchCV tunes n_estimators, max_depth, learning_rate, subsample, colsample_bytree, num_leaves.
"""

from lightgbm import LGBMRegressor

from .base_model import BaseModel
from .config import register_model, RANDOM_STATE


@register_model("lightgbm")
class LightGBMModel(BaseModel):
    name = "lightgbm"
    model_type = "unscaled_raw"

    def create_model(self):
        return LGBMRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
