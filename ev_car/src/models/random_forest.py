"""
Random Forest Regression model.

Uses unscaled features + raw VND target (tree-based models are scale-invariant).
GridSearchCV tunes n_estimators, max_depth, min_samples_split, min_samples_leaf.
"""

from sklearn.ensemble import RandomForestRegressor

from .base_model import BaseModel
from .config import register_model, RANDOM_STATE


@register_model("random_forest")
class RandomForestModel(BaseModel):
    name = "random_forest"
    model_type = "unscaled_raw"

    def create_model(self):
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
