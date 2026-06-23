"""
Linear Regression (Ridge) — full implementation.

Uses scaled features + log-transformed target.
GridSearchCV tunes the regularization strength (alpha).
"""

from sklearn.linear_model import Ridge

from .base_model import BaseModel
from .config import register_model


@register_model("linear_regression")
class LinearRegressionModel(BaseModel):
    name = "linear_regression"
    model_type = "scaled_log"

    def create_model(self):
        return Ridge(random_state=42)
