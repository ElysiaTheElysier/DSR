"""
Support Vector Regression (RBF kernel) — full implementation.

Uses scaled features + log-transformed target.
GridSearchCV tunes C, epsilon, and gamma.
"""

from sklearn.svm import SVR

from .base_model import BaseModel
from .config import register_model


@register_model("svr")
class SVRModel(BaseModel):
    name = "svr"
    model_type = "scaled_log"

    def create_model(self):
        return SVR(kernel="rbf")
