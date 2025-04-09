# XGBoost Refiner (Domain Layer)
# Single Responsibility: Refine predictions across all timeframes

import xgboost as xgb
import numpy as np

class XGBoostRefiner:
    def __init__(self, is_classifier: bool = True):
        """Initialize XGBoost model for refining predictions."""
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        self.model = xgb.XGBClassifier(**params) if is_classifier else xgb.XGBRegressor(**params)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost model."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate refined predictions."""
        if isinstance(self.model, xgb.XGBClassifier):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)