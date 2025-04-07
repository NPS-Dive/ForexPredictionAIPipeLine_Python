# XGBoost model (Domain Layer)
# Single Responsibility: Refine predictions into trade signals
import xgboost as xgb
import numpy as np

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the XGBoost model."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict refined price or signal."""
        return self.model.predict(X)

    def generate_signals(self, predictions: np.ndarray, current_price: float, threshold: float = 0.001) -> str:
        """Generate buy/sell signals based on predicted price movement."""
        predicted_change = (predictions[0] - current_price) / current_price
        if predicted_change > threshold:
            return "Buy"
        elif predicted_change < -threshold:
            return "Sell"
        return "Hold"