# XGBoost Model (Domain Layer)
import numpy as np
import xgboost as xgb

class XGBoostModel:
    def __init__(self):
        """Initialize XGBoost model for refining predictions."""
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            seed=42
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost model."""
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions."""
        return self.model.predict_proba(X)[:, 1]  # Return probability of positive class
    
    def generate_signals(self, predictions: list, current_price: float) -> str:
        """Convert predictions to trading signals."""
        pred = predictions[0]
        
        if pred > 0.7:
            return f"BUY @ {current_price:.5f}"
        elif pred < 0.3:
            return f"SELL @ {current_price:.5f}"
        else:
            return "HOLD"