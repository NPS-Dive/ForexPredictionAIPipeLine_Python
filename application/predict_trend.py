# Orchestration of the pipeline (Application Layer)
# Single Responsibility: Coordinate the prediction workflow
from domain.cnn_lstm import CNNLSTMModel
from domain.xgboost_model import XGBoostModel
from domain.monte_carlo import MonteCarloSimulator
from infrastructure.data_loader import DataLoader
from infrastructure.feature_engineer import FeatureEngineer
import pandas as pd
import numpy as np

class TrendPredictor:
    def __init__(self, symbol: str = "EURUSD", start_date: str = "2023-01-01", end_date: str = "2023-12-31", sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.data_loader = DataLoader(symbol, start_date, end_date)
        self.cnn_lstm = CNNLSTMModel(input_shape=(sequence_length, 6))  # 6 features
        self.xgboost = XGBoostModel()
        self.monte_carlo = MonteCarloSimulator()

    def preprocess_data(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Load and prepare data from Dukascopy."""
        df = self.data_loader.load_data()
        engineer = FeatureEngineer(df)
        df_processed = engineer.add_features()
        X, y = engineer.prepare_sequences(self.sequence_length)
        return X, y, df_processed

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the models."""
        self.cnn_lstm.train(X, y)
        cnn_lstm_preds = self.cnn_lstm.predict(X)
        X_xgb = np.hstack([cnn_lstm_preds, X[:, -1, :]])  # Combine predictions with last sequence
        self.xgboost.train(X_xgb, y)

    def predict(self, X_new: np.ndarray, current_price: float, volatility: float) -> dict:
        """Run full prediction pipeline."""
        cnn_lstm_pred = self.cnn_lstm.predict(X_new)[0]
        X_xgb_new = np.hstack([cnn_lstm_pred, X_new[0, -1, :]])
        xgb_pred = self.xgboost.predict(X_xgb_new.reshape(1, -1))[0]
        signal = self.xgboost.generate_signals([xgb_pred], current_price)
        mc_results = self.monte_carlo.simulate(xgb_pred, volatility)
        
        return {
            'cnn_lstm_pred': float(cnn_lstm_pred),
            'xgb_pred': float(xgb_pred),
            'signal': signal,
            'monte_carlo': mc_results
        }