# Medium Timeframe Predictor (Application Layer)
# Single Responsibility: Orchestrate hourly prediction

from domain.models.medium_cnn_lstm import MediumCNNLSTMModel
from domain.models.xgboost_refiner import XGBoostRefiner
from domain.monte_carlo import MonteCarloSimulator
from domain.signals import SignalGenerator
from infrastructure.data_loaders.duka_loader import DukaLoader
from infrastructure.feature_engineers.medium_features import MediumFeatureEngineer
import numpy as np

class MediumTrendPredictor:
    def __init__(self, symbol: str, start_date: str, end_date: str, timeframe: str):
        """Initialize medium-term predictor."""
        self.symbol = symbol
        self.data_loader = DukaLoader(symbol, start_date, end_date, timeframe)
        self.model = MediumCNNLSTMModel((20, 6))
        self.refiner = XGBoostRefiner()
        self.monte_carlo = MonteCarloSimulator()
        self.signals = SignalGenerator()

    def preprocess_data(self) -> tuple:
        """Load and preprocess data."""
        df = self.data_loader.load_data()
        if df.empty:
            raise ValueError(f"No data loaded for {self.symbol} on {self.data_loader.timeframe}")
        engineer = MediumFeatureEngineer(df)
        df_processed = engineer.add_features()
        X, y = engineer.prepare_sequences(20)
        return X, y, df_processed

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the models."""
        self.model.train(X, y)
        preds = self.model.predict(X)
        X_refined = np.hstack([preds, X[:, -1, :]])
        self.refiner.train(X_refined, y)

    def predict(self, X_new: np.ndarray, price: float, volatility: float) -> dict:
        """Generate prediction and trade signal."""
        pred = self.model.predict(X_new)[0]
        X_refined = np.hstack([pred, X_new[0, -1, :]])
        refined_pred = self.refiner.predict(X_refined.reshape(1, -1))[0]
        signal = self.signals.generate_medium(refined_pred, price)
        mc_results = self.monte_carlo.simulate(refined_pred, volatility)
        return {'pred': float(refined_pred), 'signal': signal, 'monte_carlo': mc_results}