# Feature engineering (Infrastructure Layer)
# Single Responsibility: Add technical indicators
import pandas as pd
import numpy as np
import talib

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def add_features(self) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        self.df['RSI'] = talib.RSI(self.df['Close'], timeperiod=14)
        self.df['MACD'], self.df['MACD_signal'], _ = talib.MACD(self.df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        self.df['BB_upper'], self.df['BB_middle'], self.df['BB_lower'] = talib.BBANDS(
            self.df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        return self.df.dropna()

    def prepare_sequences(self, sequence_length: int, target_col: str = 'Close') -> tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for CNN-LSTM input."""
        data = self.df[['Open', 'High', 'Low', 'Close', 'RSI', 'MACD']].values
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length, 3])  # Predict 'Close'
        return np.array(X), np.array(y)