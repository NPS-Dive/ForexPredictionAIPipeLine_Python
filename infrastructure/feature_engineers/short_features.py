# Short Timeframe Features (Infrastructure Layer)
# Single Responsibility: Engineer features for short-term data

import pandas as pd
import numpy as np

class ShortFeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """Initialize feature engineer with price data."""
        self.df = df.copy()

    def add_features(self) -> pd.DataFrame:
        """Add technical indicators for short-term prediction."""
        df = self.df
        df['EMA_3'] = df['Close'].ewm(span=3).mean()
        df['EMA_8'] = df['Close'].ewm(span=8).mean()
        df['RSI_5'] = self._calculate_rsi(5)
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        for col in ['Open', 'High', 'Low', 'Close']:
            df[f'norm_{col}'] = (df[col] - df[col].rolling(10).min()) / \
                               (df[col].rolling(10).max() - df[col].rolling(10).min() + 1e-6)
        return df.dropna()

    def _calculate_rsi(self, window: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-6)
        return 100 - (100 / (1 + rs))

    def prepare_sequences(self, seq_length: int) -> tuple:
        """Prepare data sequences for model input."""
        features = ['norm_Open', 'norm_High', 'norm_Low', 'norm_Close']
        data = self.df[features + ['Target']].values
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length, :-1])
            y.append(data[i + seq_length - 1, -1])
        return np.array(X), np.array(y)