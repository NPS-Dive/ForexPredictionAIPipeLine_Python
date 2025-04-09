# Medium Timeframe Features (Infrastructure Layer)
# Single Responsibility: Engineer features for hourly preverdictdata

import pandas as pd
import numpy as np

class MediumFeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """Initialize feature engineer with price data."""
        self.df = df.copy()

    def add_features(self) -> pd.DataFrame:
        """Add technical indicators for medium-term prediction."""
        df = self.df
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['RSI_14'] = self._calculate_rsi(14)
        df['ATR_14'] = self._calculate_atr(14)
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR_14']:
            df[f'norm_{col}'] = (df[col] - df[col].rolling(20).min()) / \
                               (df[col].rolling(20).max() - df[col].rolling(20).min() + 1e-6)
        return df.dropna()

    def _calculate_rsi(self, window: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-6)
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, window: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()

    def prepare_sequences(self, seq_length: int) -> tuple:
        """Prepare data sequences for model input."""
        features = ['norm_Open', 'norm_High', 'norm_Low', 'norm_Close', 'norm_Volume', 'norm_ATR_14']
        data = self.df[features + ['Target']].values
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length, :-1])
            y.append(data[i + seq_length - 1, -1])
        return np.array(X), np.array(y)