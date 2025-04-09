# Long Timeframe Features (Infrastructure Layer)
# Single Responsibility: Engineer features for daily/weekly data

import pandas as pd
import numpy as np

class LongFeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """Initialize feature engineer with price and macro data."""
        self.df = df.copy()

    def add_features(self) -> pd.DataFrame:
        """Add technical and macro indicators for long-term prediction."""
        df = self.df
        df['MA_50'] = df['Close'].rolling(50).mean()
        df['MA_200'] = df['Close'].rolling(200).mean()
        df['RSI_20'] = self._calculate_rsi(20)
        df['Trend'] = np.where(df['MA_50'] > df['MA_200'], 1, 0)
        if 'Interest_Rate' in df.columns:
            df['Interest_Diff'] = df['Interest_Rate'].diff()
        df['Target'] = df['Close'].shift(-1)  # Predict next price
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_50', 'MA_200', 'RSI_20']:
            df[f'norm_{col}'] = (df[col] - df[col].rolling(50).min()) / \
                               (df[col].rolling(50).max() - df[col].rolling(50).min() + 1e-6)
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
        features = ['norm_Open', 'norm_High', 'norm_Low', 'norm_Close', 
                   'norm_Volume', 'norm_MA_50', 'norm_MA_200', 'norm_RSI_20']
        data = self.df[features + ['Target']].values
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length, :-1])
            y.append(data[i + seq_length - 1, -1])
        return np.array(X), np.array(y)