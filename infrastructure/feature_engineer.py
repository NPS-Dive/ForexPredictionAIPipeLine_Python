# Feature engineering (Infrastructure Layer)
import pandas as pd
import numpy as np
from typing import Tuple

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """Initialize with raw OHLCV dataframe."""
        self.df = df.copy()
    
    def add_features(self) -> pd.DataFrame:
        """Add technical indicators and features."""
        df = self.df
        
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_Ratio'] = df['MA_5'] / df['MA_20'] 
        df['ATR'] = self._calculate_atr(df)
        
        # Momentum
        df['RSI'] = self._calculate_rsi(df, 14)
        
        # Target variable (whether price will go up in next period)
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        
        # Normalize features for deep learning
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR']:
            df[f'norm_{col}'] = self._normalize(df[col])
        
        # Return only complete rows
        return df.dropna()
    
    def _normalize(self, series: pd.Series) -> pd.Series:
        """Min-max normalization."""
        return (series - series.rolling(window=20).min()) / (series.rolling(window=20).max() - series.rolling(window=20).min() + 1e-6)
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=window).mean()
    
    def _calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_sequences(self, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        # Select features to use
        features = ['norm_Open', 'norm_High', 'norm_Low', 'norm_Close', 'norm_Volume', 'norm_ATR']
        data = self.df[features + ['Target']].values
        
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length, :-1])  # All columns except Target
            y.append(data[i+seq_length-1, -1])   # Target column
            
        return np.array(X), np.array(y)