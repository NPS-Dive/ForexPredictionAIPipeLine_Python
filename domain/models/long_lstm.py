# LSTM Model for Long Timeframes (Domain Layer)
# Single Responsibility: Predict long-term trends

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

class LongLSTMModel:
    def __init__(self, input_shape: tuple = (50, 8), dropout_rate: float = 0.1):
        """Initialize LSTM model for long-term prediction."""
        self.model = self._build_model(input_shape, dropout_rate)

    def _build_model(self, input_shape: tuple, dropout_rate: float) -> Sequential:
        """Build and compile the LSTM model."""
        model = Sequential([
            LSTM(150, input_shape=input_shape, return_sequences=False),
            Dropout(dropout_rate),
            Dense(1)  # Linear output for price prediction
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
        return model

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 30) -> None:
        """Train the LSTM model."""
        self.model.fit(X, y, batch_size=16, epochs=epochs, validation_split=0.2, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X, verbose=0)