# CNN-LSTM Model for Medium Timeframes (Domain Layer)
# Single Responsibility: Predict hourly trends

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import numpy as np

class MediumCNNLSTMModel:
    def __init__(self, input_shape: tuple = (20, 6), dropout_rate: float = 0.2):
        """Initialize CNN-LSTM model for medium-term prediction."""
        self.model = self._build_model(input_shape, dropout_rate)

    def _build_model(self, input_shape: tuple, dropout_rate: float) -> Sequential:
        """Build and compile the CNN-LSTM model."""
        model = Sequential([
            Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(dropout_rate),
            LSTM(100, return_sequences=False),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        return model

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50) -> None:
        """Train the CNN-LSTM model."""
        self.model.fit(X, y, batch_size=32, epochs=epochs, validation_split=0.2, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X, verbose=0)