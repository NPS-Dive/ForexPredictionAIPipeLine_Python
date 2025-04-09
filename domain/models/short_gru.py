# GRU Model for Short Timeframes (Domain Layer)
# Single Responsibility: Predict rapid price movements

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

class ShortGRUModel:
    def __init__(self, input_shape: tuple = (10, 4), dropout_rate: float = 0.3):
        """Initialize GRU model for short-term prediction."""
        self.model = self._build_model(input_shape, dropout_rate)

    def _build_model(self, input_shape: tuple, dropout_rate: float) -> Sequential:
        """Build and compile the GRU model."""
        model = Sequential([
            GRU(50, input_shape=input_shape, return_sequences=False),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')  # Binary up/down prediction
        ])
        model.compile(optimizer=Adam(learning_rate=0.01), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        return model

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 20) -> None:
        """Train the GRU model."""
        self.model.fit(X, y, batch_size=64, epochs=epochs, validation_split=0.2, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X, verbose=0)