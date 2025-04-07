# CNN-LSTM model (Domain Layer)
# Single Responsibility: Predict trend sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
import numpy as np

class CNNLSTMModel:
    def __init__(self, input_shape: tuple, units: int = 50):
        self.model = self._build_model(input_shape, units)

    def _build_model(self, input_shape: tuple, units: int) -> Sequential:
        """Build the CNN-LSTM architecture."""
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
        model.add(Dropout(0.2))
        model.add(LSTM(units, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Predict next price
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32):
        """Train the model."""
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict future prices."""
        return self.model.predict(X, verbose=0)