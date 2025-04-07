# Deep Learning Model (Domain Layer)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class CNNLSTMModel:
    def __init__(self, input_shape: tuple, dropout_rate: float = 0.2):
        """Initialize CNN-LSTM hybrid model."""
        self.model = self._build_model(input_shape, dropout_rate)
    
    def _build_model(self, input_shape: tuple, dropout_rate: float) -> Sequential:
        """Create a CNN-LSTM hybrid for time series feature extraction and prediction."""
        model = Sequential([
            # CNN layers for feature extraction
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(dropout_rate),
            
            Conv1D(filters=128, kernel_size=2, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(dropout_rate),
            
            # LSTM layer for sequence learning
            LSTM(units=100, return_sequences=False),
            Dropout(dropout_rate),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              batch_size: int = 32, epochs: int = 50, 
              validation_split: float = 0.2) -> None:
        """Train the model with early stopping."""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X, verbose=0)