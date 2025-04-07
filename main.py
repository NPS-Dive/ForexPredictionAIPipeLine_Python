# Entry point (Presentation Layer)
from application.predict_trend import TrendPredictor
import pandas as pd

def main():
    # Initialize predictor with Dukascopy data
    predictor = TrendPredictor(symbol="EURUSD", start_date="2023-01-01", end_date="2023-12-31", sequence_length=20)
    
    # Preprocess data
    X, y, df_processed = predictor.preprocess_data()
    df_processed.to_csv("data/eur_usd_processed.csv", index=True)
    print("Data downloaded and processed. Shape:", X.shape)
    
    # Train models
    predictor.train(X, y)
    print("Models trained.")
    
    # Predict on latest data
    latest_sequence = X[-1:]  # Last 20 candles
    current_price = df_processed['Close'].iloc[-1]
    volatility = df_processed['Close'].pct_change().std() * 100  # Hourly volatility
    result = predictor.predict(latest_sequence, current_price, volatility)
    
    # Display results
    print(f"\nCNN-LSTM Prediction: {result['cnn_lstm_pred']:.5f}")
    print(f"XGBoost Refined Prediction: {result['xgb_pred']:.5f}")
    print(f"Trade Signal: {result['signal']}")
    print(f"Monte Carlo Results: {result['monte_carlo']}")

if __name__ == "__main__":
    main()