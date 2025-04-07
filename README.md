# Forex Prediction Pipeline

**GitHub Description**: Predict EUR/USD trends using a CNN-LSTM + XGBoost model with Monte Carlo risk analysis. Fetches historical data from Dukascopy, applies technical indicators, and generates trade signals. Built with Python 3.11, TensorFlow, and Clean Architecture.

---

A machine learning pipeline to predict EUR/USD forex trends and generate trade signals using historical data from Dukascopy. This project combines a hybrid CNN-LSTM model, XGBoost refinement, and Monte Carlo simulation for robust predictions and risk assessment.

## Features
- **Data Source**: Fetches 1-hour EUR/USD data from Dukascopy via `duka`.
- **Preprocessing**: Adds technical indicators (RSI, MACD, Bollinger Bands) using `ta-lib`.
- **Model**: 
  - CNN-LSTM for trend prediction.
  - XGBoost for refining predictions and generating Buy/Sell/Hold signals.
- **Risk Analysis**: Monte Carlo simulation for probability estimates.
- **Architecture**: Follows SOLID principles and Clean Architecture.

## Project Structure
```
forex_prediction/
├── data/                    # Raw and processed data
│   ├── eur_usd_raw.csv      # Downloaded from Dukascopy
│   └── eur_usd_processed.csv # Processed with features
├── domain/                  # Business logic
│   ├── cnn_lstm.py          # CNN-LSTM model
│   ├── xgboost_model.py     # XGBoost refinement
│   └── monte_carlo.py       # Monte Carlo simulation
├── application/             # Use cases
│   └── predict_trend.py     # Main pipeline logic
├── infrastructure/          # External interactions
│   ├── data_loader.py       # Dukascopy data fetching
│   └── feature_engineer.py  # Feature extraction
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Prerequisites
- **Python**: 3.11.9 (recommended for TensorFlow compatibility).
- **Windows**: Tested on Windows with PowerShell.
- **Dependencies**: Listed in `requirements.txt`.

## Installation
1. **Install Python 3.11.9**:
   - Download from [python.org](https://www.python.org/downloads/release/python-3119/).
   - Install with "Add Python to PATH" checked.

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/forex_prediction.git
   cd forex_prediction
   ```

3. **Set Up Virtual Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

4. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
   - If `ta-lib` fails:
     - Download `TA_Lib-0.4.28-cp311-cp311-win_amd64.whl` from [GitHub](https://github.com/mrjbq7/ta-lib).
     - Install: `pip install path\to\TA_Lib-0.4.28-cp311-cp311-win_amd64.whl`.

5. **Verify TensorFlow**:
   ```powershell
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

## Usage
1. **Run the Pipeline**:
   ```powershell
   python main.py
   ```
   - Downloads EUR/USD data for 2023.
   - Trains models and outputs predictions, trade signals, and Monte Carlo results.

2. **Example Output**:
   ```
   Data downloaded and processed. Shape: (..., 20, 6)
   Models trained.
   CNN-LSTM Prediction: 1.12345
   XGBoost Refined Prediction: 1.12350
   Trade Signal: Buy
   Monte Carlo Results: {'median': 1.12348, 'lower_bound': 1.12000, 'upper_bound': 1.12600, 'success_prob': 0.65}
   ```

## Customization
- **Date Range**: Edit `start_date` and `end_date` in `predict_trend.py`.
- **Symbol**: Change `symbol` in `predict_trend.py` (e.g., "GBPUSD").
- **Model Tuning**: Adjust `epochs`, `sequence_length`, or thresholds in respective files.

## Troubleshooting
- **TensorFlow Warnings**: Normal optimization messages. Suppress with:
  ```powershell
  $env:TF_ENABLE_ONEDNN_OPTS = "0"
  ```
- **Duka Errors**: Ensure internet access and correct `data` folder permissions.
- **TA-Lib**: Install manually if `pip` fails.

## Dependencies
- `pandas`: Data manipulation.
- `numpy`: Numerical operations.
- `tensorflow`: CNN-LSTM model.
- `xgboost`: Prediction refinement.
- `scikit-learn`: Utilities.
- `ta-lib`: Technical indicators.
- `duka`: Dukascopy data fetcher.

## Contributing
Feel free to fork, submit issues, or PRs to enhance the pipeline (e.g., backtesting, real-time data).

## License
MIT License - free to use and modify.

---
Created on April 7, 2025
```



