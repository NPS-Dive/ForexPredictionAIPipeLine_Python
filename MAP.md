# Forex Prediction Pipeline - Project Map

This document outlines the directory structure and file purposes for the Forex Prediction Pipeline project, built with Python 3.11 following SOLID principles and Clean Architecture.
forex_prediction/
├── data/                    # Directory for raw and processed data (auto-generated)
│   ├── m1/                  # M1 timeframe data (e.g., eurusd_m1_raw.csv, eurusd_m1_processed.csv)
│   ├── m5/                  # M5 timeframe data
│   ├── m15/                 # M15 timeframe data
│   ├── m30/                 # M30 timeframe data
│   ├── h1/                  # H1 timeframe data
│   ├── d1/                  # D1 timeframe data
│   └── w1/                  # W1 timeframe data
├── domain/                  # Business logic layer (pure, dependency-free)
│   ├── models/              # Machine learning models
│   │   ├── short_gru.py     # GRU model for short-term (M1-M30) predictions
│   │   ├── medium_cnn_lstm.py  # CNN-LSTM model for medium-term (H1) predictions
│   │   ├── long_lstm.py     # LSTM model for long-term (D1-W1) predictions
│   │   └── xgboost_refiner.py  # XGBoost model to refine predictions across all timeframes
│   ├── monte_carlo.py       # Monte Carlo simulation for risk assessment
│   └── signals.py           # Logic to convert predictions into trade signals (BUY/SELL/HOLD)
├── application/             # Use case layer (orchestrates domain logic)
│   ├── short_predictor.py   # Orchestrates short-term prediction pipeline
│   ├── medium_predictor.py  # Orchestrates medium-term prediction pipeline
│   └── long_predictor.py    # Orchestrates long-term prediction pipeline
├── infrastructure/          # External interactions layer (data, features, utilities)
│   ├── data_loaders/        # Data fetching modules
│   │   ├── duka_loader.py   # Loads price data from Dukascopy
│   │   └── macro_loader.py  # Loads macro-economic data from FRED
│   ├── feature_engineers/   # Feature engineering for different timeframes
│   │   ├── short_features.py  # Engineers features for short-term data (e.g., EMA, RSI)
│   │   ├── medium_features.py # Engineers features for medium-term data (e.g., MA, ATR)
│   │   └── long_features.py   # Engineers features for long-term data (e.g., MA, macro)
│   └── utils.py             # Reusable utility functions (e.g., normalization)
├── presentation/            # User interface layer
│   └── cli.py               # Command-line interface for user interaction
├── config.py                # Centralized configuration (pairs, timeframes, defaults)
├── main.py                  # Application entry point (launches CLI)
├── requirements.txt         # Python dependencies list
└── README.md                # Project documentation and setup instructions

## Directory and File Purposes

- **`data/`**: Auto-generated folder storing raw and processed data, organized by timeframe (e.g., `m1/`, `h1/`). Files include raw CSV from Dukascopy and processed CSV with features.
- **`domain/`**: Contains pure business logic, independent of external systems.
  - **`models/`**: Defines machine learning models tailored to different timeframes.
  - **`monte_carlo.py`**: Simulates price movements to assess risk/reward.
  - **`signals.py`**: Generates actionable trade signals from predictions.
- **`application/`**: Implements use cases by coordinating domain models, data loading, and feature engineering.
- **`infrastructure/`**: Handles interactions with external systems (data sources, file I/O) and feature preprocessing.
  - **`data_loaders/`**: Fetches raw data from Dukascopy and FRED.
  - **`feature_engineers/`**: Prepares data with technical and macro indicators.
  - **`utils.py`**: Provides helper functions used across the project.
- **`presentation/`**: Manages user interaction via a CLI, displaying predictions and signals.
- **`config.py`**: Centralizes constants like currency pairs and timeframes for easy maintenance.
- **`main.py`**: Entry point to start the application.
- **`requirements.txt`**: Lists Python packages required to run the project.
- **`README.md`**: Guides users on installation, usage, and project overview.

This structure ensures modularity, scalability, and adherence to Clean Architecture principles.