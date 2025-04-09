# Forex Prediction Pipeline

**GitHub Description:** Predict trends for EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CHF, and XAU/USD across timeframes from 1 minute to weekly. Uses hybrid pipelines (GRU, CNN-LSTM, LSTM + XGBoost) with Monte Carlo risk analysis. Built with Python 3.11, adhering to SOLID and Clean Architecture.

---

A modular forex prediction system with distinct pipelines for short (M1-M30), medium (H1), and long (D1-W1) timeframes, supporting multiple currency pairs and user selection via CLI.

## Features
- **Pairs:** EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CHF, XAU/USD.
- **Timeframes:** M1, M5, M15, M30, H1, D1, W1.
- **Pipelines:**
  - Short: GRU + XGBoost for rapid trades.
  - Medium: CNN-LSTM + XGBoost for hourly trends.
  - Long: LSTM + XGBoost with macro data for long-term trends.
- **Data:** Dukascopy (price), FRED (macro).
- **Architecture:** SOLID, Clean Architecture.

## Installation
1. Install Python 3.11.9: [python.org](https://www.python.org/downloads/release/python-3119/)
2. Clone: `git clone https://github.com/NPS-Dive/ForexPredictionAIPipeLine_Python.git`
3. Virtual Env: `python -m venv venv; .\venv\Scripts\Activate.ps1`
4. Install: `pip install -r requirements.txt`

## Usage
Run: `python main.py`
- Enter start and end dates (or press Enter for defaults).
- Select a pair and timeframe from the menu.

## Structure
- `data/`: Stores raw and processed data by timeframe.
- `domain/`: Pure business logic (models, signals).
- `application/`: Use cases (predictors).
- `infrastructure/`: External interactions (data, features).
- `presentation/`: CLI for user input.
- `config.py`: Centralized configuration.

## License
MIT License

## Full Map and Path of the Solution
- [Project Map](MAP.md): Detailed directory structure and file purposes.