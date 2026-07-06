# Portfolio Optimization Using Time Series Forecasting and Modern Portfolio Theory

## Project Overview

This project develops a complete end-to-end financial analytics pipeline that forecasts stock prices using time series models and constructs an optimal investment portfolio using Modern Portfolio Theory (MPT). The optimized portfolio is then validated through historical backtesting against a benchmark portfolio.

The project demonstrates the application of machine learning, deep learning, financial forecasting, and portfolio optimization techniques to support quantitative investment decision-making.

---

## Project Objectives

The project consists of five main tasks:

1. Data Preparation and Exploratory Data Analysis (EDA)
2. Time Series Forecasting using ARIMA/SARIMA and LSTM
3. Future Market Trend Forecasting
4. Portfolio Optimization using Modern Portfolio Theory
5. Portfolio Strategy Backtesting

---

## Dataset

The project uses daily historical market data for three financial assets:

| Asset | Description |
|--------|-------------|
| TSLA | Tesla Inc. Stock |
| SPY | SPDR S&P 500 ETF |
| BND | Vanguard Total Bond Market ETF |

Each dataset contains:

- Date
- Open
- High
- Low
- Close
- Adjusted Close
- Volume

Additional engineered features include:

- Daily Return
- 20-Day Rolling Mean
- 20-Day Rolling Standard Deviation
- Scaled Closing Price

---

# Project Structure

```
portfolio-optimization/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── TSLA_ARIMA.pkl
│   ├── SPY_ARIMA.pkl
│   ├── BND_ARIMA.pkl
│   ├── TSLA_LSTM.keras
│   ├── SPY_LSTM.keras
│   └── BND_LSTM.keras
│
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_Time_Series_Forecasting.ipynb
│   ├── 03_Future_Market_Forecasting.ipynb
│   ├── 04_Portfolio_Optimization.ipynb
│   └── 05_Strategy_Backtesting.ipynb
│
├── outputs/
│   ├── forecasts/
│   ├── figures/
│   ├── covariance_matrix.csv
│   ├── portfolio_summary.csv
│   ├── optimal_portfolio.csv
│   └── performance_metrics.csv
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── arima_model.py
│   ├── lstm_model.py
│   ├── evaluation.py
│   ├── forecast.py
│   ├── portfolio.py
│   ├── backtesting.py
│   ├── visualization.py
│   └── utils.py
│
├── tests/
│   ├── test_arima.py
│   ├── test_lstm.py
│   ├── test_forecast.py
│   ├── test_portfolio.py
│   └── test_backtesting.py
│
├── requirements.txt
└── README.md
```

---

# Technologies Used

- Python 3.11+
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Statsmodels
- PyPortfolioOpt
- Joblib
- Pytest

---

# Methodology

## 1. Data Preparation

The historical market data was cleaned and transformed through the following steps:

- Missing value handling
- Date formatting
- Feature engineering
- Daily return calculation
- Rolling statistics
- Data normalization using MinMaxScaler

---

## 2. Time Series Forecasting

Two forecasting approaches were implemented and compared.

### ARIMA/SARIMA

- Stationarity testing
- Grid Search parameter tuning
- Model training
- Performance evaluation
- Model persistence

Evaluation metrics:

- RMSE
- MAE
- MAPE

---

### LSTM

Deep learning forecasting included:

- Sequence generation
- Min-Max scaling
- Multi-layer LSTM architecture
- Early stopping
- Hyperparameter tuning
- Multi-step forecasting

Evaluation metrics:

- RMSE
- MAE
- MAPE

---

## 3. Future Forecasting

The best-performing forecasting model was used to generate future market forecasts.

Forecast horizon:

- 252 trading days (approximately one year)

Forecast outputs include:

- Predicted prices
- Confidence intervals
- Trend analysis
- Market opportunities
- Risk assessment

---

## 4. Portfolio Optimization

Modern Portfolio Theory (MPT) was applied using:

### Expected Returns

- TSLA expected return from forecasted prices
- SPY historical annualized return
- BND historical annualized return

### Risk Estimation

Annual covariance matrix was calculated from historical daily returns.

### Optimization

Two optimal portfolios were generated:

- Maximum Sharpe Ratio Portfolio
- Minimum Volatility Portfolio

The Efficient Frontier was simulated using thousands of randomly generated portfolios.

---

## 5. Strategy Backtesting

The optimized portfolio was evaluated using historical data.

Benchmark Portfolio:

- 60% SPY
- 40% BND

Performance metrics included:

- Total Return
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown

The cumulative returns of the optimized strategy were compared against the benchmark.

---

# Outputs

The project produces:

### Forecasts

- Future stock price predictions
- Confidence intervals

### Portfolio Optimization

- Efficient Frontier
- Covariance Heatmap
- Portfolio Allocation Pie Chart

### Backtesting

- Cumulative Return Comparison
- Drawdown Plot
- Performance Metrics

---

# How to Run the Project

## 1. Clone Repository

```bash
git clone <repository-url>
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run the Notebooks

Execute the notebooks in the following order:

1. 01_Data_Preparation.ipynb
2. 02_Time_Series_Forecasting.ipynb
3. 03_Future_Market_Forecasting.ipynb
4. 04_Portfolio_Optimization.ipynb
5. 05_Strategy_Backtesting.ipynb

---

## 4. Run Unit Tests

```bash
pytest
```

---

# Results Summary

The forecasting models successfully predicted future market trends for Tesla, SPY, and BND.

The optimized portfolio achieved improved risk-adjusted returns compared with a traditional benchmark allocation by combining forecast-driven expected returns with Modern Portfolio Theory.

Backtesting demonstrated the effectiveness of integrating machine learning forecasting with portfolio optimization while highlighting the importance of diversification and risk management.

---




