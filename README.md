# AI-Powered Stock Market Analysis & Prediction Dashboard

An interactive web application built with Python and Streamlit that helps users analyze stocks, generate trading signals, backtest strategies, and forecast short-term prices.

![Dashboard Screenshot](https://via.placeholder.com/800x400?text=Stock+Dashboard+Screenshot)  
*(Replace this with a real screenshot later – take one while running the app locally)*

## Features
- Real-time historical stock data from Yahoo Finance (supports NSE `.NS` stocks, US stocks, etc.)
- Key technical indicators: SMA (20/50/200), RSI (14), MACD (12,26,9), Bollinger Bands (20,2)
- Rule-based buy/sell signals (SMA crossover + RSI filter)
- Strategy backtesting with equity curve vs buy-and-hold benchmark
- 30-day price forecast using ARIMA time-series model
- Clean, responsive Streamlit interface with customizable inputs (ticker, date range, chart zoom)

## Technologies Used
- **Python** 3
- **Streamlit** – for the web interface
- **yfinance** – stock data
- **pandas, numpy** – data processing
- **matplotlib** – charting
- **statsmodels** – ARIMA forecasting

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-ai-predictor.git
   cd stock-ai-predictor