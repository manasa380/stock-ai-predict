import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock AI Predictor", layout="wide")

st.title("📈 AI Stock Market Prediction & Trading Signals")
st.markdown("Enter a stock symbol → get charts, indicators, backtest and short-term forecast. Not financial advice — for education only.")

# ────────────────────────────────────────────────
# Inputs
# ────────────────────────────────────────────────
col1, col2, col3 = st.columns([3, 3, 2])

with col1:
    ticker = st.text_input("Stock Ticker", value="RELIANCE.NS").strip().upper()

with col2:
    start_date = st.text_input("Start Date (YYYY-MM-DD)", value="2020-01-01")

with col3:
    days_to_show = st.slider("Days to display in main charts", 100, 1500, 500, help="Zoom level for price & indicator charts")

if not ticker:
    st.warning("Please enter a ticker symbol (example: RELIANCE.NS, AAPL, TSLA)")
    st.stop()

if st.button("Analyze Stock", type="primary"):

    with st.spinner(f"Downloading {ticker} data from {start_date} ..."):

        try:
            data = yf.download(ticker, start=start_date, progress=False)
            if data.empty:
                st.error(f"No data found for {ticker}. Check symbol or try a different date.")
                st.stop()

            df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']

            latest_date = df.index[-1].date()
            latest_price = df['close'].iloc[-1]
            st.success(f"Loaded {len(df)} trading days | Latest: {latest_date} @ ₹{latest_price:,.2f}" if ".NS" in ticker else f"${latest_price:,.2f}")

            # ────────────────────────────────────────────────
            # Technical Indicators (your functions)
            # ────────────────────────────────────────────────
            def rsi(series, period=14):
                delta = series.diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            def macd(series, fast=12, slow=26, signal=9):
                ema_fast = series.ewm(span=fast, adjust=False).mean()
                ema_slow = series.ewm(span=slow, adjust=False).mean()
                macd_line = ema_fast - ema_slow
                sig_line = macd_line.ewm(span=signal, adjust=False).mean()
                hist = macd_line - sig_line
                return macd_line, sig_line, hist

            def bollinger(series, window=20, std=2):
                mid = series.rolling(window).mean()
                std_dev = series.rolling(window).std()
                return mid + std*std_dev, mid, mid - std*std_dev

            df['SMA20']  = df['close'].rolling(20).mean()
            df['SMA50']  = df['close'].rolling(50).mean()
            df['SMA200'] = df['close'].rolling(200).mean()
            df['RSI']    = rsi(df['close'])
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd(df['close'])
            df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = bollinger(df['close'])

            # ────────────────────────────────────────────────
            # Trading Signals (your logic)
            # ────────────────────────────────────────────────
            df['Signal']    = 0
            df['Position']  = np.where(df['SMA50'] > df['SMA200'], 1, 0)
            df['Crossover'] = df['Position'].diff()

            for i in range(1, len(df)):
                if df['Crossover'].iloc[i] == 1 and df['RSI'].iloc[i] < 70:
                    df.loc[df.index[i], 'Signal'] = 1
                elif df['Crossover'].iloc[i] == -1 or df['RSI'].iloc[i] > 65:
                    df.loc[df.index[i], 'Signal'] = -1

            total_signals = df['Signal'].abs().sum()
            st.write(f"**Detected {total_signals} trading signals**")

            # ────────────────────────────────────────────────
            # Backtesting
            # ────────────────────────────────────────────────
            capital = 10000.0
            position = 0
            shares = 0
            equity = [capital]

            for i in range(len(df)):
                if df['Signal'].iloc[i] == 1 and position == 0:
                    shares = capital / df['close'].iloc[i]
                    position = 1
                elif df['Signal'].iloc[i] == -1 and position == 1:
                    capital = shares * df['close'].iloc[i]
                    position = 0
                    shares = 0
                current = capital if position == 0 else shares * df['close'].iloc[i]
                equity.append(current)

            df['Equity'] = equity[1:]

            strat_return  = ((equity[-1] / 10000) - 1) * 100
            buyhold_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100

            st.metric("Strategy Return", f"{strat_return:.1f}%", delta=None)
            st.metric("Buy & Hold Return", f"{buyhold_return:.1f}%", delta=None)

            # ────────────────────────────────────────────────
            # Charts
            # ────────────────────────────────────────────────
            df_show = df.tail(days_to_show)

            st.subheader("Price + SMAs + Bollinger Bands")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(df_show.index, df_show['close'], label='Close', color='blue', lw=1.8)
            ax1.plot(df_show.index, df_show['SMA50'],  label='SMA50',  color='orange')
            ax1.plot(df_show.index, df_show['SMA200'], label='SMA200', color='red')
            ax1.plot(df_show.index, df_show['BB_Upper'], 'g--', alpha=0.5, label='BB Upper')
            ax1.plot(df_show.index, df_show['BB_Lower'], 'g--', alpha=0.5, label='BB Lower')
            ax1.fill_between(df_show.index, df_show['BB_Lower'], df_show['BB_Upper'], color='lightgreen', alpha=0.12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig1)

            # Backtest chart
            st.subheader("Backtest: Strategy vs Buy & Hold")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(df.index, df['Equity'], label='Strategy Equity', color='green', lw=2)
            ax2.plot(df.index, 10000 * (df['close'] / df['close'].iloc[0]), label='Buy & Hold', color='gray', ls='--')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)

            # ────────────────────────────────────────────────
            # ARIMA Forecast
            # ────────────────────────────────────────────────
            st.subheader("30-Day ARIMA Price Forecast")
            try:
                train = df['close'][-500:].copy()
                model = ARIMA(train, order=(5,1,2)).fit()
                forecast = model.forecast(steps=30)
                forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')

                fig3, ax3 = plt.subplots(figsize=(10, 5))
                ax3.plot(df.index[-120:], df['close'][-120:], label='Historical', color='blue')
                ax3.plot(forecast_dates, forecast, label='30-Day Forecast', color='red', ls='--', lw=2.2)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig3)

                # Recommendation
                last_close = df['close'].iloc[-1]
                avg_forecast = forecast.mean()
                rsi_now = df['RSI'].iloc[-1]

                if avg_forecast > last_close * 1.01 and rsi_now < 60:
                    reco = "BUY"
                    color = "green"
                elif avg_forecast < last_close * 0.99:
                    reco = "SELL"
                    color = "red"
                else:
                    reco = "HOLD"
                    color = "orange"

                st.markdown(f"**Final AI Recommendation: <span style='color:{color}; font-size:1.4em;'>{reco}</span>**", unsafe_allow_html=True)
                st.caption(f"Last close: {last_close:,.2f} | Avg forecast: {avg_forecast:,.2f} | RSI: {rsi_now:.1f}")

            except Exception as e:
                st.warning("ARIMA forecast failed (possibly not enough data or convergence issue). Showing other results.")

        except Exception as e:
            st.error(f"Error during processing: {str(e)}")

st.markdown("---")
st.caption("Educational project • Data from Yahoo Finance • Not investment advice • Built with Streamlit")