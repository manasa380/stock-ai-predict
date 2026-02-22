import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ================== CONFIG ==================
ticker = input("Enter stock ticker (e.g. RELIANCE.NS, AAPL, TSLA): ").strip().upper()
start_date = input("Enter start date (YYYY-MM-DD, press Enter for 2023-01-01): ").strip()
if not start_date:
    start_date = "2023-01-01"
# ===========================================

# 1. Fetch Data
print(f"Fetching {ticker} data...")
data = yf.download(ticker, start=start_date, progress=False)
df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df.columns = ['open', 'high', 'low', 'close', 'volume']
print(f"✅ {len(df)} trading days loaded | Latest: {df.index[-1].date()} @ ${df['close'].iloc[-1]:.2f}")

# 2. Technical Indicators (Advanced)
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

df['SMA20'] = df['close'].rolling(20).mean()
df['SMA50'] = df['close'].rolling(50).mean()
df['SMA200'] = df['close'].rolling(200).mean()
df['RSI'] = rsi(df['close'])
df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd(df['close'])
df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = bollinger(df['close'])

# 3. Generate Trading Signals
df['Signal'] = 0
df['Position'] = np.where(df['SMA50'] > df['SMA200'], 1, 0)
df['Crossover'] = df['Position'].diff()

for i in range(1, len(df)):
    if df['Crossover'].iloc[i] == 1 and df['RSI'].iloc[i] < 70:   # Golden + not overbought
        df.loc[df.index[i], 'Signal'] = 1
    elif df['Crossover'].iloc[i] == -1 or df['RSI'].iloc[i] > 65:
        df.loc[df.index[i], 'Signal'] = -1

print(f"\nTrading Signals: {df['Signal'].abs().sum()} total signals")

# 4. Backtesting
capital = 10000
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

print(f"Strategy Return: {((equity[-1]/10000)-1)*100:.1f}%")
print(f"Buy & Hold Return: {((df['close'].iloc[-1]/df['close'].iloc[0])-1)*100:.1f}%")

# 5. Visualizations (4 professional charts)
plt.figure(figsize=(15,8))
plt.plot(df.index, df['close'], label='Close', color='blue', lw=2)
plt.plot(df.index, df['SMA50'], label='SMA50', color='orange')
plt.plot(df.index, df['SMA200'], label='SMA200', color='red')
plt.plot(df.index, df['BB_Upper'], 'g--', alpha=0.6)
plt.plot(df.index, df['BB_Lower'], 'g--', alpha=0.6)
plt.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], color='lightgreen', alpha=0.15)
plt.title(f'{ticker} - Advanced Technical Analysis')
plt.legend(); plt.grid(True); plt.xticks(rotation=45); plt.tight_layout()
plt.savefig('1_price_bb.png'); plt.show()

# Indicators subplot (same as above)
# ... (copy the subplot code from my earlier tool version if you want)

# Backtest chart
plt.figure(figsize=(14,6))
plt.plot(df.index, df['Equity'], label='Strategy', color='green', lw=2.5)
plt.plot(df.index, 10000 * (df['close']/df['close'].iloc[0]), label='Buy & Hold', color='gray', ls='--')
plt.title('Backtest Performance'); plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('2_backtest.png'); plt.show()

# 6. ARIMA Forecast
train = df['close'][-500:]
model = ARIMA(train, order=(5,1,2)).fit()
forecast = model.forecast(30)
forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
plt.figure(figsize=(14,7))
plt.plot(df.index[-120:], df['close'][-120:], label='History', color='blue')
plt.plot(forecast_dates, forecast, label='30-Day Forecast', color='red', ls='--', lw=2)
plt.title('ARIMA 30-Day Price Prediction'); plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('3_forecast.png'); plt.show()

print(f"\n🎯 FINAL RECOMMENDATION: {'BUY' if forecast.mean() > df['close'].iloc[-1]*1.01 and df['RSI'].iloc[-1] < 60 else 'SELL' if forecast.mean() < df['close'].iloc[-1]*0.99 else 'HOLD'}")