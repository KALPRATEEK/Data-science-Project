import pandas as pd
import numpy as np
import random

# === Configuration ===
start_date = '2024-01-15'
end_date = '2025-05-23'
np.random.seed(42)

# Generate business days
dates = pd.bdate_range(start=start_date, end=end_date)
n = len(dates)

# === Generate realistic synthetic stock data ===
price = np.cumsum(np.random.normal(0.3, 2, size=n)) + 140  # start around 140
volume = np.random.randint(1e6, 2e6, size=n)

df = pd.DataFrame({
    'Date': dates,
    'Open': price + np.random.normal(0, 1, size=n),
    'High': price + np.random.normal(1, 2, size=n),
    'Low': price - np.random.normal(1, 2, size=n),
    'Close': price,
    'Volume': volume
})

# === Calculate indicators ===
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

# MACD
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD_ml'] = ema12 - ema26
df['MACD_signal'] = df['MACD_ml'].ewm(span=9, adjust=False).mean()
df['MACD_his'] = df['MACD_ml'] - df['MACD_signal']

# RSI14
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / (avg_loss + 1e-9)
df['RSI14'] = 100 - (100 / (1 + rs))

# OBV
obv = [0]
for i in range(1, len(df)):
    if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
        obv.append(obv[-1] + df['Volume'].iloc[i])
    elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
        obv.append(obv[-1] - df['Volume'].iloc[i])
    else:
        obv.append(obv[-1])
df['OBV'] = obv

# ATR (Average True Range)
df['H-L'] = df['High'] - df['Low']
df['H-PC'] = abs(df['High'] - df['Close'].shift())
df['L-PC'] = abs(df['Low'] - df['Close'].shift())
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
df['ATR'] = df['TR'].rolling(window=14).mean()

# Stochastic Momentum (dummy calculation)
df['StochasticMomentum'] = (df['Close'] - df['Low'].rolling(window=14).min()) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()) * 100

# Clean up
df = df.drop(columns=['H-L', 'H-PC', 'L-PC', 'TR', 'MACD_signal'])

# Drop initial rows with NaNs due to indicators
df = df.dropna().reset_index(drop=True)

# === Save to CSV ===
df.to_csv('goog_curr.csv', index=False)
print("✅ Synthetic GOOGL data saved as 'goog_curr.csv'")
