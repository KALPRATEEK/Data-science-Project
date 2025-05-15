import yfinance as yf
import pandas as pd

# List of stock tickers
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA']

# Download data
data = yf.download(tickers, start='2024-11-1', end='2025-02-10', group_by='ticker')

# Save to CSV files (optional)
for ticker in tickers:
    df = data[ticker].copy()
    df.to_csv(f'{ticker}_stock_data.csv')
    print(f'Saved data for {ticker}')
 