import yfinance as yf
import pandas as pd

# List of stock tickers
tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA']

# Download data
data = yf.download(tickers, start='2023-08-18', end='2024-03-18', group_by='ticker')

# Save to CSV files (optional)
for ticker in tickers:
    df = data[ticker].copy()
    df.to_csv(f'{ticker}_stock_data.csv')
    print(f'Saved data for {ticker}')
 