import yfinance as yf
import pandas as pd

# List of stock tickers
tickers = ['TSLA', 'PLTR', 'META', 'UBER', 'HOOD', 'BTC-USD', 'GC=F']

# Download data from 2023-08-18 to 2024-03-18
data = yf.download(tickers, start='2023-08-18', end='2024-03-18', group_by='ticker')

# Save each ticker's data to a CSV file
for ticker in tickers:
    df = data[ticker].copy()
    df.to_csv(f'{ticker}_stock_data.csv')
    print(f'Saved data for {ticker}')
