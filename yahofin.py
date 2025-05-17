import yfinance as yf
import os

tickers = [ "GC=F", "BTC-USD"]
data = {}

# Optional: create output directory
output_dir = "stock_data"
os.makedirs(output_dir, exist_ok=True)

for ticker in tickers:
    try:
        print(f"Downloading: {ticker}")
        df = yf.download(ticker, start='2024-11-01', end='2025-02-10', progress=False)
        if not df.empty:
            data[ticker] = df
            file_path = os.path.join(output_dir, f"{ticker.replace('=','_')}.csv")
            df.to_csv(file_path)
            print(f"Saved {ticker} to {os.path.abspath(file_path)}")
        else:
            print(f"No data found for {ticker}.")
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")
