import yfinance as yf
import pandas as pd
from pathlib import Path

def get_stock_data(ticker, start_date, end_date, output_path):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)

    # Ensure the output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    data.to_csv(output_path)
    return data

if __name__ == "__main__":
    start_date = "2017-04-01"
    end_date = "2024-04-01"

    # Fetch S&P 500 data (^GSPC is the Yahoo Finance symbol for the S&P 500)
    sp500_data = get_stock_data("^GSPC", start_date, end_date, "data/raw/sp500.csv")
    print("S&P 500 data fetched and saved.")

    # Fetch AAPL data
    aapl_data = get_stock_data("AAPL", start_date, end_date, "data/raw/aapl.csv")
    print("AAPL data fetched and saved.")

    # Fetch MSFT data
    msft_data = get_stock_data("MSFT", start_date, end_date, "data/raw/msft.csv")
    print("MSFT data fetched and saved.")