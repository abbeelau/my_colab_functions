import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def plot_rolling_return(ticker='^NDX', window_days=1008, title='Rolling 4-Year Return'):
    end_date = datetime.date.today()
    price = yf.download(ticker, start='1980-01-01', end=end_date)['Close']
    rolling_return = price.pct_change(periods=window_days) * 100

    plt.figure(figsize=(12,6))
    plt.plot(rolling_return.index, rolling_return, color='darkgreen', linewidth=2)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'{ticker} {title}', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    plt.show()

    return rolling_return

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def plot_pct_distance_from_ma(ticker: str, ma_window: int = 20, start: str = "2022-01-01", end: str = None):
    """
    Plots the % distance of the stock's closing price from its XX-day moving average.

    Parameters:
    - ticker (str): Stock ticker symbol (e.g., 'AAPL' or '^NDX')
    - ma_window (int): Number of days for moving average (e.g., 20 for 20-day MA)
    - start (str): Start date in 'YYYY-MM-DD'
    - end (str or None): End date in 'YYYY-MM-DD'. Defaults to today.
    """
    # Fetch historical stock/index data
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if df.empty:
        print("No data found for the specified ticker and date range.")
        return

    # Handle multi-index columns for index tickers like ^NDX
    if isinstance(df.columns, pd.MultiIndex):
        # Try to get 'Close' from multi-level column
        try:
            df_close = df[("Close", ticker)]
        except KeyError:
            print("Close price not found for ticker:", ticker)
            return
    else:
        # Standard single-level column
        try:
            df_close = df["Close"]
        except KeyError:
            print("'Close' column not found in data.")
            return

    # Create DataFrame for calculations
    calc_df = pd.DataFrame({'Close': df_close})
    calc_df['MA'] = calc_df['Close'].rolling(window=ma_window).mean()
    calc_df['% Distance'] = ((calc_df['Close'] - calc_df['MA']) / calc_df['MA']) * 100
    calc_df = calc_df.dropna()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(calc_df.index, calc_df['% Distance'], label=f'% Distance from {ma_window}-day MA')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"{ticker} - % Distance from {ma_window}-Day Moving Average")
    plt.xlabel("Date")
    plt.ylabel("% Distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
