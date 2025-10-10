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
