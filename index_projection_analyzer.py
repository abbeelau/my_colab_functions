import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

class IndexProjectionAnalyzer:
    """
    Analyzes index projections based on historical data and expected returns.
    Uses centered average (e.g., 24 months before + 24 months after reference date).
    """
    
    # Expanded ticker mapping
    INDEX_TICKERS = {
        # Major Indices
        'SPX': '^GSPC',
        'NASDAQ': '^IXIC',
        'HSI': '^HSI',
        'DJI': '^DJI',
        'FTSE': '^FTSE',
        'DAX': '^GDAXI',
        'NIKKEI': '^N225',
        
        # Commodities
        'GOLD': 'GC=F',
        'SILVER': 'SI=F',
        'OIL': 'CL=F',
        
        # Crypto
        'BITCOIN': 'BTC-USD',
        'ETHEREUM': 'ETH-USD',
        
        # Magnificent 7 Tech Stocks
        'AAPL': 'AAPL',
        'MSFT': 'MSFT',
        'GOOGL': 'GOOGL',
        'AMZN': 'AMZN',
        'NVDA': 'NVDA',
        'META': 'META',
        'TSLA': 'TSLA',
        
        # Other Popular Stocks
        'NFLX': 'NFLX',
        'AMD': 'AMD',
        'INTC': 'INTC',
        'JPM': 'JPM',
        'BAC': 'BAC',
        'V': 'V',
        'MA': 'MA',
        'DIS': 'DIS',
        'WMT': 'WMT',
        'PG': 'PG',
    }
    
    def __init__(self, ticker_symbol='SPX', lookback_months=48, 
                 projection_years=7, expected_return_pct=100, use_adjusted=True):
        """
        Initialize the analyzer with parameters.
        
        Parameters:
        -----------
        ticker_symbol : str
            Ticker to analyze (e.g., 'SPX', 'BITCOIN', 'AAPL')
            See INDEX_TICKERS dictionary for available options
        lookback_months : int
            Total months to average (centered around reference date, default: 48)
        projection_years : int
            Years to project forward (default: 7)
        expected_return_pct : float
            Expected return percentage over projection period (default: 100%)
        use_adjusted : bool
            Use adjusted close prices (default: True) or regular close prices
        """
        self.ticker_symbol = ticker_symbol.upper()
        
        # Get actual ticker from mapping, or use the symbol directly if not in mapping
        self.ticker = self.INDEX_TICKERS.get(self.ticker_symbol, self.ticker_symbol)
        
        self.lookback_months = lookback_months
        self.projection_years = projection_years
        self.expected_return_pct = expected_return_pct
        self.use_adjusted = use_adjusted
        self.data = None
        self.monthly_data = None
        self.projections = None
        
    @classmethod
    def list_available_tickers(cls):
        """Print all available ticker symbols."""
        print("\n" + "="*60)
        print("AVAILABLE TICKERS")
        print("="*60)
        
        categories = {
            'Major Indices': ['SPX', 'NASDAQ', 'DJI', 'HSI', 'FTSE', 'DAX', 'NIKKEI'],
            'Commodities': ['GOLD', 'SILVER', 'OIL'],
            'Crypto': ['BITCOIN', 'ETHEREUM'],
            'Magnificent 7': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
            'Other Stocks': ['NFLX', 'AMD', 'INTC', 'JPM', 'BAC', 'V', 'MA', 'DIS', 'WMT', 'PG']
        }
        
        for category, tickers in categories.items():
            print(f"\n{category}:")
            for ticker in tickers:
                actual_ticker = cls.INDEX_TICKERS.get(ticker, ticker)
                print(f"  {ticker:12} -> {actual_ticker}")
        
        print("\nYou can also use any valid Yahoo Finance ticker directly!")
        print("="*60 + "\n")
        
    def fetch_data(self, start_date='2015-01-01', end_date=None):
        """
        Fetch historical data from Yahoo Finance.
        
        Parameters:
        -----------
        start_date : str
            Start date for data retrieval (YYYY-MM-DD)
        end_date : str
            End date for data retrieval (default: today)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Fetching {self.ticker_symbol} ({self.ticker}) data from {start_date} to {end_date}...")
        
        try:
            self.data = yf.download(self.ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            
            if self.data.empty:
                print(f"ERROR: No data retrieved for {self.ticker_symbol} ({self.ticker})")
                return None
            
            # Choose between adjusted or unadjusted close prices
            # Handle both single ticker (Series-like) and multi-ticker (MultiIndex) formats
            if 'Adj Close' in self.data.columns:
                price_column = 'Adj Close' if self.use_adjusted else 'Close'
            else:
                # For single ticker downloads, columns might be at top level
                price_column = 'Adj Close' if self.use_adjusted else 'Close'
                if price_column not in self.data.columns:
                    # Fallback to Close if Adj Close not available
                    price_column = 'Close'
            
            print(f"Using {'adjusted' if self.use_adjusted else 'unadjusted'} close prices")
            
            # Calculate monthly average
            self.monthly_data = self.data[price_column].resample('M').mean()
            self.monthly_data.index = self.monthly_data.index.to_period('M').to_timestamp('M')
            
            print(f"Data fetched: {len(self.monthly_data)} monthly data points")
            return self.monthly_data
            
        except Exception as e:
            print(f"ERROR fetching data: {e}")
            return None
    
    def calculate_projection(self, reference_date):
        """
        Calculate projected index level for a given reference date.
        Uses CENTERED average: (lookback_months/2) before and after reference date.
        
        For example, with lookback_months=48 and reference_date=2022-12-31:
        - Window: 2020-12-31 to 2024-12-31 (24 months before + 24 months after)
        
        Parameters:
        -----------
        reference_date : datetime or str
            The reference date to calculate projection from
            
        Returns:
        --------
        tuple : (projected_date, projected_level, starting_avg, window_start, window_end)
        """
        if isinstance(reference_date, str):
            reference_date = pd.to_datetime(reference_date)
        
        # Calculate centered window
        half_window = self.lookback_months // 2
        window_start = reference_date - relativedelta(months=half_window)
        window_end = reference_date + relativedelta(months=half_window)
        
        # Get historical data for averaging (centered window)
        mask = (self.monthly_data.index >= window_start) & (self.monthly_data.index <= window_end)
        historical_window = self.monthly_data[mask]
        
        if len(historical_window) == 0:
            return None, None, None, None, None
        
        # Calculate average as starting point
        starting_avg = historical_window.mean()
        
        # Calculate projected level
        multiplier = 1 + (self.expected_return_pct / 100)
        projected_level = starting_avg * multiplier
        
        # Calculate projection date (from reference date)
        projected_date = reference_date + relativedelta(years=self.projection_years)
        
        return projected_date, projected_level, starting_avg, window_start, window_end
    
    def run_rolling_projections(self, start_date=None, end_date=None, freq='M'):
        """
        Run projections for multiple reference dates.
        
        Parameters:
        -----------
        start_date : str or datetime
            First reference date for projections
        end_date : str or datetime
            Last reference date for projections
        freq : str
            Frequency of reference dates ('M' for monthly)
            
        Returns:
        --------
        DataFrame : Projection results
        """
        if self.monthly_data is None:
            print("No data available. Run fetch_data() first.")
            return None
            
        # Set default dates with enough buffer for centered window
        half_window = self.lookback_months // 2
        
        if start_date is None:
            start_date = self.monthly_data.index[half_window]
        if end_date is None:
            # Ensure we have enough future data for centered window
            end_date = self.monthly_data.index[-half_window-1]
            
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate reference dates
        reference_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        results = []
        for ref_date in reference_dates:
            proj_date, proj_level, start_avg, win_start, win_end = self.calculate_projection(ref_date)
            if proj_date is not None:
                results.append({
                    'reference_date': ref_date,
                    'window_start': win_start,
                    'window_end': win_end,
                    'starting_average': start_avg,
                    'projected_date': proj_date,
                    'projected_level': proj_level
                })
        
        self.projections = pd.DataFrame(results)
        print(f"Generated {len(self.projections)} projections")
        return self.projections
    
    def plot_results(self, plot_start_date=None, figsize=(14, 8)):
        """
        Create visualization of historical data and projections.
        
        Parameters:
        -----------
        plot_start_date : str or datetime
            Start date for visualization (default: show all data)
        figsize : tuple
            Figure size (width, height)
        """
        if self.projections is None:
            print("No projections available. Run run_rolling_projections() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
        
        # Filter historical data for plotting
        hist_data = self.monthly_data
        if plot_start_date is not None:
            if isinstance(plot_start_date, str):
                plot_start_date = pd.to_datetime(plot_start_date)
            hist_data = hist_data[hist_data.index >= plot_start_date]
        
        # Plot 1: Historical and Projected Levels
        ax1.plot(hist_data.index, hist_data.values, 
                label='Historical Monthly Average', color='blue', linewidth=2)
        ax1.plot(self.projections['projected_date'], self.projections['projected_level'], 
                label=f'Projected Level ({self.projection_years}Y @ {self.expected_return_pct}%)', 
                color='red', linewidth=2, linestyle='--', marker='o', markersize=3)
        
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Price Level', fontsize=11)
        price_type = 'Adjusted' if self.use_adjusted else 'Unadjusted'
        ax1.set_title(f'{self.ticker_symbol} Historical vs Projected Levels ({price_type} Prices)\n'
                     f'(Centered {self.lookback_months}-month average, Projection: {self.projection_years} years @ {self.expected_return_pct}% return)',
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=9)
        
        # Plot 2: Starting Average Levels
        ax2.plot(self.projections['reference_date'], self.projections['starting_average'],
                label=f'Centered {self.lookback_months}-Month Average (Starting Point)', 
                color='green', linewidth=2, marker='s', markersize=3)
        
        ax2.set_xlabel('Reference Date', fontsize=11)
        ax2.set_ylabel('Average Level', fontsize=11)
        ax2.set_title(f'Centered {self.lookback_months}-Month Average Used as Starting Point', 
                     fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def export_projections(self, filename='projections.csv'):
        """
        Export projections to CSV file.
        
        Parameters:
        -----------
        filename : str
            Output filename (default: 'projections.csv')
        """
        if self.projections is None:
            print("No projections to export. Run run_rolling_projections() first.")
            return
        
        self.projections.to_csv(filename, index=False)
        print(f"Projections exported to {filename}")

# Example usage
if __name__ == "__main__":
    # Show available tickers
    IndexProjectionAnalyzer.list_available_tickers()
    
    # Initialize analyzer with parameters
    analyzer = IndexProjectionAnalyzer(
        ticker_symbol='BITCOIN',     # Try: 'SPX', 'NASDAQ', 'BITCOIN', 'AAPL', etc.
        lookback_months=48,          # Total months for centered average (24 before + 24 after)
        projection_years=7,          # Years to project forward
        expected_return_pct=100,     # Expected return percentage
        use_adjusted=True            # True for adjusted prices, False for unadjusted
    )
    
    # Fetch historical data (need extra years for centered window)
    analyzer.fetch_data(start_date='2015-01-01')
    
    # Run rolling projections
    projections = analyzer.run_rolling_projections(
        start_date='2020-01-31',    # Start of projection analysis
        end_date='2022-12-31'        # End of projection analysis (limited by available future data)
    )
    
    # Display sample results
    if projections is not None:
        print("\nSample Projections:")
        print(projections.head(10))
        print("\n")
        print(projections.tail(10))
        
        # Create visualization
        analyzer.plot_results(plot_start_date='2018-01-01')
        
        # Export results
        analyzer.export_projections('bitcoin_projections.csv')
    
    # Example: Single projection calculation for 2022-12-31
    print("\n" + "="*70)
    print("Example: Single Projection for 2022-12-31")
    print("="*70)
    proj_date, proj_level, start_avg, win_start, win_end = analyzer.calculate_projection('2022-12-31')
    if proj_date is not None:
        print(f"Reference Date: 2022-12-31")
        print(f"Averaging Window: {win_start.strftime('%Y-%m-%d')} to {win_end.strftime('%Y-%m-%d')}")
        print(f"Starting Average ({analyzer.lookback_months}-month centered): {float(start_avg):,.2f}")
        print(f"Projected Level ({analyzer.projection_years} years): {float(proj_level):,.2f}")
        print(f"Projected Date: {proj_date.strftime('%Y-%m-%d')}")
        print(f"Expected Return: {analyzer.expected_return_pct}%")
    else:
        print("Not enough data for 2022-12-31 projection. Try a date with more surrounding data.")
