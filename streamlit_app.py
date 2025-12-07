import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from index_projection_analyzer import IndexProjectionAnalyzer

# Page configuration
st.set_page_config(
    page_title="Index Projection Tool",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Index Projection Analysis Tool")
st.markdown("""
This tool helps you project future index/stock/crypto prices based on historical data and expected returns.
**How it works:** Uses a centered moving average as the baseline and projects forward based on your expected return assumption.
""")

# Sidebar for inputs
st.sidebar.header("⚙️ Configuration")

# Ticker selection
ticker_categories = {
    "Major Indices": ["SPX", "NASDAQ-100", "DJI", "HSI", "FTSE", "DAX", "NIKKEI"],
    "Magnificent 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "Crypto": ["BITCOIN", "ETHEREUM"],
    "Commodities": ["GOLD", "SILVER", "OIL"],
    "Other Stocks": ["NFLX", "AMD", "INTC", "JPM", "BAC", "V", "MA", "DIS", "WMT", "PG"]
}

# Update the ticker mapping in the categories to use
ticker_mapping = {
    "NASDAQ-100": "^NDX",  # NASDAQ-100 instead of NASDAQ Composite
}

# Category selection
st.sidebar.subheader("1️⃣ Select Asset Category")
selected_category = st.sidebar.selectbox(
    "Category",
    options=list(ticker_categories.keys())
)

# Ticker selection based on category
st.sidebar.subheader("2️⃣ Select Asset")
selected_ticker = st.sidebar.selectbox(
    "Ticker",
    options=ticker_categories[selected_category]
)

# Allow custom ticker
use_custom = st.sidebar.checkbox("Use custom ticker")
if use_custom:
    selected_ticker = st.sidebar.text_input("Enter ticker symbol", value="AAPL").upper()

st.sidebar.markdown("---")

# Parameters
st.sidebar.subheader("3️⃣ Analysis Parameters")

lookback_months = st.sidebar.slider(
    "Lookback Months (Centered Average)",
    min_value=12,
    max_value=120,
    value=48,
    step=12,
    help="Total months for centered average. E.g., 48 = 24 months before + 24 months after reference date"
)

projection_years = st.sidebar.slider(
    "Projection Years",
    min_value=1,
    max_value=20,
    value=7,
    step=1,
    help="How many years into the future to project"
)

expected_return_pct = st.sidebar.slider(
    "Expected Return (%)",
    min_value=-50,
    max_value=500,
    value=100,
    step=10,
    help="Expected percentage return over the projection period. 100% = doubling"
)

use_adjusted = st.sidebar.checkbox(
    "Use Adjusted Prices",
    value=True,
    help="Adjusted prices account for splits and dividends (recommended)"
)

st.sidebar.markdown("---")

# Date range
st.sidebar.subheader("4️⃣ Data Range")

start_date = st.sidebar.date_input(
    "Fetch data from",
    value=pd.to_datetime("1970-01-01"),
    min_value=pd.to_datetime("1970-01-01"),
    max_value=pd.to_datetime("today")
)

plot_start_date = st.sidebar.date_input(
    "Display chart from",
    value=pd.to_datetime("2000-01-01"),
    min_value=pd.to_datetime("1970-01-01"),
    max_value=pd.to_datetime("today")
)

# Run button
st.sidebar.markdown("---")
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Info box
st.sidebar.markdown("---")
st.sidebar.info("""
**Tips:**
- Larger lookback = smoother projections
- Higher expected return = steeper projection line
- Adjusted prices recommended for stocks
- Centered average needs future data, so projections won't reach current date
""")

# Main content area
if run_analysis:
    try:
        with st.spinner(f"Fetching data for {selected_ticker}..."):
            # Map ticker if needed
            actual_ticker = ticker_mapping.get(selected_ticker, selected_ticker)
            
            # Create analyzer
            analyzer = IndexProjectionAnalyzer(
                ticker_symbol=actual_ticker,
                lookback_months=lookback_months,
                projection_years=projection_years,
                expected_return_pct=expected_return_pct,
                use_adjusted=use_adjusted
            )
            
            # Fetch data
            monthly_data = analyzer.fetch_data(start_date=start_date.strftime('%Y-%m-%d'))
            
            if monthly_data is None or len(monthly_data) == 0:
                st.error(f"❌ Could not fetch data for {selected_ticker}. Please check the ticker symbol.")
                st.stop()
        
        with st.spinner("Calculating projections..."):
            # Run projections
            projections = analyzer.run_rolling_projections()
            
            if projections is None or len(projections) == 0:
                st.error("❌ Not enough data to generate projections. Try adjusting the date range or lookback period.")
                st.stop()
            
            # Store numeric statistics IMMEDIATELY after creation
            starting_avg_min = float(projections['starting_average'].min())
            starting_avg_max = float(projections['starting_average'].max())
            starting_avg_mean = float(projections['starting_average'].mean())
            projected_lvl_min = float(projections['projected_level'].min())
            projected_lvl_max = float(projections['projected_level'].max())
            projected_lvl_mean = float(projections['projected_level'].mean())
        
        # Calculate deviation (actual vs projected)
        # For each historical date, find if there's a corresponding projection
        deviation_data = []
        for date in monthly_data.index:
            actual_price = float(monthly_data.loc[date])
            
            # Find if this date was a projected date from earlier reference dates
            matching_projections = projections[projections['projected_date'] == date]
            
            if len(matching_projections) > 0:
                # Use the earliest projection (most historical reference)
                projected_price = float(matching_projections.iloc[0]['projected_level'])
                deviation_pct = ((actual_price - projected_price) / projected_price) * 100
                
                deviation_data.append({
                    'date': date,
                    'actual_price': actual_price,
                    'projected_price': projected_price,
                    'deviation_pct': deviation_pct,
                    'deviation_abs': actual_price - projected_price
                })
        
        deviation_df = pd.DataFrame(deviation_data) if deviation_data else None
        
        # Display success message
        st.success(f"✅ Analysis complete for **{selected_ticker}**!")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        current_price = float(monthly_data.iloc[-1])
        latest_projection = float(projections.iloc[-1]['projected_level'])
        proj_date = projections.iloc[-1]['projected_date']
        potential_gain = ((latest_projection - current_price) / current_price) * 100
        
        # Calculate annualized return
        years_to_projection = (proj_date - monthly_data.index[-1]).days / 365.25
        annualized_return = ((latest_projection / current_price) ** (1 / years_to_projection) - 1) * 100
        
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}")
        
        with col2:
            st.metric("Latest Projection", f"${latest_projection:,.2f}")
        
        with col3:
            st.metric("Projection Date", proj_date.strftime('%Y-%m-%d'))
        
        with col4:
            st.metric("Total Return", f"{potential_gain:,.1f}%", 
                     delta=f"{potential_gain:,.1f}%" if potential_gain > 0 else None)
        
        with col5:
            st.metric("Annualized Return", f"{annualized_return:,.1f}%/yr",
                     delta=f"{annualized_return:,.1f}%" if annualized_return > 0 else None)
        
        st.markdown("---")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Projection Charts", "📉 Deviation Analysis", "📋 Data Table", "ℹ️ Explanation"])
        
        with tab1:
            st.subheader("📊 Historical vs Projected Levels")
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
            
            # Filter historical data for plotting
            hist_data = monthly_data[monthly_data.index >= pd.to_datetime(plot_start_date)]
            
            # Filter projections by reference date (not projected date)
            proj_filtered = projections[projections['reference_date'] >= pd.to_datetime(plot_start_date)]
            
            # Plot 1: Historical and Projected Levels
            ax1.plot(hist_data.index, hist_data.values, 
                    label='Historical Monthly Average', color='#1f77b4', linewidth=2.5)
            
            if len(proj_filtered) > 0:
                ax1.plot(proj_filtered['projected_date'], proj_filtered['projected_level'], 
                        label=f'Projected Level ({projection_years}Y @ {expected_return_pct}%)', 
                        color='#d62728', linewidth=2.5, linestyle='--', marker='o', markersize=4)
            
            ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Price Level', fontsize=12, fontweight='bold')
            price_type = 'Adjusted' if use_adjusted else 'Unadjusted'
            ax1.set_title(f'{selected_ticker} - Historical vs Projected Levels ({price_type} Prices)\n'
                         f'Centered {lookback_months}-month average, {projection_years}-year projection @ {expected_return_pct}% return',
                         fontsize=14, fontweight='bold', pad=20)
            ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.tick_params(axis='both', labelsize=10)
            
            # Plot 2: Starting Average Levels
            ref_filtered = projections[projections['reference_date'] >= pd.to_datetime(plot_start_date)]
            
            if len(ref_filtered) > 0:
                ax2.plot(ref_filtered['reference_date'], ref_filtered['starting_average'],
                        label=f'Centered {lookback_months}-Month Average (Starting Point)', 
                        color='#2ca02c', linewidth=2.5, marker='s', markersize=4)
            
            ax2.set_xlabel('Reference Date', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Average Level', fontsize=12, fontweight='bold')
            ax2.set_title(f'Rolling {lookback_months}-Month Centered Average (Baseline for Projections)', 
                         fontsize=13, fontweight='bold', pad=15)
            ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.tick_params(axis='both', labelsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            st.subheader("📉 Deviation Analysis: Actual vs Projected")
            
            if deviation_df is not None and len(deviation_df) > 0:
                # Filter deviation data for display
                dev_display = deviation_df[deviation_df['date'] >= pd.to_datetime(plot_start_date)]
                
                if len(dev_display) > 0:
                    # Statistics
                    st.markdown("### 📊 Deviation Statistics")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Mean Deviation", f"{dev_display['deviation_pct'].mean():.2f}%")
                    
                    with col2:
                        st.metric("Std Deviation", f"{dev_display['deviation_pct'].std():.2f}%")
                    
                    with col3:
                        st.metric("Max Over-performance", f"{dev_display['deviation_pct'].max():.2f}%")
                    
                    with col4:
                        st.metric("Max Under-performance", f"{dev_display['deviation_pct'].min():.2f}%")
                    
                    with col5:
                        # Count periods above/below projection
                        above_count = (dev_display['deviation_pct'] > 0).sum()
                        total_count = len(dev_display)
                        above_pct = (above_count / total_count) * 100 if total_count > 0 else 0
                        st.metric("% Time Above", f"{above_pct:.1f}%")
                    
                    st.markdown("---")
                    
                    # Create deviation chart
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 2])
                    
                    # Plot 1: Deviation percentage over time
                    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
                    ax1.fill_between(dev_display['date'], dev_display['deviation_pct'], 0, 
                                    where=(dev_display['deviation_pct'] >= 0),
                                    color='green', alpha=0.3, label='Above Projection')
                    ax1.fill_between(dev_display['date'], dev_display['deviation_pct'], 0,
                                    where=(dev_display['deviation_pct'] < 0),
                                    color='red', alpha=0.3, label='Below Projection')
                    ax1.plot(dev_display['date'], dev_display['deviation_pct'], 
                            color='#1f77b4', linewidth=2, label='Deviation %')
                    
                    # Add mean and std deviation bands
                    mean_dev = dev_display['deviation_pct'].mean()
                    std_dev = dev_display['deviation_pct'].std()
                    ax1.axhline(y=mean_dev, color='orange', linestyle='--', linewidth=2, label=f'Mean ({mean_dev:.1f}%)')
                    ax1.axhline(y=mean_dev + std_dev, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'+1 SD ({mean_dev+std_dev:.1f}%)')
                    ax1.axhline(y=mean_dev - std_dev, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'-1 SD ({mean_dev-std_dev:.1f}%)')
                    
                    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Deviation (%)', fontsize=12, fontweight='bold')
                    ax1.set_title(f'{selected_ticker} - Actual Price Deviation from Projection\n'
                                 f'Positive = Outperforming, Negative = Underperforming',
                                 fontsize=14, fontweight='bold', pad=20)
                    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
                    ax1.grid(True, alpha=0.3, linestyle='--')
                    ax1.tick_params(axis='both', labelsize=10)
                    
                    # Plot 2: Distribution histogram
                    ax2.hist(dev_display['deviation_pct'], bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
                    ax2.axvline(x=0, color='black', linestyle='-', linewidth=2, label='Zero Deviation')
                    ax2.axvline(x=mean_dev, color='orange', linestyle='--', linewidth=2, label=f'Mean ({mean_dev:.1f}%)')
                    ax2.set_xlabel('Deviation (%)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                    ax2.set_title('Distribution of Deviations', fontsize=13, fontweight='bold', pad=15)
                    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
                    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
                    ax2.tick_params(axis='both', labelsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Interpretation guide
                    st.markdown("---")
                    st.markdown("""
                    ### 📖 How to Read This Analysis
                    
                    **Deviation Percentage:**
                    - **Positive (Green)**: Actual price is ABOVE the projection → Market outperforming expectations
                    - **Negative (Red)**: Actual price is BELOW the projection → Market underperforming expectations
                    - **Zero Line**: Actual price matches projection perfectly
                    
                    **Statistical Bands:**
                    - **Orange Dashed Line**: Average deviation over the period
                    - **Gray Dotted Lines**: ±1 Standard Deviation (68% of data falls within these bands)
                    
                    **Investment Insights:**
                    - When actual price is significantly below projection (red zone) → Potentially undervalued
                    - When actual price is significantly above projection (green zone) → Potentially overvalued
                    - Mean deviation shows systematic bias in projections
                    - Large standard deviation indicates high volatility vs projections
                    """)
                else:
                    st.info("No deviation data available for the selected date range. Try adjusting 'Display chart from' date.")
            else:
                st.info("No deviation data available. This requires historical dates that match projected dates from earlier reference periods.")
        
        with tab3:
            st.subheader("📋 Projection Data")
            
            # Format the dataframe for display
            display_df = projections.copy()
            
            # Convert dates to strings
            display_df['reference_date'] = display_df['reference_date'].dt.strftime('%Y-%m-%d')
            display_df['window_start'] = display_df['window_start'].dt.strftime('%Y-%m-%d')
            display_df['window_end'] = display_df['window_end'].dt.strftime('%Y-%m-%d')
            display_df['projected_date'] = display_df['projected_date'].dt.strftime('%Y-%m-%d')
            
            # Format numeric columns as strings (avoid rounding issues)
            display_df['starting_average'] = display_df['starting_average'].apply(lambda x: f"{float(x):.2f}")
            display_df['projected_level'] = display_df['projected_level'].apply(lambda x: f"{float(x):.2f}")
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Download button for data
            csv = projections.to_csv(index=False)
            st.download_button(
                label="💾 Download Projection Data as CSV",
                data=csv,
                file_name=f"{selected_ticker}_projections.csv",
                mime="text/csv"
            )
            
            # Deviation data download
            if deviation_df is not None and len(deviation_df) > 0:
                st.markdown("---")
                st.subheader("📉 Deviation Data")
                
                dev_csv = deviation_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Deviation Data as CSV",
                    data=dev_csv,
                    file_name=f"{selected_ticker}_deviation.csv",
                    mime="text/csv"
                )
            
            # Summary statistics
            st.markdown("---")
            st.subheader("📊 Summary Statistics")
            col1, col2 = st.columns(2)
            
            # Use pre-calculated numeric values
            with col1:
                st.write("**Starting Average (Baseline)**")
                st.write(f"- Minimum: ${starting_avg_min:,.2f}")
                st.write(f"- Maximum: ${starting_avg_max:,.2f}")
                st.write(f"- Mean: ${starting_avg_mean:,.2f}")
            
            with col2:
                st.write("**Projected Level**")
                st.write(f"- Minimum: ${projected_lvl_min:,.2f}")
                st.write(f"- Maximum: ${projected_lvl_max:,.2f}")
                st.write(f"- Mean: ${projected_lvl_mean:,.2f}")
        
        with tab4:
            st.subheader("ℹ️ How This Analysis Works")
            
            st.markdown(f"""
            ### 📐 Methodology
            
            **1. Centered Moving Average:**
            - For each reference date, we calculate a {lookback_months}-month centered average
            - This means {lookback_months//2} months **before** and {lookback_months//2} months **after** the reference date
            - Example: Reference date 2022-12-31 → Average from {2022-lookback_months//12}-12-31 to {2022+lookback_months//12}-12-31
            
            **2. Projection Calculation:**
            - Starting Point = Centered Average
            - Multiplier = 1 + ({expected_return_pct}% / 100) = {1 + expected_return_pct/100:.2f}
            - Projected Level = Starting Point × {1 + expected_return_pct/100:.2f}
            - Projected Date = Reference Date + {projection_years} years
            
            **3. Deviation Analysis:**
            - Compares actual historical prices with what was projected from earlier reference dates
            - Shows whether market outperformed or underperformed expectations
            - Helps identify potential over/undervaluation periods
            
            **4. Annualized Return:**
            - Formula: (Projected Price / Current Price)^(1/Years) - 1
            - Shows compound annual growth rate needed to reach projection
            
            ### 📊 Chart Interpretation
            
            **Projection Charts:**
            - **Blue Line**: Historical monthly average prices (actual data)
            - **Red Dashed Line**: Projected prices based on your {expected_return_pct}% return assumption
            - **Green Line**: The centered moving average used as the baseline
            
            **Deviation Charts:**
            - **Positive (Green)**: Market performing better than projected
            - **Negative (Red)**: Market performing worse than projected
            - **Statistical bands**: Show normal variation range
            
            ### ⚠️ Important Notes
            
            - **This is NOT a prediction**: It's a scenario analysis based on your assumptions
            - **Past performance ≠ future results**: Markets are unpredictable
            - **Centered average limitation**: We need future data, so projections don't reach today's date
            - **Educational purpose only**: Not financial advice - always consult professionals
            
            ### 💡 Use Cases
            
            - **Scenario Planning**: "What if the market grows by X% over Y years?"
            - **Valuation Check**: See if current prices deviate significantly from historical patterns
            - **Comparison**: Analyze different assets with the same assumptions
            - **Risk Assessment**: Understand historical deviation patterns
            """)
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)

else:
    # Welcome screen
    st.info("👈 Configure your analysis in the sidebar and click **🚀 Run Analysis** to start!")
    
    st.markdown("""
    ### 🎯 Quick Start Guide
    
    1. **Select Asset Category** - Choose from indices, stocks, crypto, or commodities
    2. **Select Asset** - Pick the specific ticker you want to analyze
    3. **Set Parameters**:
       - **Lookback Months**: How many months to average (centered around reference date)
       - **Projection Years**: How far into the future to project
       - **Expected Return**: Your assumption for total return over the projection period
    4. **Adjust Dates** - Set data fetching and chart display ranges
    5. **Click Run Analysis** - Get your charts and projections!
    
    ### 📈 New Features
    
    **Enhanced Metrics:**
    - Current Price
    - Latest Projection
    - Total Return (%)
    - **Annualized Return (%/yr)** - Shows compound annual growth rate
    
    **Deviation Analysis:**
    - See how actual prices deviated from projections over time
    - Statistical analysis with mean, standard deviation, min/max
    - Visual charts showing over/under performance
    - Distribution histogram
    
    ### 📊 Example Scenarios
    
    **Conservative S&P 500:**
    - Ticker: SPX
    - Lookback: 48 months
    - Projection: 7 years
    - Expected Return: 70% (≈8% annually)
    
    **Aggressive Tech Stock:**
    - Ticker: NVDA
    - Lookback: 36 months
    - Projection: 5 years
    - Expected Return: 150%
    
    **Long-term Bitcoin:**
    - Ticker: BITCOIN
    - Lookback: 24 months
    - Projection: 10 years
    - Expected Return: 500%
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📈 Index Projection Analysis Tool | Built with Streamlit | Educational Purpose Only</p>
    <p style='font-size: 0.8em;'>Not financial advice. Always do your own research and consult professionals.</p>
</div>
""", unsafe_allow_html=True)
