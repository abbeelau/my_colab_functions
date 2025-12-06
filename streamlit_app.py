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
    "Major Indices": ["SPX", "NASDAQ", "DJI", "HSI", "FTSE", "DAX", "NIKKEI"],
    "Magnificent 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "Crypto": ["BITCOIN", "ETHEREUM"],
    "Commodities": ["GOLD", "SILVER", "OIL"],
    "Other Stocks": ["NFLX", "AMD", "INTC", "JPM", "BAC", "V", "MA", "DIS", "WMT", "PG"]
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
    value=pd.to_datetime("1990-01-01"),
    min_value=pd.to_datetime("1970-01-01"),
    max_value=pd.to_datetime("today")
)

col_start, col_end = st.sidebar.columns(2)

with col_start:
    plot_start_date = st.date_input(
        "Chart start date",
        value=pd.to_datetime("2000-01-01"),
        min_value=pd.to_datetime("1970-01-01"),
        max_value=pd.to_datetime("today")
    )

with col_end:
    plot_end_date = st.date_input(
        "Chart end date",
        value=pd.to_datetime("today"),
        min_value=pd.to_datetime("1970-01-01"),
        max_value=pd.to_datetime("2050-12-31")
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
- Use zoom controls in Charts tab to focus on specific time periods
""")

# Main content area
if run_analysis:
    try:
        with st.spinner(f"Fetching data for {selected_ticker}..."):
            # Create analyzer
            analyzer = IndexProjectionAnalyzer(
                ticker_symbol=selected_ticker,
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
        
        # Display success message
        st.success(f"✅ Analysis complete for **{selected_ticker}**!")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = float(monthly_data.iloc[-1])
            st.metric("Current Price", f"${current_price:,.2f}")
        
        with col2:
            latest_projection = float(projections.iloc[-1]['projected_level'])
            st.metric("Latest Projection", f"${latest_projection:,.2f}")
        
        with col3:
            proj_date = projections.iloc[-1]['projected_date']
            st.metric("Projection Date", proj_date.strftime('%Y-%m-%d'))
        
        with col4:
            potential_gain = ((latest_projection - current_price) / current_price) * 100
            st.metric("Potential Gain", f"{potential_gain:,.1f}%")
        
        st.markdown("---")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Charts", "📋 Data Table", "ℹ️ Explanation"])
        
        with tab1:
            st.subheader("📊 Interactive Charts")
            
            # Add zoom/date range controls
            col_zoom1, col_zoom2, col_zoom3 = st.columns([2, 2, 1])
            
            with col_zoom1:
                chart_start = st.date_input(
                    "Zoom: Start Date",
                    value=plot_start_date,
                    min_value=pd.to_datetime(start_date),
                    max_value=pd.to_datetime("2050-12-31"),
                    key="chart_start"
                )
            
            with col_zoom2:
                chart_end = st.date_input(
                    "Zoom: End Date", 
                    value=plot_end_date,
                    min_value=pd.to_datetime(start_date),
                    max_value=pd.to_datetime("2050-12-31"),
                    key="chart_end"
                )
            
            with col_zoom3:
                st.write("")  # Spacer
                st.write("")  # Spacer
                if st.button("🔄 Reset Zoom"):
                    st.rerun()
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
            
            # Filter historical data for plotting based on zoom range
            hist_data = monthly_data[
                (monthly_data.index >= pd.to_datetime(chart_start)) & 
                (monthly_data.index <= pd.to_datetime(chart_end))
            ]
            
            # Filter projection data - show ALL projections that START in the zoom range
            # This allows the red projection line to extend into the future
            proj_filtered = projections[
                (projections['reference_date'] >= pd.to_datetime(chart_start)) & 
                (projections['reference_date'] <= pd.to_datetime(chart_end))
            ]
            
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
            ref_filtered = projections[
                (projections['reference_date'] >= pd.to_datetime(chart_start)) & 
                (projections['reference_date'] <= pd.to_datetime(chart_end))
            ]
            
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
            
            # Download button for chart
            st.download_button(
                label="💾 Download Chart",
                data=open("temp_chart.png", "rb") if plt.savefig("temp_chart.png") or True else None,
                file_name=f"{selected_ticker}_projection_chart.png",
                mime="image/png"
            )
        
        with tab2:
            st.subheader("📋 Projection Data")
            
            # Format the dataframe for display
            display_df = projections.copy()
            
            # Round numeric columns BEFORE converting dates to strings
            display_df['starting_average'] = display_df['starting_average'].round(2)
            display_df['projected_level'] = display_df['projected_level'].round(2)
            
            # Now convert dates to strings
            display_df['reference_date'] = display_df['reference_date'].dt.strftime('%Y-%m-%d')
            display_df['window_start'] = display_df['window_start'].dt.strftime('%Y-%m-%d')
            display_df['window_end'] = display_df['window_end'].dt.strftime('%Y-%m-%d')
            display_df['projected_date'] = display_df['projected_date'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Download button for data
            csv = projections.to_csv(index=False)
            st.download_button(
                label="💾 Download Data as CSV",
                data=csv,
                file_name=f"{selected_ticker}_projections.csv",
                mime="text/csv"
            )
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Starting Average (Baseline)**")
                st.write(f"- Minimum: ${projections['starting_average'].min():,.2f}")
                st.write(f"- Maximum: ${projections['starting_average'].max():,.2f}")
                st.write(f"- Mean: ${projections['starting_average'].mean():,.2f}")
            
            with col2:
                st.write("**Projected Level**")
                st.write(f"- Minimum: ${projections['projected_level'].min():,.2f}")
                st.write(f"- Maximum: ${projections['projected_level'].max():,.2f}")
                st.write(f"- Mean: ${projections['projected_level'].mean():,.2f}")
        
        with tab3:
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
            
            **3. Rolling Analysis:**
            - We repeat this calculation for multiple reference dates
            - This creates the projection line you see in the chart
            
            ### 📊 Chart Interpretation
            
            **Top Chart:**
            - **Blue Line**: Historical monthly average prices (actual data)
            - **Red Dashed Line**: Projected prices based on your {expected_return_pct}% return assumption
            
            **Bottom Chart:**
            - **Green Line**: The centered moving average used as the baseline for each projection
            - This shows how the "starting point" evolves over time
            
            ### ⚠️ Important Notes
            
            - **This is NOT a prediction**: It's a scenario analysis based on your assumptions
            - **Past performance ≠ future results**: Markets are unpredictable
            - **Centered average limitation**: We need future data, so projections don't reach today's date
            - **Educational purpose only**: Not financial advice - always consult professionals
            
            ### 💡 Use Cases
            
            - **Scenario Planning**: "What if the market grows by X% over Y years?"
            - **Goal Setting**: "What return do I need to reach my target?"
            - **Comparison**: Analyze different assets with the same assumptions
            - **Sensitivity Analysis**: See how different return assumptions affect outcomes
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
    
    ### 📈 Example Scenarios
    
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
    
    # Show sample visualization
    st.markdown("---")
    st.subheader("📊 Sample Output")
    st.image("https://via.placeholder.com/1200x600/1f77b4/ffffff?text=Your+Charts+Will+Appear+Here", 
             caption="Interactive charts will be displayed here after running analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📈 Index Projection Analysis Tool | Built with Streamlit | Educational Purpose Only</p>
    <p style='font-size: 0.8em;'>Not financial advice. Always do your own research and consult professionals.</p>
</div>
""", unsafe_allow_html=True)
