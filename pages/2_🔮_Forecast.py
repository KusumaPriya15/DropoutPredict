import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import load_data, preprocess_data
from utils.forecaster import forecast_dropout_rate, forecast_all_states, get_forecast_summary
from utils.visualizations import create_forecast_chart, create_forecast_map
from streamlit_folium import folium_static
import pandas as pd

st.set_page_config(page_title="Dropout Forecast", page_icon="🔮", layout="wide")

st.title("🔮 Dropout Rate Forecast")
st.markdown("Prophet-based time-series forecasting for 2-3 years ahead")

@st.cache_data
def load_and_process_data():
    df = load_data()
    if df is not None:
        df = preprocess_data(df)
    return df

df = load_and_process_data()

if df is None:
    st.error("Failed to load data. Please check if the dataset file exists.")
    st.stop()

st.sidebar.header("Forecast Settings")

forecast_years = st.sidebar.slider(
    "Years to Forecast Ahead",
    min_value=1,
    max_value=5,
    value=3,
    help="Number of years to forecast into the future"
)

view_mode = st.sidebar.radio(
    "View Mode",
    ["Individual State", "All States Summary"],
    help="Choose between detailed state view or summary of all states"
)

if view_mode == "Individual State":
    st.markdown("---")
    st.subheader("📍 Select State for Detailed Forecast")
    
    states = sorted([s for s in df['India/State/UT'].unique() if s != 'India'])
    selected_state = st.selectbox(
        "Choose a State/UT",
        options=states,
        index=0
    )
    
    with st.spinner(f"Generating forecast for {selected_state}..."):
        forecast = forecast_dropout_rate(df, selected_state, years_ahead=forecast_years)
    
    if forecast is not None:
        st.markdown("---")
        st.subheader(f"📈 Forecast Chart: {selected_state}")
        
        forecast_fig = create_forecast_chart(df, selected_state, forecast)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Forecast Data")
        
        latest_year = df['Year'].max()
        forecast_future = forecast[forecast['year'] > latest_year].copy()
        
        forecast_display = forecast_future[['year', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_display.columns = ['Year', 'Predicted Rate (%)', 'Lower Bound (%)', 'Upper Bound (%)']
        forecast_display['Risk Category'] = forecast_display['Predicted Rate (%)'].apply(
            lambda x: 'High' if x > 10 else ('Medium' if x >= 5 else 'Low')
        )
        
        st.dataframe(
            forecast_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Year": st.column_config.NumberColumn(format="%d"),
                "Predicted Rate (%)": st.column_config.NumberColumn(format="%.2f"),
                "Lower Bound (%)": st.column_config.NumberColumn(format="%.2f"),
                "Upper Bound (%)": st.column_config.NumberColumn(format="%.2f")
            }
        )
        
        st.markdown("---")
        st.subheader("💡 Key Insights")
        
        current_rate = df[
            (df['India/State/UT'] == selected_state) & 
            (df['Year'] == latest_year)
        ]['Average Dropout Rate'].values[0]
        
        future_rate = forecast_display['Predicted Rate (%)'].iloc[-1]
        change = future_rate - current_rate
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                f"Current Rate ({latest_year})",
                f"{current_rate:.2f}%"
            )
        
        with col2:
            st.metric(
                f"Predicted Rate ({int(forecast_display['Year'].iloc[-1])})",
                f"{future_rate:.2f}%",
                delta=f"{change:.2f}%"
            )
        
        with col3:
            trend = "📈 Increasing" if change > 0 else "📉 Decreasing" if change < 0 else "➡️ Stable"
            st.metric("Trend", trend)
        
        if change > 2:
            st.warning(f"⚠️ **Alert**: Dropout rate is predicted to increase by {change:.2f}% over the next {forecast_years} years. Intervention may be needed.")
        elif change < -2:
            st.success(f"✅ **Positive Trend**: Dropout rate is predicted to decrease by {abs(change):.2f}% over the next {forecast_years} years.")
        else:
            st.info(f"ℹ️ **Stable**: Dropout rate is expected to remain relatively stable over the next {forecast_years} years.")
    
    else:
        st.error(f"Unable to generate forecast for {selected_state}. Insufficient historical data.")

else:
    st.markdown("---")
    st.subheader("🌐 All States Forecast Summary")
    
    forecast_year_select = st.selectbox(
        "Select Forecast Year",
        options=list(range(df['Year'].max() + 1, df['Year'].max() + forecast_years + 1)),
        index=forecast_years - 1
    )
    
    with st.spinner("Generating forecasts for all states..."):
        all_forecasts = forecast_all_states(df, years_ahead=forecast_years)
    
    if all_forecasts:
        forecast_summary = get_forecast_summary(all_forecasts, forecast_year_select)
        
        st.markdown("---")
        st.subheader(f"🗺️ Forecast Map ({forecast_year_select})")
        
        forecast_map = create_forecast_map(forecast_summary)
        folium_static(forecast_map, width=1200, height=600)
        
        st.markdown("---")
        st.subheader("📊 Forecast Summary Table")
        
        forecast_summary_display = forecast_summary.sort_values('Predicted_Dropout_Rate', ascending=False)
        
        st.dataframe(
            forecast_summary_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Predicted_Dropout_Rate": st.column_config.NumberColumn(
                    "Predicted Rate (%)",
                    format="%.2f"
                ),
                "Lower_Bound": st.column_config.NumberColumn(
                    "Lower Bound (%)",
                    format="%.2f"
                ),
                "Upper_Bound": st.column_config.NumberColumn(
                    "Upper Bound (%)",
                    format="%.2f"
                )
            }
        )
        
        st.markdown("---")
        st.subheader("📈 Risk Distribution Forecast")
        
        risk_counts = forecast_summary['Risk_Category'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_count = risk_counts.get('High', 0)
            st.markdown(
                f'<div style="background-color: #FF4444; padding: 1rem; border-radius: 5px; text-align: center;">'
                f'<h3 style="color: white; margin: 0;">🔴 {high_count}</h3>'
                f'<p style="color: white; margin: 0;">High Risk States</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            medium_count = risk_counts.get('Medium', 0)
            st.markdown(
                f'<div style="background-color: #FFA500; padding: 1rem; border-radius: 5px; text-align: center;">'
                f'<h3 style="color: white; margin: 0;">🟠 {medium_count}</h3>'
                f'<p style="color: white; margin: 0;">Medium Risk States</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        with col3:
            low_count = risk_counts.get('Low', 0)
            st.markdown(
                f'<div style="background-color: #00CC66; padding: 1rem; border-radius: 5px; text-align: center;">'
                f'<h3 style="color: white; margin: 0;">🟢 {low_count}</h3>'
                f'<p style="color: white; margin: 0;">Low Risk States</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        st.info(f"💡 **Note**: Forecasts are based on historical trends from 2019-{df['Year'].max()} using Prophet time-series model. Predictions include 95% confidence intervals.")
    
    else:
        st.error("Unable to generate forecasts. Please check the data.")

st.markdown("---")
st.markdown("""
### 📖 About Prophet Forecasting

**Prophet** is a time-series forecasting model developed by Facebook that works particularly well with:
- Strong seasonal patterns
- Multiple years of historical data
- Missing data and outliers
- Non-linear growth trends

The forecasts include **confidence intervals** (shown as shaded regions) that indicate the uncertainty in predictions.
""")
