import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import load_data, preprocess_data, filter_data, calculate_key_metrics
from utils.visualizations import (
    create_india_map, create_trend_chart, create_state_comparison_chart, 
    create_risk_distribution_pie
)
from streamlit_folium import folium_static

st.set_page_config(page_title="Overview Dashboard", page_icon="📊", layout="wide")

st.title("📊 Overview Dashboard")
st.markdown("Comprehensive analysis of school dropout rates across India")

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

st.sidebar.header("Filters")

years = sorted(df['Year'].unique())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=(int(min(years)), int(max(years)))
)

states = sorted([s for s in df['India/State/UT'].unique() if s != 'India'])
selected_states = st.sidebar.multiselect(
    "Select States/UTs",
    options=states,
    default=[]
)

risk_categories = ['High', 'Medium', 'Low']
selected_risks = st.sidebar.multiselect(
    "Select Risk Categories",
    options=risk_categories,
    default=risk_categories
)

df_filtered = filter_data(
    df, 
    year_range=year_range,
    states=selected_states if selected_states else None,
    risk_categories=selected_risks
)

metrics = calculate_key_metrics(df)

st.markdown("---")
st.subheader(f"📈 Key Metrics ({metrics['latest_year']})")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total States/UTs",
        f"{metrics['total_states']}",
        help="Number of states and union territories"
    )

with col2:
    st.metric(
        "Avg. Dropout Rate",
        f"{metrics['avg_dropout']:.2f}%",
        help="National average dropout rate"
    )

with col3:
    st.metric(
        "Total Schools",
        f"{metrics['total_schools']:,}",
        help="Total number of schools across India"
    )

with col4:
    st.metric(
        "Total Enrolment",
        f"{metrics['total_enrolment']:,}",
        help="Total student enrolment"
    )

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f'<div style="background-color: #FF4444; padding: 1rem; border-radius: 5px; text-align: center;">'
        f'<h3 style="color: white; margin: 0;">🔴 {metrics["high_risk_states"]}</h3>'
        f'<p style="color: white; margin: 0;">High Risk States</p>'
        f'</div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f'<div style="background-color: #FFA500; padding: 1rem; border-radius: 5px; text-align: center;">'
        f'<h3 style="color: white; margin: 0;">🟠 {metrics["medium_risk_states"]}</h3>'
        f'<p style="color: white; margin: 0;">Medium Risk States</p>'
        f'</div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f'<div style="background-color: #00CC66; padding: 1rem; border-radius: 5px; text-align: center;">'
        f'<h3 style="color: white; margin: 0;">🟢 {metrics["low_risk_states"]}</h3>'
        f'<p style="color: white; margin: 0;">Low Risk States</p>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("---")
st.subheader("🗺️ Geographical Visualization")

selected_map_year = st.selectbox(
    "Select Year for Map",
    options=sorted(df['Year'].unique(), reverse=True),
    index=0
)

india_map = create_india_map(df, year=selected_map_year)
folium_static(india_map, width=1200, height=600)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 National Dropout Trend")
    trend_fig = create_trend_chart(df, metric='Average Dropout Rate')
    st.plotly_chart(trend_fig, use_container_width=True)

with col2:
    st.subheader("🥧 Risk Distribution")
    pie_fig = create_risk_distribution_pie(df, year=selected_map_year)
    st.plotly_chart(pie_fig, use_container_width=True)

st.markdown("---")
st.subheader("📊 State-wise Comparison")

comparison_year = st.selectbox(
    "Select Year for Comparison",
    options=sorted(df['Year'].unique(), reverse=True),
    index=0,
    key='comparison_year'
)

top_n = st.slider("Number of States to Display", min_value=5, max_value=37, value=10)

comparison_fig = create_state_comparison_chart(df, year=comparison_year, top_n=top_n)
st.plotly_chart(comparison_fig, use_container_width=True)

st.markdown("---")
st.subheader("📋 Detailed Data Table")

df_display = df_filtered[df_filtered['India/State/UT'] != 'India'].copy()
df_display = df_display[[
    'India/State/UT', 'Year', 'Average Dropout Rate', 'Risk_Category',
    'Total Schools', 'Total Enrolment', 'Total Teachers'
]].sort_values(['Year', 'Average Dropout Rate'], ascending=[False, False])

st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Average Dropout Rate": st.column_config.NumberColumn(
            "Dropout Rate (%)",
            format="%.2f"
        ),
        "Total Schools": st.column_config.NumberColumn(
            format="%d"
        ),
        "Total Enrolment": st.column_config.NumberColumn(
            format="%d"
        ),
        "Total Teachers": st.column_config.NumberColumn(
            format="%d"
        )
    }
)

st.info("💡 **Tip**: Use the filters in the sidebar to focus on specific years, states, or risk categories!")
