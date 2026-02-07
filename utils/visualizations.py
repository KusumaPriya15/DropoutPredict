# =========================================================
# 📊 visualizations.py — Final Version (DropoutPredict)
# Supports: Model performance charts, India maps, forecasts
# Compatible with CatXGB Ensemble (CatBoost + XGBoost Weighted)
# =========================================================

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import folium
from folium import plugins
import streamlit as st
from utils.data_processor import get_risk_color, get_state_coordinates

# =========================================================
# 🗺️ INDIA MAP VISUALIZATION
# =========================================================
def create_india_map(df, year=None):
    if year is None:
        year = df['Year'].max()
    
    df_year = df[df['Year'] == year].copy()
    india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='OpenStreetMap')
    coordinates = get_state_coordinates()
    
    for _, row in df_year.iterrows():
        state = row['India/State/UT']
        if state == 'India':
            continue
        
        if state in coordinates:
            lat, lon = coordinates[state]
            dropout_rate = row['Average Dropout Rate']
            risk = row['Risk_Category']
            color = get_risk_color(risk)
            
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4 style="margin: 0;">{state}</h4>
                <hr style="margin: 5px 0;">
                <p><b>Dropout Rate:</b> {dropout_rate:.2f}%</p>
                <p><b>Risk:</b> <span style="color: {color}; font-weight: bold;">{risk}</span></p>
                <p><b>Total Schools:</b> {int(row['Total Schools']):,}</p>
                <p><b>Total Enrolment:</b> {int(row['Total Enrolment']):,}</p>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=8 + (dropout_rate * 0.5),
                popup=folium.Popup(popup_html, max_width=250),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(india_map)
    
    return india_map

# =========================================================
# 📈 NATIONAL TREND CHART
# =========================================================
def create_trend_chart(df, metric='Average Dropout Rate'):
    df_india = df[df['India/State/UT'] == 'India'].copy().sort_values('Year')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_india['Year'],
        y=df_india[metric],
        mode='lines+markers',
        name=metric,
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f'National {metric} Trend (2019–2024)',
        xaxis_title='Year',
        yaxis_title=metric,
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    return fig

# =========================================================
# 🧾 STATE COMPARISON BAR CHART
# =========================================================
def create_state_comparison_chart(df, year=None, top_n=10):
    if year is None:
        year = df['Year'].max()
    
    df_year = df[df['Year'] == year].copy()
    df_year = df_year[df_year['India/State/UT'] != 'India']
    
    df_sorted = df_year.sort_values('Average Dropout Rate', ascending=False).head(top_n)
    colors = [get_risk_color(risk) for risk in df_sorted['Risk_Category']]
    
    fig = go.Figure([
        go.Bar(
            x=df_sorted['Average Dropout Rate'],
            y=df_sorted['India/State/UT'],
            orientation='h',
            marker=dict(color=colors),
            text=df_sorted['Average Dropout Rate'].round(2),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_n} States by Dropout Rate ({year})',
        xaxis_title='Dropout Rate (%)',
        yaxis_title='State/UT',
        height=500,
        template='plotly_white',
        showlegend=False
    )
    return fig

# =========================================================
# 🥧 RISK DISTRIBUTION PIE CHART
# =========================================================
def create_risk_distribution_pie(df, year=None):
    if year is None:
        year = df['Year'].max()
    
    df_year = df[df['Year'] == year]
    df_year = df_year[df_year['India/State/UT'] != 'India']
    
    risk_counts = df_year['Risk_Category'].value_counts()
    colors = [get_risk_color(risk) for risk in risk_counts.index]
    
    fig = go.Figure([
        go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker=dict(colors=colors),
            textinfo='label+percent+value',
            hole=0.3
        )
    ])
    
    fig.update_layout(
        title=f'Risk Distribution Across States ({year})',
        height=400,
        template='plotly_white'
    )
    return fig

# =========================================================
# 🔍 CONFUSION MATRIX PLOT
# =========================================================
def create_confusion_matrix_plot(cm, class_names):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=500,
        template='plotly_white'
    )
    return fig

# =========================================================
# 🧠 MODEL PERFORMANCE COMPARISON (3 Models)
# =========================================================
def create_model_comparison_chart(metrics_dict):
    model_names = list(metrics_dict.keys())
    accuracy = [metrics_dict[m]['accuracy'] for m in model_names]
    precision = [metrics_dict[m]['precision'] for m in model_names]
    recall = [metrics_dict[m]['recall'] for m in model_names]
    f1 = [metrics_dict[m]['f1_score'] for m in model_names]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy', x=model_names, y=accuracy, marker_color='#1f77b4'))
    fig.add_trace(go.Bar(name='Precision', x=model_names, y=precision, marker_color='#ff7f0e'))
    fig.add_trace(go.Bar(name='Recall', x=model_names, y=recall, marker_color='#2ca02c'))
    fig.add_trace(go.Bar(name='F1-Score', x=model_names, y=f1, marker_color='#d62728'))
    
    fig.update_layout(
        title='Model Performance Comparison (3 Final Models)',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        height=500,
        template='plotly_white',
        yaxis=dict(range=[0, 1])
    )
    return fig

# =========================================================
# 🧩 FEATURE IMPORTANCE CHART (Supports CatXGB Weighted)
# =========================================================
def create_feature_importance_chart(importance_dict, top_n=15):
    if not importance_dict:
        return None
    
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_features)
    
    fig = go.Figure([
        go.Bar(
            x=list(importances),
            y=list(features),
            orientation='h',
            marker=dict(
                color='#00CC66',
                line=dict(color='#004C1A', width=1.5)
            ),
            text=[f"{val:.4f}" for val in importances],
            textposition="auto"
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_n} Feature Importances',
        xaxis_title='Importance (Normalized)',
        yaxis_title='Feature',
        height=600,
        template='plotly_white'
    )
    
    fig.add_annotation(
        text="For CatXGB Ensemble: Importance = 0.6×CatBoost + 0.4×XGBoost",
        xref="paper", yref="paper", x=0, y=-0.15, showarrow=False, font=dict(size=12, color="gray")
    )
    
    return fig

# =========================================================
# 🔮 FORECAST CHART (State-level)
# =========================================================
def create_forecast_chart(df, state_name, forecast):
    df_state = df[df['India/State/UT'] == state_name].copy().sort_values('Year')
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_state['Year'],
        y=df_state['Average Dropout Rate'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    latest_year = df['Year'].max()
    forecast_future = forecast[forecast['year'] > latest_year]
    
    fig.add_trace(go.Scatter(
        x=forecast_future['year'],
        y=forecast_future['yhat'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_future['year'],
        y=forecast_future['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_future['year'],
        y=forecast_future['yhat_lower'],
        mode='lines',
        name='Confidence Interval',
        fill='tonexty',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(width=0)
    ))
    
    fig.update_layout(
        title=f'Dropout Rate Forecast for {state_name}',
        xaxis_title='Year',
        yaxis_title='Dropout Rate (%)',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    return fig

# =========================================================
# 🗺️ FORECAST MAP (Predicted Risks)
# =========================================================
def create_forecast_map(forecast_summary):
    india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='OpenStreetMap')
    coordinates = get_state_coordinates()
    
    for _, row in forecast_summary.iterrows():
        state = row['State']
        if state in coordinates:
            lat, lon = coordinates[state]
            predicted_rate = row['Predicted_Dropout_Rate']
            risk = row['Risk_Category']
            color = get_risk_color(risk)
            
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4>{state}</h4>
                <hr>
                <p><b>Predicted Rate:</b> {predicted_rate:.2f}%</p>
                <p><b>Risk:</b> <span style="color:{color}; font-weight:bold;">{risk}</span></p>
                <p><b>Range:</b> {row['Lower_Bound']:.2f}% – {row['Upper_Bound']:.2f}%</p>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=8 + (predicted_rate * 0.5),
                popup=folium.Popup(popup_html, max_width=250),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(india_map)
    
    return india_map
