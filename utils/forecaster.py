import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

def prepare_prophet_data(df, state_name):
    df_state = df[df['India/State/UT'] == state_name].copy()
    
    df_state = df_state.sort_values('Year')
    
    df_prophet = pd.DataFrame({
        'ds': pd.to_datetime(df_state['Year'].astype(str) + '-01-01'),
        'y': df_state['Average Dropout Rate']
    })
    
    return df_prophet

def forecast_dropout_rate(df, state_name, years_ahead=3):
    df_prophet = prepare_prophet_data(df, state_name)
    
    if len(df_prophet) < 2:
        return None
    
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.95
    )
    
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=years_ahead, freq='YS')
    forecast = model.predict(future)
    
    forecast['year'] = forecast['ds'].dt.year
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0, upper=100)
    
    return forecast

def forecast_all_states(df, years_ahead=3):
    states = df['India/State/UT'].unique()
    states = [s for s in states if s != 'India']
    
    all_forecasts = {}
    
    for state in states:
        try:
            forecast = forecast_dropout_rate(df, state, years_ahead)
            if forecast is not None:
                all_forecasts[state] = forecast
        except Exception as e:
            print(f"Error forecasting for {state}: {e}")
            continue
    
    return all_forecasts

def get_forecast_summary(forecast_dict, year):
    summary = []
    
    for state, forecast in forecast_dict.items():
        forecast_year = forecast[forecast['year'] == year]
        
        if not forecast_year.empty:
            predicted_rate = forecast_year['yhat'].values[0]
            lower_bound = forecast_year['yhat_lower'].values[0]
            upper_bound = forecast_year['yhat_upper'].values[0]
            
            if predicted_rate > 10:
                risk = 'High'
            elif predicted_rate >= 5:
                risk = 'Medium'
            else:
                risk = 'Low'
            
            summary.append({
                'State': state,
                'Predicted_Dropout_Rate': predicted_rate,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound,
                'Risk_Category': risk
            })
    
    return pd.DataFrame(summary)

def calculate_forecast_metrics(df, forecast_dict):
    metrics = {}
    
    latest_year = df['Year'].max()
    
    for state, forecast in forecast_dict.items():
        historical = forecast[forecast['year'] <= latest_year]
        future = forecast[forecast['year'] > latest_year]
        
        if len(future) > 0:
            avg_forecast = future['yhat'].mean()
            trend = 'Increasing' if future['yhat'].iloc[-1] > future['yhat'].iloc[0] else 'Decreasing'
            
            metrics[state] = {
                'avg_forecast': avg_forecast,
                'trend': trend,
                'future_predictions': len(future)
            }
    
    return metrics

def compare_historical_vs_forecast(df, state_name, forecast):
    df_state = df[df['India/State/UT'] == state_name].copy()
    
    historical_years = df_state['Year'].values
    historical_rates = df_state['Average Dropout Rate'].values
    
    forecast_years = forecast['year'].values
    forecast_rates = forecast['yhat'].values
    
    comparison = pd.DataFrame({
        'Year': forecast_years,
        'Forecasted': forecast_rates
    })
    
    historical_df = pd.DataFrame({
        'Year': historical_years,
        'Actual': historical_rates
    })
    
    merged = comparison.merge(historical_df, on='Year', how='left')
    
    return merged
