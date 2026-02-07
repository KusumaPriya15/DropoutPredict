import pandas as pd
import numpy as np
import streamlit as st

import os
import pandas as pd
import streamlit as st

def load_data(file_name='UDISE_2019-24_Combined_YearFixed.csv'):
    """Load dataset from either /data folder or root directory"""
    # Define possible paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', file_name)
    root_path = os.path.join(base_dir, file_name)

    # Try /data folder first
    if os.path.exists(data_path):
        file_path = data_path
    elif os.path.exists(root_path):
        file_path = root_path
    else:
        st.error(f"❌ Dataset not found.\nTried paths:\n- {data_path}\n- {root_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        st.success(f"✅ Dataset loaded successfully from: {file_path}")
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {e}")
        return None

def classify_risk(dropout_rate):
    if dropout_rate > 10:
        return 'High'
    elif dropout_rate >= 5:
        return 'Medium'
    else:
        return 'Low'

def get_risk_color(risk_level):
    colors = {
        'High': '#FF4444',
        'Medium': '#FFA500', 
        'Low': '#00CC66'
    }
    return colors.get(risk_level, '#CCCCCC')

def preprocess_data(df):
    if df is None or df.empty:
        return None
    
    df_processed = df.copy()
    
    df_processed['Risk_Category'] = df_processed['Average Dropout Rate'].apply(classify_risk)
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['Year', 'Average Dropout Rate']:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed

def prepare_features_and_target(df):
    if df is None or df.empty:
        return None, None, None, None
    
    df_ml = df.copy()
    
    feature_cols = [col for col in df_ml.columns if col not in [
        'India/State/UT', 'Year', 'Average Dropout Rate', 'Risk_Category'
    ]]
    
    X = df_ml[feature_cols]
    y = df_ml['Risk_Category']
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y, y_encoded, le

def get_state_coordinates():
    coordinates = {
        'India': (20.5937, 78.9629),
        'Andaman & Nicobar Islands': (11.7401, 92.6586),
        'Andhra Pradesh': (15.9129, 79.7400),
        'Arunachal Pradesh': (28.2180, 94.7278),
        'Assam': (26.2006, 92.9376),
        'Bihar': (25.0961, 85.3131),
        'Chandigarh': (30.7333, 76.7794),
        'Chhattisgarh': (21.2787, 81.8661),
        'Dadra and Nagar Haveli and Daman & Diu': (20.4283, 72.8397),
        'Delhi': (28.7041, 77.1025),
        'Goa': (15.2993, 74.1240),
        'Gujarat': (22.2587, 71.1924),
        'Haryana': (29.0588, 76.0856),
        'Himachal Pradesh': (31.1048, 77.1734),
        'Jammu and Kashmir': (33.7782, 76.5762),
        'Jharkhand': (23.6102, 85.2799),
        'Karnataka': (15.3173, 75.7139),
        'Kerala': (10.8505, 76.2711),
        'Ladakh': (34.1526, 77.5771),
        'Lakshadweep': (10.5667, 72.6417),
        'Madhya Pradesh': (22.9734, 78.6569),
        'Maharashtra': (19.7515, 75.7139),
        'Manipur': (24.6637, 93.9063),
        'Meghalaya': (25.4670, 91.3662),
        'Mizoram': (23.1645, 92.9376),
        'Nagaland': (26.1584, 94.5624),
        'Odisha': (20.9517, 85.0985),
        'Puducherry': (11.9416, 79.8083),
        'Punjab': (31.1471, 75.3412),
        'Rajasthan': (27.0238, 74.2179),
        'Sikkim': (27.5330, 88.5122),
        'Tamil Nadu': (11.1271, 78.6569),
        'Telangana': (18.1124, 79.0193),
        'Tripura': (23.9408, 91.9882),
        'Uttar Pradesh': (26.8467, 80.9462),
        'Uttarakhand': (30.0668, 79.0193),
        'West Bengal': (22.9868, 87.8550)
    }
    return coordinates

def filter_data(df, year_range=None, states=None, risk_categories=None):
    if df is None or df.empty:
        return None
    
    df_filtered = df.copy()
    
    if year_range:
        df_filtered = df_filtered[
            (df_filtered['Year'] >= year_range[0]) & 
            (df_filtered['Year'] <= year_range[1])
        ]
    
    if states and len(states) > 0:
        df_filtered = df_filtered[df_filtered['India/State/UT'].isin(states)]
    
    if risk_categories and len(risk_categories) > 0:
        df_filtered = df_filtered[df_filtered['Risk_Category'].isin(risk_categories)]
    
    return df_filtered

def calculate_key_metrics(df):
    if df is None or df.empty:
        return {}
    
    latest_year = df['Year'].max()
    df_latest = df[df['Year'] == latest_year]
    
    total_states = df_latest['India/State/UT'].nunique()
    avg_dropout = df_latest['Average Dropout Rate'].mean()
    
    high_risk_states = len(df_latest[df_latest['Risk_Category'] == 'High'])
    medium_risk_states = len(df_latest[df_latest['Risk_Category'] == 'Medium'])
    low_risk_states = len(df_latest[df_latest['Risk_Category'] == 'Low'])
    
    total_schools = df_latest['Total Schools'].sum()
    total_enrolment = df_latest['Total Enrolment'].sum()
    
    metrics = {
        'total_states': total_states,
        'avg_dropout': avg_dropout,
        'high_risk_states': high_risk_states,
        'medium_risk_states': medium_risk_states,
        'low_risk_states': low_risk_states,
        'total_schools': total_schools,
        'total_enrolment': total_enrolment,
        'latest_year': latest_year
    }
    
    return metrics
