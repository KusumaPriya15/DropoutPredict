import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_processor import load_data, preprocess_data, prepare_features_and_target
from utils.model_trainer import train_all_models
from utils.visualizations import create_feature_importance_chart

st.set_page_config(page_title="Feature Importance", page_icon="🔍", layout="wide")
st.title("🔍 Feature Importance — Model Analysis Dashboard")
st.markdown("Explore which features most influence dropout risk predictions for each model.")

@st.cache_data
def load_and_process_data():
    df = load_data()
    if df is not None:
        df = preprocess_data(df)
    return df

df = load_and_process_data()
if df is None:
    st.error("❌ Dataset not found. Upload via the Upload & Retrain page first.")
    st.stop()

X, y, y_encoded, le = prepare_features_and_target(df)
if X is None:
    st.error("⚠️ Could not prepare feature data.")
    st.stop()

# Load trained models
models_dir = "models"
if not os.path.exists(models_dir) or not os.listdir(models_dir):
    st.warning("⚠️ Models missing. Retraining now...")
    with st.spinner("Training models..."):
        train_all_models()
    st.success("✅ Models trained successfully!")

try:
    logistic_model = joblib.load(os.path.join(models_dir, "logistic_regression.pkl"))
    xgb_model = joblib.load(os.path.join(models_dir, "xgboost.pkl"))
    catxgb_cat = joblib.load(os.path.join(models_dir, "catboost.pkl"))
    catxgb_xgb = joblib.load(os.path.join(models_dir, "xgboost_part.pkl"))
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# Sidebar Controls
model_options = ["Logistic Regression", "XGBoost (Standalone)", "CatXGB Ensemble (Proposed)"]
st.sidebar.header("⚙️ Analysis Settings")
selected_model = st.sidebar.selectbox("Select Model", options=model_options)
top_n = st.sidebar.slider("Top N Features", min_value=5, max_value=min(30, len(X.columns)), value=15)

# Helper: normalize importance
def normalize_to_0_1(arr):
    arr = np.array(arr, dtype=float)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

# Feature importance
st.markdown("---")
st.subheader(f"📊 Feature Importance — {selected_model}")

importance_dict = {}
feature_names = [f for f in X.columns if f.lower() != "year"]

try:
    if selected_model == "Logistic Regression":
        coefs = np.abs(logistic_model.coef_).mean(axis=0)
        norm_imp = normalize_to_0_1(coefs)
        importance_dict = dict(zip(feature_names, norm_imp))

    elif selected_model == "XGBoost (Standalone)":
        xgb_imp = np.array(xgb_model.feature_importances_)
        if xgb_imp.shape[0] != len(feature_names):
            st.warning("⚠️ Adjusting XGBoost feature length mismatch.")
            min_len = min(xgb_imp.shape[0], len(feature_names))
            xgb_imp = xgb_imp[:min_len]
            feature_names = feature_names[:min_len]
        norm_imp = normalize_to_0_1(xgb_imp)
        importance_dict = dict(zip(feature_names, norm_imp))

    elif selected_model == "CatXGB Ensemble (Proposed)":
        cat_imp = np.array(catxgb_cat.get_feature_importance())
        xgb_imp = np.array(catxgb_xgb.feature_importances_)
        min_len = min(cat_imp.shape[0], xgb_imp.shape[0], len(feature_names))
        feature_names = feature_names[:min_len]
        combined = 0.6 * normalize_to_0_1(cat_imp[:min_len]) + 0.4 * normalize_to_0_1(xgb_imp[:min_len])
        importance_dict = dict(zip(feature_names, normalize_to_0_1(combined)))
except Exception as e:
    st.error(f"⚠️ Error computing importance: {e}")
    st.stop()

# Display
if importance_dict:
    fig = create_feature_importance_chart(importance_dict, top_n=top_n)
    st.plotly_chart(fig, use_container_width=True)
    df_imp = pd.DataFrame(importance_dict.items(), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
    st.dataframe(df_imp.head(top_n), use_container_width=True, hide_index=True)
else:
    st.warning("⚠️ No importances found.")

st.markdown("---")
st.subheader("📚 Notes")
st.markdown("""
- Importances are **relative**, not causal.
- XGBoost importance is based on feature gain.
- CatXGB Ensemble combines CatBoost (60%) and XGBoost (40%).
- Year column excluded to avoid time-based bias.
""")
