import streamlit as st
import sys
import os
import pandas as pd
import shutil
import joblib

# --- Fix import path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import preprocess_data, prepare_features_and_target
from utils.model_trainer import train_all_models

# ============================================================
# ⚙️ Streamlit Configuration
# ============================================================
st.set_page_config(page_title="Upload & Retrain", page_icon="📤", layout="wide")
st.title("📤 Upload New Data & Retrain Models")
st.markdown("Upload fresh data and retrain all 3 dropout prediction models: Logistic Regression, XGBoost, and CatXGB Ensemble.")

# ============================================================
# 📂 Upload Dataset
# ============================================================
uploaded_file = st.file_uploader("📁 Upload CSV dataset", type=['csv'])

if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded: {len(df_uploaded)} rows, {len(df_uploaded.columns)} columns.")
    st.dataframe(df_uploaded.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Data Validation")

    required_cols = ["India/State/UT", "Year", "Average Dropout Rate"]
    missing = [c for c in required_cols if c not in df_uploaded.columns]
    if missing:
        st.error(f"❌ Missing required columns: {', '.join(missing)}")
        st.stop()
    st.success("✅ Validation passed.")

    st.markdown("---")
    st.subheader("🚀 Retrain Models")

    if st.button("🎯 Process & Train"):
        with st.spinner("Processing data..."):
            df_processed = preprocess_data(df_uploaded)
            if df_processed is None:
                st.error("❌ Failed to process uploaded dataset.")
                st.stop()

        st.success("✅ Data processed successfully.")

        with st.spinner("Training 3 models (Logistic Regression, XGBoost, CatXGB Ensemble)..."):
            models, metrics = train_all_models()

        st.success("🎉 All models trained successfully!")
        df_metrics = pd.DataFrame.from_dict(metrics, orient="index")
        st.dataframe(df_metrics.style.format("{:.4f}"), use_container_width=True)

        best = max(metrics.keys(), key=lambda m: metrics[m]["accuracy"])
        st.success(f"🏆 Best Model: {best} ({metrics[best]['accuracy']*100:.2f}% accuracy)")

        st.balloons()
else:
    st.info("👆 Upload a CSV to start retraining.")

# ============================================================
# 🔄 Cache Control
# ============================================================
st.markdown("---")
st.subheader("🔄 Refresh Cache")

if st.button("Clear Cache & Reload"):
    st.cache_data.clear()
    st.success("✅ Cache cleared. Reload the dashboard to see updates.")
