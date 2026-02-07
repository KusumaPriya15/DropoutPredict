
import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- Path setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_processor import load_data, preprocess_data, prepare_features_and_target
from utils.visualizations import create_model_comparison_chart, create_confusion_matrix_plot
from utils.model_trainer import train_all_models  # auto-trains if models missing

# ============================================================
# ⚙️ Streamlit Page Configuration
# ============================================================
st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Model Performance Dashboard")
st.markdown("Comprehensive evaluation of **3 dropout risk prediction models** trained on the UDISE dataset.")

# ============================================================
# 📥 Load and Preprocess Data
# ============================================================
@st.cache_data
def load_and_process_data():
    df = load_data()
    if df is not None:
        df = preprocess_data(df)
    return df

df = load_and_process_data()
if df is None:
    st.error("❌ Failed to load data. Please ensure the dataset file exists.")
    st.stop()

# ============================================================
# 🧠 Load Pre-trained Models / Metrics
# ============================================================
st.sidebar.header("⚙️ Model Settings")

if not os.path.exists("models/metrics_summary.csv"):
    st.sidebar.warning("⚠️ No pre-trained models found. Training models now...")
    with st.spinner("Training all models... please wait ⏳"):
        _, metrics = train_all_models()
else:
    st.sidebar.success("✅ Pre-trained models loaded from disk.")
    metrics = pd.read_csv("models/metrics_summary.csv", index_col=0).to_dict(orient="index")

# ============================================================
# 📈 Display Model Performance
# ============================================================
if metrics:
    st.markdown("---")
    st.subheader("📊 Model Performance Summary")

    model_names = list(metrics.keys())
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("**Model**")
        for name in model_names:
            st.write(name)

    with col2:
        st.markdown("**Accuracy**")
        for name in model_names:
            st.write(f"{metrics[name]['accuracy']:.4f}")

    with col3:
        st.markdown("**Precision**")
        for name in model_names:
            st.write(f"{metrics[name]['precision']:.4f}")

    with col4:
        st.markdown("**Recall**")
        for name in model_names:
            st.write(f"{metrics[name]['recall']:.4f}")

    with col5:
        st.markdown("**F1-Score**")
        for name in model_names:
            st.write(f"{metrics[name]['f1_score']:.4f}")

    # ============================================================
    # 📉 Performance Comparison Chart
    # ============================================================
    st.markdown("---")
    st.subheader("📊 Performance Comparison Chart")
    comparison_fig = create_model_comparison_chart(metrics)
    st.plotly_chart(comparison_fig, use_container_width=True)

    # ============================================================
    # 🏆 Best Performing Model
    # ============================================================
    st.markdown("---")
    st.subheader("🏆 Best Performing Model")

    best_model_name = max(metrics.keys(), key=lambda x: metrics[x]['accuracy'])
    best_accuracy = metrics[best_model_name]['accuracy']

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
            <div style="background-color: #1f77b4; padding: 2rem; border-radius: 15px; text-align: center;">
                <h2 style="color: white; margin: 0;">{best_model_name}</h2>
                <h3 style="color: white; margin: 1rem 0;">Accuracy: {best_accuracy:.2%}</h3>
                <p style="color: white; margin: 0;">✅ Recommended for deployment</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ============================================================
    # 📋 Detailed Metrics Table
    # ============================================================
    st.markdown("---")
    st.subheader("📋 Detailed Metrics Table")

    df_metrics = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": f"{metrics[name]['accuracy']:.4f}",
            "Precision": f"{metrics[name]['precision']:.4f}",
            "Recall": f"{metrics[name]['recall']:.4f}",
            "F1-Score": f"{metrics[name]['f1_score']:.4f}",
            "ROC-AUC": f"{metrics[name]['roc_auc']:.4f}",
        }
        for name in model_names
    ])
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # ============================================================
    # 💡 Model Insights
    # ============================================================
    st.markdown("---")
    st.subheader("💡 Model Insights")

    st.markdown("""
    ### 🧠 Model Descriptions:
    - **Logistic Regression** — Interpretable baseline linear model.  
    - **XGBoost (Standalone)** — Powerful gradient boosting model optimized for multi-class dropout risk prediction.  
    - **CatXGB Ensemble (Proposed)** — Weighted ensemble (CatBoost 60% + XGBoost 40%) for highest accuracy.

    ### 📏 Metrics Explained:
    - **Accuracy** — Overall correct predictions ratio.  
    - **Precision** — Correct positive predictions ratio.  
    - **Recall** — Actual positives identified ratio.  
    - **F1-Score** — Balance between precision and recall.  
    - **ROC-AUC** — Discrimination power of the classifier.
    """)

else:
    st.warning("⚠️ Metrics not available. Ensure models are trained and saved properly.")
