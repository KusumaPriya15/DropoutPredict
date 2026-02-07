
import streamlit as st
import os

st.title("AI-Driven Early Warning System for Predicting Student dropout")


# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #d3d3d3;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .risk-high {
        color: #FF4444;
        font-weight: bold;
    }
    .risk-medium {
        color: #FFA500;
        font-weight: bold;
    }
    .risk-low {
        color: #00CC66;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# MAIN PAGE UI
# ============================================================
def main():
    st.markdown('<h1 class="main-header">🎓 UDISE+ School Dropout Risk Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive Prediction & Analysis Dashboard (2019–2024)</p>', unsafe_allow_html=True)
    
    st.markdown("---")

    # ============================================================
    # INTRO SECTION
    # ============================================================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 About This Application")
        st.markdown("""
        This analytics platform uses the **UDISE+ dataset (2019–2024)** to analyze and predict **school dropout risk**
        across Indian states and union territories.

        **Core Capabilities**
        - AI-powered prediction
        - Time-series forecasting
        - Interactive dashboards & maps
        - Live retraining
        - E-learning support system
        """)

    with col2:
        st.markdown("### 🎯 Risk Classification System")
        st.markdown("""
        The dropout risk classification uses 3 tier thresholds:

        - <span class="risk-high">🔴 High Risk</span> — > 10%
        - <span class="risk-medium">🟠 Medium Risk</span> — 5–10%
        - <span class="risk-low">🟢 Low Risk</span> — < 5%

        Applied consistently across visualizations, forecasts, and ML predictions.
        """, unsafe_allow_html=True)

    st.markdown("---")


    # ============================================================
    # FEATURE GRID – 6 FEATURE BOXES
    # ============================================================
    st.markdown("### 🚀 Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-box">
        <h4>📈 Overview Dashboard</h4>
        <ul>
        <li>National & state-level metrics</li>
        <li>Interactive geographical maps</li>
        <li>Trend analysis (2019–2024)</li>
        <li>Dynamic filters</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
        <h4>🔮 Dropout Forecast</h4>
        <ul>
        <li>Prophet model predictions</li>
        <li>2–3 year projections</li>
        <li>Upper/Lower bounds</li>
        <li>State-level forecasting</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-box">
        <h4>🤖 ML Model Performance</h4>
        <ul>
        <li>3 trained ML models</li>
        <li>Accuracy & F1 score</li>
        <li>Confusion matrices</li>
        <li>ROC-AUC analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div class="feature-box">
        <h4>🔍 Feature Importance</h4>
        <ul>
        <li>Top predictive features</li>
        <li>Explainability per model</li>
        <li>Interpretation dashboard</li>
        <li>Normalized scoring</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)



    with col5:
        st.markdown("""
        <div class="feature-box">
        <h4>📤 Upload & Retrain</h4>
        <ul>
        <li>CSV upload support</li>
        <li>Automatic validation</li>
        <li>Live training pipeline</li>
        <li>Performance re-evaluation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


    # 🔥 NEW FEATURE BOX (Your Website — Solution B)
    with col6:
        st.markdown("""
        <div class="feature-box">
        <h4>🌐 Community Learning Hub</h4>
        <ul>
        <li>Adaptive learning platform</li>
        <li>GPT-generated quizzes</li>
        <li>Analytics for educators</li>
        </ul>
        <p style='margin-top:10px'>
        👉 <a href="https://api-dev-1.vercel.app/" target="_blank">Open E-Learning Platform</a>
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")


    # ============================================================
    # MACHINE LEARNING MODELS — UPDATED
    # ============================================================
    st.markdown("### 🧠 Machine Learning Models")

    models_info = """
    | Model | Description | Use Case |
    |-------|-------------|----------|
    | **Logistic Regression** | Linear baseline classifier | Fast & interpretable |
    | **XGBoost (Standalone)** | Gradient boosting for structured data | High accuracy, handles imbalance |
    | **CatXGB Ensemble (Proposed)** | 60% CatBoost + 40% XGBoost weighted | Best overall performance |
    """
    st.markdown(models_info)

    st.markdown("---")


    # ============================================================
    # DATASET DETAILS
    # ============================================================
    st.markdown("### 📚 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Coverage**
        - 📅 Years: 2019–2024
        - 🗺️ 37 States / UTs
        - 🔢 200+ Rows
        - 📊 40+ Indicators
        """)

    with col2:
        st.markdown("""
        **Core Metrics**
        - Enrollment
        - School count
        - Teacher availability
        - Infrastructure indicators
        - Gender & SC/ST/OBC breakdown
        """)

    st.markdown("---")


    # ============================================================
    # NAVIGATION INFO
    # ============================================================
    st.markdown("### 🧭 Navigation")
    st.info("""
👈 Use the Sidebar to move between pages:
- Overview
- Forecast
- Model Performance
- Feature Importance
- Upload & Retrain
- Community Learning Hub
    """)

    st.markdown("---")

    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>Built with Streamlit • Data from UDISE+ (2019–2024)</p>
    <p>© 2025 School Dropout Risk Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

