# 🎓 UDISE+ School Dropout Risk Analysis Dashboard

A comprehensive Streamlit web application for predicting and analyzing school dropout risk across Indian states using the UDISE+ dataset (2019-2024). This application combines machine learning, time-series forecasting, and interactive visualizations to provide actionable insights for educational policy makers.

## 🌟 Features

### 📊 Overview Dashboard
- **Key Metrics**: National and state-level statistics
- **Interactive Maps**: Geographical visualization of dropout risk with color-coded markers
- **Trend Analysis**: Historical dropout rate trends from 2019-2024
- **Risk Distribution**: Visual breakdown of high/medium/low risk states
- **Dynamic Filtering**: Filter by year, state, and risk category

### 🔮 Dropout Forecast
- **Prophet-based Forecasting**: 2-3 year ahead predictions using Facebook's Prophet
- **Individual State Analysis**: Detailed forecasts with confidence intervals
- **All States Summary**: Comprehensive forecast overview with risk maps
- **Trend Detection**: Automatic identification of increasing/decreasing patterns

### 🤖 Model Performance
- **5 ML Models**:
  - Logistic Regression (baseline)
  - CatBoost (gradient boosting)
  - Hybrid Ensemble (voting classifier)
  - CatXStackNet (stacking ensemble)
  - CatXStackNet Ultra (advanced stacking)
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Confusion Matrices**: Visual representation of model predictions
- **Model Comparison**: Side-by-side performance analysis

### 🔍 Feature Importance
- **Importance Rankings**: Identify key factors influencing dropout risk
- **Visual Analysis**: Interactive bar charts of top features
- **Model-Specific Insights**: Compare importance across different models
- **Policy Recommendations**: Actionable insights based on important features

### 📤 Upload & Retrain
- **CSV Upload**: Upload new UDISE+ data
- **Data Validation**: Automatic checking for required columns
- **Dynamic Retraining**: Train all 5 models on new data
- **Real-time Updates**: Dashboard automatically reflects new predictions

## 🎯 Risk Classification

The application uses a strict 3-color coding system:

- 🔴 **High Risk**: Dropout Rate > 10%
- 🟠 **Medium Risk**: Dropout Rate 5-10%
- 🟢 **Low Risk**: Dropout Rate < 5%

## 📋 Dataset

- **Source**: UDISE+ (Unified District Information System for Education Plus)
- **Time Period**: 2019-2024
- **Coverage**: 37 Indian States and Union Territories
- **Features**: 40+ attributes including:
  - School infrastructure indicators
  - Teacher statistics
  - Demographic information
  - Social category enrolments
  - Facilities (electricity, internet, toilets, etc.)

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11+
- pip or uv package manager

### Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

Or using uv:
```bash
uv add pandas numpy plotly folium prophet scikit-learn catboost xgboost lightgbm matplotlib seaborn joblib streamlit-folium
```

### Run the Application

```bash
streamlit run app.py --server.port 5000
```

The application will be available at `http://localhost:5000`

## 📁 Project Structure

```
.
├── app.py                          # Main landing page
├── pages/
│   ├── 1_📊_Overview.py           # Overview dashboard
│   ├── 2_🔮_Forecast.py           # Dropout forecasting
│   ├── 3_🤖_Model_Performance.py  # ML model evaluation
│   ├── 4_🔍_Feature_Importance.py # Feature analysis
│   └── 5_📤_Upload_&_Retrain.py   # Data upload & retraining
├── utils/
│   ├── data_processor.py          # Data loading & preprocessing
│   ├── model_trainer.py           # ML model training & evaluation
│   ├── forecaster.py              # Prophet time-series forecasting
│   └── visualizations.py          # Plotly & Folium visualizations
├── data.csv                        # UDISE+ dataset
├── models/                         # Trained model files (generated)
├── .streamlit/
│   └── config.toml                # Streamlit configuration
└── README.md                       # This file
```

## 🧠 Machine Learning Models

### 1. Logistic Regression
- **Type**: Linear classifier
- **Advantages**: Fast, interpretable, good baseline
- **Use Case**: Quick predictions and feature interpretation

### 2. CatBoost
- **Type**: Gradient boosting on decision trees
- **Advantages**: High accuracy, handles categorical features well
- **Use Case**: Production deployment with mixed data types

### 3. Hybrid Ensemble
- **Type**: Voting classifier (LR + CatBoost + XGBoost)
- **Advantages**: Balanced predictions, reduces overfitting
- **Use Case**: Robust predictions across different scenarios

### 4. CatXStackNet
- **Type**: Stacking ensemble (3 base + 1 meta model)
- **Advantages**: Advanced ensemble learning
- **Use Case**: High-stakes predictions requiring accuracy

### 5. CatXStackNet Ultra
- **Type**: Advanced stacking (5 base + CatBoost meta)
- **Advantages**: Maximum accuracy, comprehensive learning
- **Use Case**: Research and benchmark comparisons

## 📊 Usage Guide

### 1. Explore Current Data
- Navigate to **Overview** to see current statistics
- Use filters to focus on specific years or states
- Examine the interactive map for geographical insights

### 2. View Forecasts
- Go to **Forecast** page
- Select a state for detailed predictions
- Review confidence intervals and trends

### 3. Evaluate Models
- Visit **Model Performance** page
- Train new models or load existing ones
- Compare accuracy, precision, recall, and F1-scores

### 4. Analyze Features
- Check **Feature Importance** page
- Identify key factors affecting dropout rates
- Use insights for policy recommendations

### 5. Upload New Data
- Navigate to **Upload & Retrain**
- Upload CSV file with same format as UDISE+ data
- Train models on new data
- Dashboard automatically updates with new predictions

## 🔧 Configuration

### Streamlit Settings
The `.streamlit/config.toml` file contains:
- Server configuration (port, address)
- Theme settings
- Performance optimizations

### Model Settings
Adjust model hyperparameters in `utils/model_trainer.py`:
- Learning rates
- Number of estimators
- Tree depths
- Ensemble configurations

## 📈 Performance Metrics

All models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive identification rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve

## 🛠️ Troubleshooting

### Models Not Found
- Navigate to **Model Performance** page
- Select "Train New Models"
- Wait 2-5 minutes for training to complete

### Data Loading Errors
- Ensure `data.csv` exists in the project root
- Check CSV format matches UDISE+ structure
- Verify all required columns are present

### Forecast Errors
- Ensure historical data has at least 2 years
- Check for missing values in dropout rate column
- Verify year column contains valid numeric years

## 📝 Citation

If you use this application in your research or policy work, please cite:

```
UDISE+ School Dropout Risk Analysis Dashboard (2025)
Data Source: UDISE+ (Unified District Information System for Education Plus)
Time Period: 2019-2024
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional forecasting models (ARIMA, LSTM)
- More visualization types
- API integration for real-time data
- Export functionality for reports
- Mobile-responsive design

## 📄 License

This project uses publicly available UDISE+ data for educational and research purposes.

## 🙏 Acknowledgments

- **UDISE+** for providing comprehensive educational data
- **Prophet** (Facebook) for time-series forecasting
- **Streamlit** for the web application framework
- **CatBoost, XGBoost, LightGBM** for ML algorithms

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on the repository
- Check the documentation in each page
- Review the troubleshooting section

---

**Built with ❤️ for improving educational outcomes across India**
