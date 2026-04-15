# 🎓 UDISE+ School Dropout Risk Analysis Dashboard

A comprehensive Streamlit web application for predicting and analyzing school dropout risk across Indian states using the UDISE+ dataset (2019–2024). This application combines machine learning, time-series forecasting, and interactive visualizations to provide actionable insights for educational policy makers.


## 🌟 Features

### 📊 Overview Dashboard

* Key metrics (national & state level)
* Interactive maps with risk visualization
* Trend analysis (2019–2024)
* Risk distribution (High/Medium/Low)
* Dynamic filtering (year/state/category)


### 🔮 Dropout Forecast

* Prophet-based forecasting (2–3 years ahead)
* State-wise predictions with confidence intervals
* Trend detection (increasing/decreasing)


### 🤖 Model Performance

* Logistic Regression
* XGBoost
* Ensemble Model (XGBoost + CatBoost)

Metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

Includes confusion matrices and model comparison.


### 🔍 Feature Importance

* Key factors influencing dropout
* Visual feature rankings
* Policy insights


### 📤 Upload & Retrain

* Upload new CSV data
* Auto validation
* Retrain models dynamically
* Real-time dashboard updates


## 🎯 Risk Classification

* 🔴 High Risk: > 10%
* 🟠 Medium Risk: 5–10%
* 🟢 Low Risk: < 5%


## 📋 Dataset

* Source: UDISE+
* Years: 2019–2024
* Coverage: 37 States/UTs
* Features: Infrastructure, teachers, demographics, facilities, etc.


# 🚀 Streamlit App Execution Steps (DropoutPredict)

## 🔹 STEP 0: Prerequisites

* Python (3.9+)
* Git

```bash
python --version
pip --version
git --version
```


## 🔹 STEP 1: Clone Project

```bash
git clone https://github.com/KusumaPriya15/DropoutPredict.git
cd DropoutPredict
```


## 🔹 STEP 2: Create Virtual Environment

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```


## 🔹 STEP 3: Install Dependencies

```bash
pip install -r requirements.txt
```


## 🔹 STEP 4: Run App

```bash
streamlit run app.py
```


## 🌐 Access App

```
http://localhost:8501
```


# 🚀 VidyaSetu App Execution Steps

## 🔹 STEP 0: Prerequisites

* Node.js (v18+)
* Git

```bash
node -v
npm -v
git -v
```


## 🔹 STEP 1: Clone Project

```bash
git clone https://github.com/KusumaPriya15/Vidya-Setu.git
cd Vidya-Setu
```


## 🔹 STEP 2: Install Dependencies

```bash
npm install
```


## 🔹 STEP 3: Create `.env`

```bash
New-Item .env
```

Add:

```env
VITE_SUPABASE_URL=https://uzaeicbdbscljwavnpew.supabase.co
VITE_SUPABASE_ANON_KEY=your_key_here
```


## 🔹 STEP 4: Run Frontend

```bash
npm run dev
```

Open:

```
http://localhost:5173/
```


## 🔹 STEP 5: Run Backend (if needed)

```bash
cd server
npm install
npm start
```


## 🔹 STEP 6: Setup Database (Supabase)

Run in SQL Editor:

* schema.sql
* fix_rls_policies.sql
* supabase_trigger_fix.sql


## 🧠 Machine Learning Models

### 1. Logistic Regression

* Simple and interpretable baseline model

### 2. XGBoost

* High-performance gradient boosting model

### 3. Ensemble Model (XGBoost + CatBoost)

* Combines strengths of both models
* Improves accuracy and robustness


## 📁 Project Structure

```
.
├── app.py
├── pages/
├── utils/
├── data.csv
├── models/
├── .streamlit/
└── README.md
```


## 📊 Usage Guide

1. Explore dashboard (Overview)
2. View predictions (Forecast)
3. Compare models (Model Performance)
4. Analyze features (Feature Importance)
5. Upload data (Retrain)


## 🛠️ Troubleshooting

### Models not found

→ Train from Model Performance page

### Data errors

→ Check CSV format

### App not running

→ Ensure dependencies installed


## 🙏 Acknowledgments

* UDISE+ dataset
* Prophet
* Streamlit
* XGBoost & CatBoost


## 🎉 Final Note

Built to support **data-driven educational decisions in India 🇮🇳**
