

# ============================================================
# utils/model_trainer.py — Final Optimized Version
# Three Models: Logistic Regression, XGBoost, CatXGB Ensemble
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# ============================================================
# 1️⃣ Data Loading + Preprocessing
# ============================================================

def load_and_prepare_data(csv_path="data/UDISE_2019-24_Combined_YearFixed.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    if "Average Dropout Rate" not in df.columns:
        raise ValueError("❌ Missing 'Average Dropout Rate' column.")

    # --- Define Risk Categories ---
    def classify_dropout(rate):
        if rate < 5:
            return "Low"
        elif rate < 10:
            return "Medium"
        else:
            return "High"

    df["Risk Category"] = df["Average Dropout Rate"].apply(classify_dropout)
    le = LabelEncoder()
    df["Risk_Label"] = le.fit_transform(df["Risk Category"])

    # --- Feature Engineering ---
    X = df.select_dtypes(include=[np.number]).drop(columns=["Average Dropout Rate", "Risk_Label"], errors="ignore")
    y = df["Risk_Label"]

    # Clean column names for XGBoost safety
    X.columns = X.columns.str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)

    # Handle missing values
    X = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X), columns=X.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Standardization for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, le

# ============================================================
# 2️⃣ Evaluation Helper
# ============================================================

def evaluate_model(name, model, X_test, y_test):
    """Evaluate model and return performance metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        y_prob = model.predict_proba(X_test)
        roc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    except:
        roc = 0.0

    print(f"\n📊 {name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"🎯 Accuracy: {acc*100:.2f}%")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "roc_auc": roc}

# ============================================================
# 3️⃣ Train All Models (Logistic, XGBoost, CatXGB Ensemble)
# ============================================================

def train_all_models():
    X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, le = load_and_prepare_data()
    models, metrics = {}, {}

    # -------------------- Logistic Regression --------------------
    print("⚙️ Training Logistic Regression...")
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    log_model = LogisticRegression(
        max_iter=10000, class_weight=dict(zip(np.unique(y_train), cw)),
        solver="saga", random_state=42
    )
    log_model.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = log_model
    metrics["Logistic Regression"] = evaluate_model("Logistic Regression", log_model, X_test_scaled, y_test)

    # -------------------- XGBoost (Standalone) --------------------
    print("🚀 Training XGBoost Classifier (Standalone)...")

    xgb_model_standalone = XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=3,
        reg_lambda=2,
        gamma=0.3,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=1
    )
    xgb_model_standalone.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    models["XGBoost (Standalone)"] = xgb_model_standalone
    metrics["XGBoost (Standalone)"] = evaluate_model("XGBoost (Standalone)", xgb_model_standalone, X_test, y_test)

    # -------------------- CatXGB Ensemble --------------------
    print("🤖 Training CatXGB Ensemble (CatBoost + XGBoost)...")

    cat_model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=10,
        l2_leaf_reg=6,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        random_seed=42,
        auto_class_weights="Balanced",
        verbose=False
    )
    cat_model.fit(X_train, y_train)

    xgb_model_for_ensemble = XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,
        reg_lambda=3,
        random_state=42,
        n_jobs=-1
    )
    xgb_model_for_ensemble.fit(X_train, y_train)

    # Weighted ensemble prediction
    cat_probs = cat_model.predict_proba(X_test)
    xgb_probs = xgb_model_for_ensemble.predict_proba(X_test)
    ensemble_probs = 0.6 * cat_probs + 0.4 * xgb_probs
    y_pred_ensemble = np.argmax(ensemble_probs, axis=1)

    print("\n📊 CatXGB Ensemble — Classification Report:")
    print(classification_report(y_test, y_pred_ensemble, target_names=le.classes_))
    print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ensemble))
    acc = accuracy_score(y_test, y_pred_ensemble)
    print(f"🎯 Accuracy: {acc*100:.2f}%")

    metrics["CatXGB Ensemble (Final)"] = {
        "accuracy": acc,
        "precision": precision_score(y_test, y_pred_ensemble, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred_ensemble, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred_ensemble, average="weighted", zero_division=0),
        "roc_auc": 0.0
    }

    # -------------------- Save Models --------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(log_model, "models/logistic_regression.pkl")
    joblib.dump(xgb_model_standalone, "models/xgboost.pkl")  # new standalone model
    joblib.dump(cat_model, "models/catboost.pkl")
    joblib.dump(xgb_model_for_ensemble, "models/xgboost_part.pkl")  # ensemble partner
    joblib.dump(le, "models/label_encoder.pkl")

    pd.DataFrame.from_dict(metrics, orient="index").to_csv("models/metrics_summary.csv")
    print("\n✅ Training completed successfully. Models saved in /models/")
    return models, metrics
