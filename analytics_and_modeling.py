import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def load_or_create():
    if not os.path.exists("customer_data.csv"):
        from create_sample_data import generate_telecom_banking_data
        generate_telecom_banking_data()
    return pd.read_csv("customer_data.csv")

def run_pipeline():
    df = load_or_create()
    
    # Basit bir CLV Proxy metriği (Tenure * Monthly Spend)
    df["estimated_historical_value"] = df["tenure_months"] * df["monthly_spend"]
    
    features = [
        "tenure_months", 
        "monthly_spend", 
        "support_tickets_last_quarter", 
        "days_since_last_login", 
        "has_annual_contract",
        "estimated_historical_value"
    ]
    target = "churned"
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dengesiz veri için class_weight dengeli lojistik regresyon ve GBDT karşılaştırması
    models = {
        "Weighted Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, learning_rate=0.08, max_depth=3, random_state=42)
    }
    
    print("\n--- Model Benchmark Results ---")
    for name, model in models.items():
        if "Logistic" in name:
            model.fit(X_train_scaled, y_train)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)
            
        auc = roc_auc_score(y_test, probs)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
        
        print(f"\n[{name}]")
        print(f"ROC-AUC: {auc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

if __name__ == "__main__":
    run_pipeline()
