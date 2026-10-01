# Customer Churn Risk & Retention Analytics Pipeline

A production-style analytical pipeline to predict user retention risk, evaluate retention economics, and identify key drivers behind customer churn.

## Key Objectives
- Address imbalanced churn distribution through cost-sensitive weighting.
- Compare interpretable baseline (Regularized Logistic Regression) against non-linear tree-based ensembles (Gradient Boosting).
- Derive customer historical value proxies to prioritize retention intervention for high-value accounts.

## Project Structure
- `create_sample_data.py`: Simulates behavioral engagement and subscription logs.
- `analytics_and_modeling.py`: Feature scaling, model fitting, and evaluation via ROC-AUC & F1-score.

## How to Run

```bash
pip install -r requirements.txt
python analytics_and_modeling.py
