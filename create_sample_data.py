import numpy as np
import pandas as pd

def generate_telecom_banking_data(n_users=1500):
    np.random.seed(42)
    
    customer_ids = [f"CUST_{10000 + i}" for i in range(n_users)]
    tenure_months = np.random.randint(1, 72, size=n_users)
    monthly_spend = np.random.uniform(20.0, 180.0, size=n_users)
    support_tickets = np.random.poisson(lam=1.5, size=n_users)
    
    # Kullanım sıklığı ve gecikme günleri
    days_since_last_login = np.random.exponential(scale=12.0, size=n_users).astype(int)
    has_contract = np.random.binomial(n=1, p=0.45, size=n_users)
    
    # Churn olasılığı (Lojistik fonksiyon simülasyonu)
    log_odds = (
        -1.8 
        - 0.03 * tenure_months 
        + 0.015 * monthly_spend 
        + 0.45 * support_tickets 
        + 0.05 * days_since_last_login 
        - 1.2 * has_contract
    )
    prob_churn = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.rand(n_users) < prob_churn).astype(int)
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure_months": tenure_months,
        "monthly_spend": np.round(monthly_spend, 2),
        "support_tickets_last_quarter": support_tickets,
        "days_since_last_login": days_since_last_login,
        "has_annual_contract": has_contract,
        "churned": churn
    })
    
    df.to_csv("customer_data.csv", index=False)
    print(f"Dataset generated. Churn rate: {df['churned'].mean():.2%}")

if __name__ == "__main__":
    generate_telecom_banking_data()
