"""
generate_data.py
────────────────
Generates 50,000 synthetic customer records that mirror the
real-world patterns in Asurion's churn dataset:
  - ~12% annual churn rate
  - Behavioral, product, and financial features
  - Realistic correlations (high support tickets → more churn, etc.)

Run:
    python data/generate_data.py
Output:
    data/customers.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 50_000

def generate():
    # ── Demographics & Account ─────────────────────────────────
    tenure_months      = np.random.gamma(shape=2.5, scale=18, size=N).clip(1, 120).astype(int)
    plan_type          = np.random.choice(["Starter","Growth","Enterprise"], size=N, p=[0.45, 0.35, 0.20])
    contract_length    = np.random.choice(["month-to-month","one_year","two_year"], size=N, p=[0.55, 0.30, 0.15])
    monthly_charges    = np.where(plan_type=="Starter",
                            np.random.normal(35, 8, N),
                            np.where(plan_type=="Growth",
                                np.random.normal(65, 12, N),
                                np.random.normal(110, 20, N))).clip(15, 200)
    total_charges      = (monthly_charges * tenure_months * np.random.uniform(0.9, 1.1, N)).round(2)
    num_products       = np.random.choice([1, 2, 3, 4], size=N, p=[0.40, 0.35, 0.18, 0.07])

    # ── Behavioral signals ─────────────────────────────────────
    days_since_login   = np.random.exponential(scale=12, size=N).clip(0, 180).astype(int)
    avg_monthly_usage  = np.random.gamma(shape=3, scale=10, size=N).clip(0, 100).round(1)
    support_tickets    = np.random.poisson(lam=1.2, size=N).clip(0, 15)
    payment_failures   = np.random.poisson(lam=0.4, size=N).clip(0, 6)
    nps_score          = np.random.choice(range(0, 11), size=N,
                            p=[0.04,0.03,0.04,0.05,0.06,0.08,0.10,0.15,0.20,0.15,0.10])
    feature_usage_pct  = np.random.beta(a=2, b=3, size=N).round(3)  # 0–1

    # ── Churn probability (realistic business logic) ───────────
    churn_score = (
        0.30 * (days_since_login / 180)           # inactivity hurts
      + 0.20 * (support_tickets / 15)             # frustration signal
      + 0.15 * (payment_failures / 6)             # financial stress
      + 0.10 * ((10 - nps_score) / 10)            # unhappiness
      + 0.10 * (1 - feature_usage_pct)            # low engagement
      + 0.08 * (1 / np.maximum(tenure_months, 1)) # new customers riskier
      + 0.07 * np.where(contract_length=="month-to-month", 1, 0)  # no lock-in
      - 0.08 * (num_products / 4)                 # stickiness
      - 0.05 * (avg_monthly_usage / 100)          # active users less likely
    )
    # add noise
    churn_score += np.random.normal(0, 0.08, N)
    churn_score  = churn_score.clip(0, 1)

    # target ~12% churn
    threshold = np.percentile(churn_score, 88)
    churned   = (churn_score >= threshold).astype(int)

    # ── Assemble DataFrame ─────────────────────────────────────
    df = pd.DataFrame({
        "customer_id":        [f"CUST_{i:06d}" for i in range(N)],
        "tenure_months":      tenure_months,
        "plan_type":          plan_type,
        "contract_length":    contract_length,
        "monthly_charges":    monthly_charges.round(2),
        "total_charges":      total_charges,
        "num_products":       num_products,
        "days_since_login":   days_since_login,
        "avg_monthly_usage":  avg_monthly_usage,
        "support_tickets_90d": support_tickets,
        "payment_failures_6m": payment_failures,
        "nps_score":          nps_score,
        "feature_usage_pct":  feature_usage_pct,
        "churn_risk_score":   churn_score.round(4),
        "churned":            churned,
    })

    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/customers.csv", index=False)
    print(f"Generated {N:,} records")
    print(f"Churn rate: {churned.mean():.1%}")
    print(f"Saved → data/customers.csv")
    return df

if __name__ == "__main__":
    generate()
