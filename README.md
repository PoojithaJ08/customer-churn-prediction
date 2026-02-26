# Customer Churn Prediction Model

[![ML Pipeline CI](https://github.com/PoojithaJ08/customer-churn-prediction/actions/workflows/ml_ci.yml/badge.svg)](https://github.com/PoojithaJ08/customer-churn-prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.3+-orange.svg)](https://scikit-learn.org)

**Predict customer churn 60 days before it happens.**  
Random Forest classifier trained on 50,000 customer records — 12% baseline churn rate, 80+ AUC, $500K+ annual savings through targeted retention campaigns.

---

## The Business Problem

A 12% annual churn rate on 50K customers = **6,000 lost accounts per year**.  
Most retention teams act *after* a customer cancels — too late.

This model identifies at-risk customers **60 days before they churn** so marketing and CS teams can intervene with targeted offers, outreach, or product education.

```
Without model:  React after cancel  →  0% retention
With model:     Predict 60 days out →  25% retention improvement
Result:         ~1,000 customers saved annually → $500K+ in retained revenue
```

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC (5-fold CV) | **0.809 ± 0.005** |
| Test Accuracy | **75.9%** |
| Recall (churn) | **67.5%** — catches 2 in 3 churners |
| At-risk customers flagged | **10,000** (top 20% by risk) |
| Estimated annual savings | **$500K+** |

**Why recall over precision?**  
A false positive costs ~$10 (retention email). A missed churner costs ~$500 (lost LTV).  
The model is tuned to catch churners, accepting some false positives.

---

## How It Works

```
Raw customer data
       ↓
Feature engineering (29 features)
  - Behavioral: days_since_login, feature_usage_pct
  - Financial:  monthly_charges, payment_failures
  - Sentiment:  nps_score
  - Product:    num_products, contract_length
       ↓
Random Forest (300 trees, 5-fold CV)
       ↓
Churn probability score (0–1)
       ↓
Risk tier: Low / Medium / High / Critical
       ↓
Marketing team prioritizes Critical + High
```

---

## Top Churn Drivers

1. **NPS score** — unhappy customers are 3x more likely to churn
2. **Month-to-month contract** — no lock-in means easier to leave
3. **Days since last login** — disengaged = unaware of value
4. **Feature usage %** — low adoption = low perceived ROI
5. **Support tickets** — frustration signal before cancellation

---

## Quickstart

```bash
git clone https://github.com/PoojithaJ08/customer-churn-prediction
cd customer-churn-prediction
pip install -r requirements.txt

make all          # generate data → train → test → score
make dashboard    # launch dashboard at localhost:8501
```

Or step by step:
```bash
make data      # generate 50K synthetic customer records
make train     # train Random Forest, save to models/
make test      # run 8 unit tests
make score     # score all customers → data/scores.csv
```

---

## Project Structure

```
customer-churn-prediction/
├── data/
│   └── generate_data.py       # synthetic data generator (50K records)
├── src/
│   ├── features.py            # feature engineering pipeline
│   ├── train.py               # model training + evaluation
│   └── predict.py             # scoring pipeline
├── models/                    # saved model artifacts (git-ignored)
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── tests/
│   └── test_features.py       # 8 unit tests
├── docs/
│   └── model_card.md          # model documentation
├── .github/workflows/
│   └── ml_ci.yml              # CI: train + test on every push
├── Makefile                   # convenience commands
└── requirements.txt
```

---

## CI Pipeline

Every push to `main` automatically:
1. Generates synthetic data
2. Trains the model
3. Runs 8 unit tests
4. Scores all customers
5. Validates AUC ≥ 0.75 and accuracy ≥ 70%

---

## Model Card

See [`docs/model_card.md`](docs/model_card.md) for full documentation:
feature definitions, performance breakdown, limitations, and retraining guidance.

---

## Stack

`Python 3.11` · `scikit-learn` · `pandas` · `numpy` · `Streamlit` · `Plotly` · `pytest` · `GitHub Actions`

---

*See also: [saas-churn-plan-change-analysis](https://github.com/PoojithaJ08/saas-churn-plan-change-analysis) — SQL + dbt churn pipeline*
