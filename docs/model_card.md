# Model Card — Customer Churn Prediction

## Overview

| Attribute        | Value |
|------------------|-------|
| Model type       | Random Forest Classifier |
| Training data    | 50,000 customer records |
| Features         | 29 engineered features |
| Target           | Binary churn within 90 days |
| CV AUC           | 0.809 ± 0.005 |
| Test Accuracy    | ~76% |
| Training date    | 2026 |

## Intended Use

Identify customers at high risk of cancelling within the next 60–90 days so marketing and customer success teams can run targeted retention campaigns before churn occurs.

**Primary users:** Marketing, Customer Success, Revenue Operations  
**Out-of-scope:** Real-time predictions (batch scoring only), credit decisions, HR decisions

## Features

### Raw Features (from billing + product data)
| Feature | Type | Description |
|---|---|---|
| tenure_months | Numeric | Months as a customer |
| plan_type | Categorical | Starter / Growth / Enterprise |
| contract_length | Categorical | Month-to-month / Annual |
| monthly_charges | Numeric | Current MRR in USD |
| total_charges | Numeric | Lifetime spend |
| num_products | Numeric | Number of products subscribed |
| days_since_login | Numeric | Days since last product login |
| avg_monthly_usage | Numeric | Average monthly active usage (0–100) |
| support_tickets_90d | Numeric | Support tickets in last 90 days |
| payment_failures_6m | Numeric | Failed payments in last 6 months |
| nps_score | Numeric | Net Promoter Score (0–10) |
| feature_usage_pct | Numeric | % of features used (0–1) |

### Engineered Features (derived)
- `is_inactive_30d` / `is_inactive_60d` — inactivity flags
- `has_support_issues` — 3+ tickets in 90 days
- `has_payment_issues` — 2+ failures in 6 months
- `composite_risk` — sum of risk signals
- `nps_detractor` / `nps_promoter` — NPS segmentation
- `is_annual_contract` — contract stickiness flag
- `charges_per_month` — normalized revenue signal

## Performance

| Metric | Value |
|---|---|
| CV ROC-AUC (5-fold) | 0.809 ± 0.005 |
| Test ROC-AUC | 0.805 |
| Accuracy | 75.9% |
| Precision (churn=1) | 28.6% |
| Recall (churn=1) | 67.5% |

**Why is precision low?**  
Intentional. The model is tuned for high recall — we'd rather flag 3 false positives for every true positive than miss a customer who actually churns. The cost of a retention outreach ($5–10) is far lower than the cost of losing a customer ($500–2,000 LTV).

## Top Predictors

1. `nps_score` — unhappy customers churn
2. `is_annual_contract` — month-to-month customers at higher risk
3. `contract_length_month-to-month` — no lock-in = easier to leave
4. `days_since_login` — disengaged customers churn
5. `feature_usage_pct` — low adoption = low perceived value

## Business Impact

Given a 12% annual churn rate on 50,000 customers:
- ~6,000 customers churn per year
- Model identifies ~4,000 of those before they churn (67.5% recall)
- Retention campaigns at 25% success rate = ~1,000 customers retained
- Average customer LTV = $500 → **$500K annual savings**

## Limitations

- Trained on synthetic data — requires retraining on production data before deployment
- No real-time scoring — batch pipeline runs nightly
- Does not account for macroeconomic conditions or product changes
- NPS data often missing in practice — model degrades without it

## Retraining

Retrain monthly. Trigger immediate retraining if:
- AUC drops below 0.75 on holdout set
- Churn rate changes by more than 3 percentage points
- New features become available (e.g., support sentiment scores)
