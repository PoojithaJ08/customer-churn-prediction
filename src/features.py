"""
features.py
───────────
Feature engineering pipeline for churn prediction.
Transforms raw customer data into model-ready features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


CATEGORICAL_COLS = ["plan_type", "contract_length"]
NUMERIC_COLS = [
    "tenure_months", "monthly_charges", "total_charges", "num_products",
    "days_since_login", "avg_monthly_usage", "support_tickets_90d",
    "payment_failures_6m", "nps_score", "feature_usage_pct",
]
TARGET = "churned"
DROP_COLS = ["customer_id", "churn_risk_score", TARGET]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve model signal."""
    df = df.copy()

    # Revenue signals
    df["charges_per_month"]     = (df["total_charges"] / df["tenure_months"].clip(1)).round(2)
    df["is_high_value"]         = (df["monthly_charges"] > df["monthly_charges"].quantile(0.75)).astype(int)

    # Engagement signals
    df["is_inactive_30d"]       = (df["days_since_login"] > 30).astype(int)
    df["is_inactive_60d"]       = (df["days_since_login"] > 60).astype(int)
    df["low_feature_adoption"]  = (df["feature_usage_pct"] < 0.2).astype(int)

    # Risk signals
    df["has_support_issues"]    = (df["support_tickets_90d"] >= 3).astype(int)
    df["has_payment_issues"]    = (df["payment_failures_6m"] >= 2).astype(int)
    df["composite_risk"]        = (
        df["has_support_issues"] +
        df["has_payment_issues"] +
        df["is_inactive_60d"] +
        df["low_feature_adoption"]
    )

    # Loyalty signals
    df["is_long_tenure"]        = (df["tenure_months"] > 24).astype(int)
    df["is_multi_product"]      = (df["num_products"] >= 3).astype(int)
    df["is_annual_contract"]    = (df["contract_length"] != "month-to-month").astype(int)

    # NPS segmentation
    df["nps_detractor"]         = (df["nps_score"] <= 6).astype(int)
    df["nps_promoter"]          = (df["nps_score"] >= 9).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    return df


def get_feature_matrix(df: pd.DataFrame):
    """Return X (features) and y (target) ready for sklearn."""
    df = engineer_features(df)
    df = encode_categoricals(df)

    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)
    y = df[TARGET] if TARGET in df.columns else None

    return X, y


def get_feature_names(df: pd.DataFrame) -> list:
    """Return the feature names after engineering + encoding."""
    X, _ = get_feature_matrix(df)
    return list(X.columns)
