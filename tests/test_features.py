"""
test_features.py
────────────────
Unit tests for the feature engineering pipeline.
Run: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import engineer_features, encode_categoricals, get_feature_matrix


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customer_id":         ["CUST_000001", "CUST_000002"],
        "tenure_months":       [12, 36],
        "plan_type":           ["Starter", "Enterprise"],
        "contract_length":     ["month-to-month", "one_year"],
        "monthly_charges":     [35.0, 110.0],
        "total_charges":       [420.0, 3960.0],
        "num_products":        [1, 4],
        "days_since_login":    [5, 75],
        "avg_monthly_usage":   [45.0, 80.0],
        "support_tickets_90d": [0, 5],
        "payment_failures_6m": [0, 2],
        "nps_score":           [9, 3],
        "feature_usage_pct":   [0.8, 0.1],
        "churn_risk_score":    [0.1, 0.85],
        "churned":             [0, 1],
    })


def test_engineer_features_adds_columns(sample_df):
    result = engineer_features(sample_df)
    expected = [
        "charges_per_month", "is_high_value", "is_inactive_30d",
        "is_inactive_60d", "low_feature_adoption", "has_support_issues",
        "has_payment_issues", "composite_risk", "is_long_tenure",
        "is_multi_product", "is_annual_contract", "nps_detractor", "nps_promoter"
    ]
    for col in expected:
        assert col in result.columns, f"Missing column: {col}"


def test_inactive_flags(sample_df):
    result = engineer_features(sample_df)
    # Customer 1: 5 days → not inactive
    assert result.iloc[0]["is_inactive_30d"] == 0
    assert result.iloc[0]["is_inactive_60d"] == 0
    # Customer 2: 75 days → both inactive
    assert result.iloc[1]["is_inactive_30d"] == 1
    assert result.iloc[1]["is_inactive_60d"] == 1


def test_nps_flags(sample_df):
    result = engineer_features(sample_df)
    # Customer 1: NPS 9 → promoter
    assert result.iloc[0]["nps_promoter"] == 1
    assert result.iloc[0]["nps_detractor"] == 0
    # Customer 2: NPS 3 → detractor
    assert result.iloc[1]["nps_detractor"] == 1
    assert result.iloc[1]["nps_promoter"] == 0


def test_encode_categoricals_removes_original(sample_df):
    result = encode_categoricals(sample_df)
    assert "plan_type" not in result.columns
    assert "contract_length" not in result.columns


def test_encode_categoricals_adds_dummies(sample_df):
    result = encode_categoricals(sample_df)
    assert any(col.startswith("plan_type_") for col in result.columns)
    assert any(col.startswith("contract_length_") for col in result.columns)


def test_get_feature_matrix_returns_correct_shape(sample_df):
    X, y = get_feature_matrix(sample_df)
    assert X.shape[0] == 2
    assert y.shape[0] == 2
    assert "churned" not in X.columns
    assert "customer_id" not in X.columns
    assert "churn_risk_score" not in X.columns


def test_no_null_features(sample_df):
    X, y = get_feature_matrix(sample_df)
    assert X.isnull().sum().sum() == 0, "Feature matrix contains nulls"


def test_churn_rate_is_binary(sample_df):
    _, y = get_feature_matrix(sample_df)
    assert set(y.unique()).issubset({0, 1})
