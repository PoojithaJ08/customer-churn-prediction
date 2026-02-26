"""
predict.py
──────────
Scoring pipeline — takes new customer data and outputs churn
probability + risk tier for each customer.

Usage:
    python src/predict.py --input data/customers.csv --output data/scores.csv
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import get_feature_matrix, engineer_features, encode_categoricals, DROP_COLS, TARGET


def load_model():
    model        = joblib.load("models/churn_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Add churn_probability and risk_tier columns to df."""
    model, feature_names = load_model()

    # engineer + encode
    df_feat = engineer_features(df)
    df_feat = encode_categoricals(df_feat)

    drop = [c for c in DROP_COLS if c in df_feat.columns]
    X    = df_feat.drop(columns=drop)

    # align columns to training
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    proba = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["churn_probability"] = proba.round(4)
    df["risk_tier"] = pd.cut(
        proba,
        bins=[0, 0.30, 0.60, 0.80, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True,
    )
    df["days_to_act"] = np.where(
        proba >= 0.80, 7,
        np.where(proba >= 0.60, 30,
        np.where(proba >= 0.30, 60, 90))
    )
    return df.sort_values("churn_probability", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Score customers for churn risk")
    parser.add_argument("--input",  default="data/customers.csv")
    parser.add_argument("--output", default="data/scores.csv")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)

    print("Scoring...")
    scored = score(df)

    scored.to_csv(args.output, index=False)
    print(f"\nScored {len(scored):,} customers → {args.output}")
    print("\nRisk tier breakdown:")
    print(scored["risk_tier"].value_counts().to_string())

    critical = scored[scored["risk_tier"] == "Critical"]
    print(f"\nCritical risk customers: {len(critical):,}")
    print(f"  → Recommend immediate outreach within 7 days")


if __name__ == "__main__":
    main()
