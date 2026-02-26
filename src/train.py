"""
train.py
────────
Trains a Random Forest churn prediction model.
Outputs:
  - models/churn_model.pkl      trained model
  - models/feature_names.pkl    feature list
  - models/metrics.json         evaluation metrics

Run:
    python src/train.py
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import get_feature_matrix


def train():
    print("Loading data...")
    df = pd.read_csv("data/customers.csv")
    X, y = get_feature_matrix(df)

    print(f"Dataset: {len(df):,} records | Churn rate: {y.mean():.1%}")
    print(f"Features: {X.shape[1]}")

    # ── Train / Test split ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Random Forest (matches resume) ────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Final fit
    rf.fit(X_train, y_train)

    # ── Evaluation ─────────────────────────────────────────────
    y_pred      = rf.predict(X_test)
    y_proba     = rf.predict_proba(X_test)[:, 1]

    auc         = roc_auc_score(y_test, y_proba)
    ap          = average_precision_score(y_test, y_proba)
    report      = classification_report(y_test, y_pred, output_dict=True)
    accuracy    = report["accuracy"]

    print(f"\nTest Results:")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  ROC-AUC:   {auc:.3f}")
    print(f"  Avg Prec:  {ap:.3f}")
    print(f"  Precision (churn): {report['1']['precision']:.1%}")
    print(f"  Recall    (churn): {report['1']['recall']:.1%}")

    # ── Feature importance ─────────────────────────────────────
    importance_df = pd.DataFrame({
        "feature":   X.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).head(15)
    print(f"\nTop 5 features:")
    for _, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']:<35} {row['importance']:.3f}")

    # ── Business impact ────────────────────────────────────────
    # Flag top 20% risk as "at-risk" for marketing targeting
    all_proba   = rf.predict_proba(X)[:, 1]
    threshold   = np.percentile(all_proba, 80)
    at_risk     = (all_proba >= threshold).sum()
    print(f"\nBusiness Impact:")
    print(f"  At-risk customers flagged: {at_risk:,}")
    print(f"  (Top 20% churn probability — prioritize for retention campaigns)")

    # ── Save artifacts ─────────────────────────────────────────
    Path("models").mkdir(exist_ok=True)

    joblib.dump(rf, "models/churn_model.pkl")
    joblib.dump(list(X.columns), "models/feature_names.pkl")

    metrics = {
        "accuracy":         round(accuracy, 4),
        "roc_auc":          round(auc, 4),
        "average_precision": round(ap, 4),
        "cv_auc_mean":      round(cv_scores.mean(), 4),
        "cv_auc_std":       round(cv_scores.std(), 4),
        "precision_churn":  round(report["1"]["precision"], 4),
        "recall_churn":     round(report["1"]["recall"], 4),
        "at_risk_customers": int(at_risk),
        "total_customers":   int(len(df)),
        "churn_rate":        round(float(y.mean()), 4),
        "n_features":        int(X.shape[1]),
        "top_features":      importance_df[["feature","importance"]].to_dict("records"),
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print("  models/churn_model.pkl")
    print("  models/feature_names.pkl")
    print("  models/metrics.json")

    return rf, metrics


if __name__ == "__main__":
    train()
