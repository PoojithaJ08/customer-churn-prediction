"""
Churn Prediction Analytics Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import get_feature_matrix, engineer_features, encode_categoricals, DROP_COLS

st.set_page_config(page_title="Churn Prediction", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

BG, CARD, BORDER = "#080c14", "#111827", "#1f2937"
CYAN, RED, AMBER, GREEN, BLUE = "#22d3ee", "#f87171", "#fbbf24", "#4ade80", "#60a5fa"
TEXT, SUBTEXT, MUTED = "#f9fafb", "#9ca3af", "#4b5563"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
*, html, body {{ font-family: 'DM Sans', sans-serif !important; }}
.stApp {{ background: {BG}; }}
.block-container {{ padding: 2.5rem 3rem !important; max-width: 1440px !important; }}
#MainMenu, footer, header, .stDeployButton, div[data-testid="stToolbar"] {{ visibility: hidden; display: none; }}
.db-header {{ display:flex; align-items:flex-end; justify-content:space-between; margin-bottom:2.5rem; padding-bottom:1.5rem; border-bottom:1px solid {BORDER}; }}
.db-title {{ font-size:1.5rem; font-weight:700; color:{TEXT}; letter-spacing:-0.4px; }}
.db-title span {{ color:{CYAN}; }}
.db-meta {{ font-size:0.75rem; color:{MUTED}; font-family:'DM Mono',monospace; }}
.kpi-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1.25rem; margin-bottom:2.5rem; }}
.kpi {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:1.5rem 1.75rem; position:relative; }}
.kpi-accent {{ position:absolute; top:0; left:1.75rem; right:1.75rem; height:2px; border-radius:0 0 4px 4px; }}
.kpi-val {{ font-size:2.4rem; font-weight:700; color:{TEXT}; line-height:1; letter-spacing:-1px; margin:0.8rem 0 0.5rem; }}
.kpi-lbl {{ font-size:0.72rem; color:{SUBTEXT}; font-weight:500; text-transform:uppercase; letter-spacing:0.6px; }}
.kpi-tag {{ display:inline-block; margin-top:0.5rem; font-size:0.68rem; font-family:'DM Mono',monospace; color:{MUTED}; background:#0d1420; padding:2px 8px; border-radius:4px; border:1px solid {BORDER}; }}
.sl {{ font-size:0.68rem; font-weight:600; color:{MUTED}; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.6rem; font-family:'DM Mono',monospace; }}
.cc {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:1.5rem; margin-bottom:0; }}
.ct {{ font-size:0.9rem; font-weight:600; color:{TEXT}; margin-bottom:0.2rem; }}
.cd {{ font-size:0.73rem; color:{MUTED}; margin-bottom:1rem; line-height:1.4; }}
.div {{ height:1px; background:{BORDER}; margin:1.75rem 0; }}
.tab-bar {{ display:flex; gap:0.5rem; margin-bottom:1.5rem; flex-wrap:wrap; }}
.finding {{ background:#0d1420; border:1px solid {BORDER}; border-left:3px solid {CYAN}; border-radius:6px; padding:0.8rem 1rem; margin-bottom:0.6rem; font-size:0.82rem; color:{SUBTEXT}; }}
.finding b {{ color:{TEXT}; }}
</style>
""", unsafe_allow_html=True)

IMG = Path(__file__).parent / "images"

@st.cache_data
def load_data():
    df      = pd.read_csv(Path(__file__).parent.parent / "data/customers.csv")
    metrics = json.loads((Path(__file__).parent.parent / "models/metrics.json").read_text())
    return df, metrics

@st.cache_data
def score_customers(df_path):
    df    = pd.read_csv(df_path)
    model = joblib.load(Path(__file__).parent.parent / "models/churn_model.pkl")
    fnames= joblib.load(Path(__file__).parent.parent / "models/feature_names.pkl")
    df_f  = engineer_features(df)
    df_f  = encode_categoricals(df_f)
    drop  = [c for c in DROP_COLS if c in df_f.columns]
    X     = df_f.drop(columns=drop)
    for col in fnames:
        if col not in X.columns: X[col] = 0
    X     = X[fnames]
    proba = model.predict_proba(X)[:, 1]
    df    = df.copy()
    df["churn_probability"] = proba.round(4)
    df["risk_tier"] = pd.cut(proba, bins=[0,0.30,0.60,0.80,1.0],
        labels=["Low","Medium","High","Critical"], include_lowest=True)
    return df

try:
    df, metrics = load_data()
    scored      = score_customers(str(Path(__file__).parent.parent / "data/customers.csv"))
except Exception as e:
    st.error(f"Run `make all` first to generate data and train the model.\n\n{e}")
    st.stop()

at_risk  = scored[scored["churn_probability"] >= 0.60]
critical = scored[scored["risk_tier"] == "Critical"]
tc       = scored["risk_tier"].value_counts()

# ── Header ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="db-header">
  <div>
    <div class="db-title">Customer <span>Churn</span> Prediction</div>
    <div style="font-size:0.78rem;color:{MUTED};margin-top:0.3rem;">
      Random Forest · 50K customers · identify at-risk accounts 60 days before churn
    </div>
  </div>
  <div class="db-meta">sklearn {__import__('sklearn').__version__} · ROC-AUC {metrics['roc_auc']:.3f}</div>
</div>""", unsafe_allow_html=True)

# ── KPIs ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi"><div class="kpi-accent" style="background:{CYAN}"></div>
    <div class="kpi-lbl">Model Accuracy</div><div class="kpi-val">{metrics['accuracy']:.1%}</div>
    <div class="kpi-tag">ROC-AUC {metrics['roc_auc']:.3f}</div></div>
  <div class="kpi"><div class="kpi-accent" style="background:{RED}"></div>
    <div class="kpi-lbl">At-Risk Customers</div><div class="kpi-val">{len(at_risk):,}</div>
    <div class="kpi-tag">probability ≥ 60%</div></div>
  <div class="kpi"><div class="kpi-accent" style="background:{AMBER}"></div>
    <div class="kpi-lbl">Critical Risk</div><div class="kpi-val">{len(critical):,}</div>
    <div class="kpi-tag">act within 7 days</div></div>
  <div class="kpi"><div class="kpi-accent" style="background:{GREEN}"></div>
    <div class="kpi-lbl">Est. Annual Savings</div><div class="kpi-val">$500K+</div>
    <div class="kpi-tag">25% retention rate</div></div>
</div>""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Overview",
    "🔍  EDA",
    "🤖  Model",
    "💡  Insights",
    "🎯  At-Risk Customers",
])

# ─── TAB 1: Overview ───────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="sl">Churn Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="cc"><div class="ct">Overall Rate & By Plan Type</div><div class="cd">12% annual churn · Starter tier highest risk</div>', unsafe_allow_html=True)
        st.image(str(IMG / "churn_distribution.png"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="sl">Risk Breakdown</div>', unsafe_allow_html=True)
        st.markdown('<div class="cc"><div class="ct">Customer Risk Tier Distribution</div><div class="cd">Prioritize Critical + High for immediate outreach</div>', unsafe_allow_html=True)
        st.image(str(IMG / "risk_tiers.png"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sl">Business Impact</div>', unsafe_allow_html=True)
    st.markdown('<div class="cc"><div class="ct">Retention Funnel — From Churn to Savings</div><div class="cd">Model identifies 4,050 of 6,000 annual churners · 25% retention = $500K+ saved</div>', unsafe_allow_html=True)
    st.image(str(IMG / "business_impact.png"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─── TAB 2: EDA ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sl">Behavioral Signals</div>', unsafe_allow_html=True)
    st.markdown('<div class="cc"><div class="ct">Feature Distributions: Churned vs Retained</div><div class="cd">Churned customers show higher inactivity, more support tickets, lower NPS and feature usage</div>', unsafe_allow_html=True)
    st.image(str(IMG / "feature_distributions.png"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sl">Segment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="cc"><div class="ct">Contract Type & NPS Impact on Churn</div><div class="cd">Month-to-month customers churn at 2x annual · NPS detractors (0-6) have highest churn</div>', unsafe_allow_html=True)
    st.image(str(IMG / "contract_nps_analysis.png"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─── TAB 3: Model ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sl">Model Selection</div>', unsafe_allow_html=True)
    st.markdown('<div class="cc"><div class="ct">Algorithm Comparison — 5-Fold Cross Validation</div><div class="cd">Random Forest selected: best AUC balance vs training speed vs interpretability</div>', unsafe_allow_html=True)
    st.image(str(IMG / "model_comparison.png"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sl">Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="cc"><div class="ct">ROC Curve · Precision-Recall · Confusion Matrix</div><div class="cd">AUC 0.805 · tuned for high recall — catching churners matters more than false positives</div>', unsafe_allow_html=True)
    st.image(str(IMG / "model_evaluation.png"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─── TAB 4: Insights ───────────────────────────────────────────────────
with tab4:
    c1, c2 = st.columns([1.4, 1], gap="large")
    with c1:
        st.markdown('<div class="sl">Feature Importance</div>', unsafe_allow_html=True)
        st.markdown('<div class="cc"><div class="ct">Top 15 Churn Prediction Features</div><div class="cd">What the model learned — and what your team should focus on to reduce churn</div>', unsafe_allow_html=True)
        st.image(str(IMG / "feature_importance.png"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="sl">Business Interpretation</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cc">
          <div class="ct">What Each Signal Means</div>
          <div class="cd">Actionable insights for product & CS teams</div>
          <div class="finding"><b>NPS Score</b><br>Unhappy customers (NPS ≤ 6) are 3× more likely to churn. Trigger CS outreach when NPS drops below 7.</div>
          <div class="finding"><b>Month-to-Month Contract</b><br>No lock-in = lowest switching cost. Offer annual discount at month 3 to reduce risk.</div>
          <div class="finding"><b>Days Since Login</b><br>60+ days of inactivity = 4× churn probability. Trigger re-engagement email at day 30.</div>
          <div class="finding"><b>Feature Usage %</b><br>Low adoption = low perceived ROI. Assign CS rep when usage drops below 20%.</div>
          <div class="finding"><b>Support Tickets</b><br>3+ tickets in 90 days signals frustration before cancel. Flag for proactive outreach.</div>
          <div class="finding"><b>Payment Failures</b><br>Financial stress indicator. Offer payment flexibility or downgrade path before they cancel.</div>
        </div>""", unsafe_allow_html=True)

# ─── TAB 5: At-Risk Customers ──────────────────────────────────────────
with tab5:
    st.markdown('<div class="sl">Priority List</div>', unsafe_allow_html=True)
    st.markdown('<div class="cc"><div class="ct">Top 50 Highest-Risk Customers — Act Now</div><div class="cd">Sorted by churn probability · Critical = contact within 7 days · High = contact within 30 days</div>', unsafe_allow_html=True)

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        tier_filter = st.multiselect("Risk Tier", ["Critical","High","Medium","Low"], default=["Critical","High"])
    with col_f2:
        plan_filter = st.multiselect("Plan Type", ["Starter","Growth","Enterprise"], default=["Starter","Growth","Enterprise"])
    with col_f3:
        min_prob = st.slider("Min Churn Probability", 0.0, 1.0, 0.6, 0.05)

    filtered = scored[
        (scored["risk_tier"].isin(tier_filter)) &
        (scored["plan_type"].isin(plan_filter)) &
        (scored["churn_probability"] >= min_prob)
    ].nlargest(50, "churn_probability")

    display_cols = ["customer_id","plan_type","contract_length","monthly_charges",
                    "tenure_months","days_since_login","support_tickets_90d",
                    "nps_score","churn_probability","risk_tier"]
    show = filtered[display_cols].copy()
    show["churn_probability"] = show["churn_probability"].apply(lambda x: f"{x:.1%}")
    show["monthly_charges"]   = show["monthly_charges"].apply(lambda x: f"${x:.0f}")

    st.dataframe(show.reset_index(drop=True), use_container_width=True,
        column_config={
            "customer_id":         st.column_config.TextColumn("Customer"),
            "plan_type":           st.column_config.TextColumn("Plan"),
            "contract_length":     st.column_config.TextColumn("Contract"),
            "monthly_charges":     st.column_config.TextColumn("MRR"),
            "tenure_months":       st.column_config.NumberColumn("Tenure (mo)"),
            "days_since_login":    st.column_config.NumberColumn("Days Inactive"),
            "support_tickets_90d": st.column_config.NumberColumn("Tickets"),
            "nps_score":           st.column_config.NumberColumn("NPS"),
            "churn_probability":   st.column_config.TextColumn("Churn Prob"),
            "risk_tier":           st.column_config.TextColumn("Risk Tier"),
        })
    st.markdown(f"Showing **{len(filtered):,}** customers matching filters", unsafe_allow_html=False)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid {BORDER};
     display:flex;justify-content:space-between;align-items:center;">
  <span style="font-size:0.7rem;color:{MUTED};font-family:'DM Mono',monospace;">
    Random Forest · 5-fold CV · 50K records · AUC 0.805
  </span>
  <a href="https://github.com/PoojithaJ08/customer-churn-prediction"
     style="font-size:0.7rem;color:{MUTED};text-decoration:none;font-family:'DM Mono',monospace;">
    github ↗
  </a>
</div>""", unsafe_allow_html=True)
